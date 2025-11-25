#include "gpirt.h"
#include "mvnormal.h"

inline arma::vec draw_beta_ess_sparse_ws(const arma::vec& beta, 
                                         const arma::vec& f, 
                                         const arma::vec& y, 
                                         const arma::mat& cholS,
                                         const arma::mat& X, 
                                         const arma::vec& thresholds,
                                         const arma::uvec& obs_idx,
                                         Workspace& ws) {
    arma::uword n = 2;
    
    // Use pre-allocated workspace
    ws.nu.set_size(n);
    ws.nu = rmvnorm(cholS);
    
    double u = R::runif(0.0, 1.0);
    double log_y = ll_bar_sparse(f, y, X*beta, thresholds, obs_idx) + std::log(u);
    
    bool reject = true;
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = R::runif(epsilon_min, epsilon_max);
    epsilon_min = epsilon - M_2PI;
    
    // Use pre-allocated workspace
    ws.beta_prime.set_size(n);
    
    while (reject) {
        ws.beta_prime = beta * std::cos(epsilon) + ws.nu * std::sin(epsilon);
        if (ll_bar_sparse(f, y, X*ws.beta_prime, thresholds, obs_idx) > log_y) {
            reject = false;
        }
        else {
            if (epsilon < 0.0) {
                epsilon_min = epsilon;
            }
            else {
                epsilon_max = epsilon;
            }
            epsilon = R::runif(epsilon_min, epsilon_max);
        }
    }
    return ws.beta_prime;
}

arma::cube draw_beta(arma::cube& beta, const arma::cube& X,
                    const arma::cube& y, const arma::cube& f,
                    const arma::mat& prior_means, const arma::mat& prior_sds,
                    const arma::cube& thresholds,
                    const arma::field<arma::uvec>& obs_persons,
                    Workspace& ws) {
    arma::uword p = beta.n_rows;
    arma::uword m = beta.n_cols;
    arma::uword horizon = beta.n_slices;
    arma::cube result(p, m, horizon);

    for (arma::uword h = 0; h < horizon; h++){
        for ( arma::uword j = 0; j < m; ++j ) {
            arma::uvec obs_idx = obs_persons(j, h);
            
            if(obs_idx.n_elem > 0) {
                arma::vec f_obs = f.slice(h).col(j);
                arma::vec y_obs = y.slice(h).col(j);
                arma::mat X_obs = X.slice(h);
                
                f_obs = f_obs(obs_idx);
                y_obs = y_obs(obs_idx);
                X_obs = X_obs.rows(obs_idx);
                
                arma::mat cholS(3,3, arma::fill::zeros);
                cholS.diag() = prior_sds.col(j);
                cholS = arma::powmat(cholS,2);
                cholS.diag() += 1e-6;
                cholS = arma::chol(cholS, "lower");
                
                arma::uvec valid_idx = arma::linspace<arma::uvec>(0, obs_idx.n_elem-1, obs_idx.n_elem);
                
                result.slice(h).col(j) = draw_beta_ess_sparse_ws(beta.slice(h).col(j),
                                                                 f_obs, y_obs, cholS, X_obs, 
                                                                 thresholds.slice(h).row(j).t(),
                                                                 valid_idx, ws);
            } else {
                result.slice(h).col(j) = beta.slice(h).col(j);
            }
        }
    }
    
    return result;
}