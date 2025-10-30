#include "gpirt.h"
#include "mvnormal.h"

inline arma::vec draw_beta_ess_sparse(const arma::vec& beta, 
                                      const arma::vec& f, 
                                      const arma::vec& y, 
                                      const arma::mat& cholS,
                                      const arma::mat& X, 
                                      const arma::vec& thresholds,
                                      const arma::uvec& obs_idx) {
    arma::uword n = 2;
    arma::vec nu = rmvnorm(cholS);
    double u = R::runif(0.0, 1.0);
    
    // Only compute likelihood for observed data
    double log_y = ll_bar_sparse(f, y, X*beta, thresholds, obs_idx) + std::log(u);
    
    bool reject = true;
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = R::runif(epsilon_min, epsilon_max);
    epsilon_min = epsilon - M_2PI;
    arma::vec beta_prime(n);
    
    while ( reject ) {
        beta_prime = beta * std::cos(epsilon) + nu * std::sin(epsilon);
        // Only compute likelihood for observed data in acceptance check
        if ( ll_bar_sparse(f, y, X*beta_prime, thresholds, obs_idx) > log_y ) {
            reject = false;
        }
        else {
            if ( epsilon < 0.0 ) {
                epsilon_min = epsilon;
            }
            else {
                epsilon_max = epsilon;
            }
            epsilon = R::runif(epsilon_min, epsilon_max);
        }
    }
    return beta_prime;
}

arma::cube draw_beta(arma::cube& beta, const arma::cube& X,
                    const arma::cube& y, const arma::cube& f,
                    const arma::mat& prior_means, const arma::mat& prior_sds,
                    const arma::cube& thresholds,
                    const arma::field<arma::uvec>& obs_persons) {
    // Bookkeeping variables
    arma::uword p = beta.n_rows; // # of mean function variables (3 now)
    arma::uword m = beta.n_cols; // # of response functions we are learning
    arma::uword horizon = beta.n_slices; // # of time periods

    // Setup result object
    arma::cube result(p, m, horizon);

    // Update coefficients (ess) one at a time
    for (arma::uword h = 0; h < horizon; h++){
        for ( arma::uword j = 0; j < m; ++j ) {
            // Get observed respondents for this item/horizon
            arma::uvec obs_idx = obs_persons(j, h);
            
            // Only process if there are observed values
            if(obs_idx.n_elem > 0) {
                // Extract only observed data
                arma::vec f_obs = f.slice(h).col(j);
                arma::vec y_obs = y.slice(h).col(j);
                arma::mat X_obs = X.slice(h);
                
                // Keep only observed rows
                f_obs = f_obs(obs_idx);
                y_obs = y_obs(obs_idx);
                X_obs = X_obs.rows(obs_idx);
                
                arma::mat cholS(3,3, arma::fill::zeros);
                cholS.diag() = prior_sds.col(j);
                cholS = arma::powmat(cholS,2);
                cholS.diag() += 1e-6;
                cholS = arma::chol(cholS, "lower");
                
                // Create index for sparse function (all observed are valid)
                arma::uvec valid_idx = arma::linspace<arma::uvec>(0, obs_idx.n_elem-1, obs_idx.n_elem);
                
                result.slice(h).col(j) = draw_beta_ess_sparse(beta.slice(h).col(j),
                                                              f_obs, y_obs, cholS, X_obs, 
                                                              thresholds.slice(h).row(j).t(),
                                                              valid_idx);
            } else {
                // No observed data, keep prior
                result.slice(h).col(j) = beta.slice(h).col(j);
            }
        }
    }
    
    return result;
}
