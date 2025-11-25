#include "gpirt.h"
#include "mvnormal.h"

// Sparse ESS function with pre-allocated workspace
arma::vec ess_sparse_ws(const arma::vec& f, const arma::vec& y, const arma::mat& cholS,
                        const arma::vec& mu, const arma::vec& thresholds,
                        const arma::uvec& obs_idx, Workspace& ws) {
    arma::uword n = f.n_elem;
    
    // Use pre-allocated workspace
    ws.nu.set_size(n);
    ws.nu = rmvnorm(cholS);
    
    double u = R::runif(0.0, 1.0);
    double log_y = ll_bar_sparse(f, y, mu, thresholds, obs_idx) + std::log(u);
    
    bool reject = true;
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = R::runif(epsilon_min, epsilon_max);
    epsilon_min = epsilon - M_2PI;
    
    // Use pre-allocated f_prime
    ws.f_prime.set_size(n);
    
    while (reject) {
        ws.f_prime = f * std::cos(epsilon) + ws.nu * std::sin(epsilon);
        if (ll_bar_sparse(ws.f_prime, y, mu, thresholds, obs_idx) > log_y) {
            reject = false;
        } else {
            if (epsilon < 0.0) {
                epsilon_min = epsilon;
            } else {
                epsilon_max = epsilon;
            }
            epsilon = R::runif(epsilon_min, epsilon_max);
        }
    }
    return ws.f_prime;
}

// Updated draw_f_ with workspace
inline arma::mat draw_f_ws(const arma::mat& f, const arma::mat& y, const arma::mat& cholS,
                           const arma::mat& mu, const arma::mat& thresholds,
                           const arma::field<arma::uvec>& obs_persons_h, Workspace& ws) {
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::mat result(n, m);
    
    for (arma::uword j = 0; j < m; ++j) {
        result.col(j) = ess_sparse_ws(f.col(j), y.col(j), cholS, mu.col(j), 
                                      thresholds.row(j).t(), obs_persons_h(j), ws);
    }
    return result;
}

arma::cube draw_f(const arma::cube& f, const arma::mat& theta, const arma::cube& y, 
                  CholeskyCache& chol_cache, const arma::mat& beta_prior_sds, 
                  const arma::cube& mu, const arma::cube& thresholds, 
                  const int constant_IRF,
                  const arma::field<arma::uvec>& obs_persons,
                  Workspace& ws) {
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::uword horizon = f.n_slices;
    arma::cube result(n, m, horizon);

    // Update cache if needed
    update_cholesky_cache(chol_cache, theta, beta_prior_sds, 0, 0, "");

    if (constant_IRF == 0) {
        for (arma::uword h = 0; h < horizon; ++h) {
            arma::field<arma::uvec> obs_persons_h(m);
            for (arma::uword j = 0; j < m; ++j) {
                obs_persons_h(j) = obs_persons(j, h);
            }
            
            // Use cached Cholesky factor
            result.slice(h) = draw_f_ws(f.slice(h), y.slice(h), chol_cache.L.slice(h),
                                        mu.slice(h), thresholds.slice(h), obs_persons_h, ws);
        }
    } else {
        // For constant IRF case, build combined data
        arma::mat f_constant(n*horizon, m);
        arma::mat y_constant(n*horizon, m);
        arma::mat mu_constant(n*horizon, m);
        arma::vec theta_constant(n*horizon);
        
        arma::field<arma::uvec> obs_persons_constant(m);
        for (arma::uword j = 0; j < m; ++j) {
            arma::uvec all_obs;
            for (arma::uword h = 0; h < horizon; ++h) {
                arma::uvec h_obs = obs_persons(j, h);
                all_obs = arma::join_cols(all_obs, h_obs + h*n);
            }
            obs_persons_constant(j) = all_obs;
        }
        
        for (arma::uword h = 0; h < horizon; h++) {
            theta_constant.subvec(h*n, (h+1)*n-1) = theta.col(h);
            for (arma::uword j = 0; j < m; ++j) {
                f_constant.col(j).subvec(h*n, (h+1)*n-1) = f.slice(h).col(j);
                y_constant.col(j).subvec(h*n, (h+1)*n-1) = y.slice(h).col(j);
                mu_constant.col(j).subvec(h*n, (h+1)*n-1) = mu.slice(h).col(j);
            }
        }
        
        // Compute combined Cholesky once
        arma::mat S_constant = K(theta_constant, theta_constant, beta_prior_sds.col(0));
        S_constant.diag() += 1e-6;
        arma::mat L_constant = arma::chol(S_constant, "lower");
        
        arma::mat f_prime = draw_f_ws(f_constant, y_constant, L_constant, 
                                      mu_constant, thresholds.slice(0), 
                                      obs_persons_constant, ws);

        for (arma::uword h = 0; h < horizon; ++h) {
            for (arma::uword j = 0; j < m; ++j) {
                result.slice(h).col(j) = f_prime.col(j).subvec(h*n, (h+1)*n-1);
            }
        }
    }
    
    return result;
}