#include "gpirt.h"
#include "mvnormal.h"

inline void draw_beta_ess_sparse_threadsafe(arma::vec& out,
                                            const arma::vec& beta, 
                                            const arma::vec& f, 
                                            const arma::vec& y, 
                                            const arma::mat& cholS,
                                            const arma::mat& X, 
                                            const arma::vec& thresholds,
                                            const arma::uvec& obs_idx,
                                            Workspace& ws) {
    arma::uword n_beta = beta.n_elem;
    
    // Use thread-safe RNG with pre-allocated workspace
    ws.nu.head(n_beta) = rmvnorm_threadsafe(cholS, ws.rng);
    
    double u = ws.rng.runif();
    double log_y = ll_bar_sparse(f, y, X*beta, thresholds, obs_idx) + std::log(u);
    
    bool reject = true;
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = ws.rng.runif(epsilon_min, epsilon_max);
    epsilon_min = epsilon - M_2PI;
    
    while (reject) {
        double cos_eps = std::cos(epsilon);
        double sin_eps = std::sin(epsilon);
        for (arma::uword i = 0; i < n_beta; ++i) {
            ws.beta_prime(i) = beta(i) * cos_eps + ws.nu(i) * sin_eps;
        }
        
        if (ll_bar_sparse(f, y, X*ws.beta_prime.head(n_beta), thresholds, obs_idx) > log_y) {
            reject = false;
        }
        else {
            if (epsilon < 0.0) {
                epsilon_min = epsilon;
            }
            else {
                epsilon_max = epsilon;
            }
            epsilon = ws.rng.runif(epsilon_min, epsilon_max);
        }
    }
    
    out = ws.beta_prime.head(n_beta);
}

void draw_beta(arma::cube& result, const arma::cube& beta, const arma::cube& X,
               const arma::cube& y, const arma::cube& f,
               const arma::mat& prior_means, const arma::mat& prior_sds,
               const arma::cube& thresholds,
               const arma::field<arma::uvec>& obs_persons,
               WorkspacePool& ws_pool) {
    arma::uword p = beta.n_rows;
    arma::uword m = beta.n_cols;
    arma::uword horizon = beta.n_slices;

    for (arma::uword h = 0; h < horizon; h++) {
        // Parallelize over items
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (arma::uword j = 0; j < m; ++j) {
            int tid = get_thread_id();
            Workspace& ws = ws_pool.get(tid);
            
            // Use const reference to avoid copy
            const arma::uvec& obs_idx = obs_persons(j, h);
            
            if(obs_idx.n_elem > 0) {
                // Extract observed data using subviews where possible
                arma::vec f_obs = f.slice(h).col(j).elem(obs_idx);
                arma::vec y_obs = y.slice(h).col(j).elem(obs_idx);
                arma::mat X_obs = X.slice(h).rows(obs_idx);
                
                // Compute Cholesky using workspace (reuse cholS_small)
                ws.cholS_small.zeros();
                for (arma::uword i = 0; i < 3; ++i) {
                    ws.cholS_small(i, i) = prior_sds(i, j) * prior_sds(i, j) + 1e-6;
                }
                ws.cholS_small = arma::chol(ws.cholS_small, "lower");
                
                // Create valid indices for the extracted observations
                arma::uvec valid_idx = arma::regspace<arma::uvec>(0, obs_idx.n_elem - 1);
                
                arma::vec beta_new(p);
                draw_beta_ess_sparse_threadsafe(beta_new,
                    beta.slice(h).col(j), f_obs, y_obs, ws.cholS_small, X_obs, 
                    thresholds.slice(h).row(j).t(), valid_idx, ws);
                    
                result.slice(h).col(j) = beta_new;
            } else {
                result.slice(h).col(j) = beta.slice(h).col(j);
            }
        }
    }
}