#include "gpirt.h"
#include "mvnormal.h"

// Sparse ESS function with thread-safe RNG
arma::vec ess_sparse_threadsafe(const arma::vec& f, const arma::vec& y, const arma::mat& cholS,
                                const arma::vec& mu, const arma::vec& thresholds,
                                const arma::uvec& obs_idx, Workspace& ws) {
    arma::uword n = f.n_elem;
    
    // Use workspace with thread-safe RNG
    ws.nu.set_size(n);
    ws.nu = rmvnorm_threadsafe(cholS, ws.rng);
    
    double u = ws.rng.runif();
    double log_y = ll_bar_sparse(f, y, mu, thresholds, obs_idx) + std::log(u);
    
    bool reject = true;
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = ws.rng.runif(epsilon_min, epsilon_max);
    epsilon_min = epsilon - M_2PI;
    
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
            epsilon = ws.rng.runif(epsilon_min, epsilon_max);
        }
    }
    return ws.f_prime;
}

// Draw f for a single horizon slice - can be called in parallel for different items
inline void draw_f_item(arma::vec& result, const arma::vec& f_col, const arma::vec& y_col,
                        const arma::mat& cholS, const arma::vec& mu_col, 
                        const arma::vec& thresholds_row, const arma::uvec& obs_idx,
                        Workspace& ws) {
    result = ess_sparse_threadsafe(f_col, y_col, cholS, mu_col, thresholds_row, obs_idx, ws);
}

arma::cube draw_f(const arma::cube& f, const arma::mat& theta, const arma::cube& y, 
                  CholeskyCache& chol_cache, const arma::mat& beta_prior_sds, 
                  const arma::cube& mu, const arma::cube& thresholds, 
                  const int constant_IRF,
                  const arma::field<arma::uvec>& obs_persons,
                  WorkspacePool& ws_pool) {
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::uword horizon = f.n_slices;
    arma::cube result(n, m, horizon);

    // Update cache if needed
    update_cholesky_cache(chol_cache, theta, beta_prior_sds, 0, 0, "");

    if (constant_IRF == 0) {
        for (arma::uword h = 0; h < horizon; ++h) {
            // Get the Cholesky factor for this horizon
            const arma::mat& L_h = chol_cache.L.slice(h);
            
            // Parallelize over items
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for (arma::uword j = 0; j < m; ++j) {
                int tid = get_thread_id();
                Workspace& ws = ws_pool.get(tid);
                
                arma::vec result_col(n);
                draw_f_item(result_col, f.slice(h).col(j), y.slice(h).col(j), 
                           L_h, mu.slice(h).col(j), thresholds.slice(h).row(j).t(),
                           obs_persons(j, h), ws);
                result.slice(h).col(j) = result_col;
            }
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
        
        // Parallelize over items
        arma::mat f_prime(n*horizon, m);
        
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (arma::uword j = 0; j < m; ++j) {
            int tid = get_thread_id();
            Workspace& ws = ws_pool.get(tid);
            
            arma::vec result_col(n*horizon);
            draw_f_item(result_col, f_constant.col(j), y_constant.col(j),
                       L_constant, mu_constant.col(j), thresholds.slice(0).row(j).t(),
                       obs_persons_constant(j), ws);
            f_prime.col(j) = result_col;
        }

        for (arma::uword h = 0; h < horizon; ++h) {
            for (arma::uword j = 0; j < m; ++j) {
                result.slice(h).col(j) = f_prime.col(j).subvec(h*n, (h+1)*n-1);
            }
        }
    }
    
    return result;
}