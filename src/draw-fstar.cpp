#include "gpirt.h"
#include "mvnormal.h"

void draw_fstar(arma::cube& results, const arma::cube& f, 
                const arma::mat& theta,
                const arma::vec& theta_star, 
                const arma::mat& beta_prior_sds,
                CholeskyCache& chol_cache,
                const arma::cube& mu_star,
                const int constant_IRF,
                WorkspacePool& ws_pool) {
    arma::uword n = f.n_rows;
    arma::uword horizon = f.n_slices;
    arma::uword m = f.n_cols;
    arma::uword N = theta_star.n_elem;
    
    // Zero out results (reusing pre-allocated memory)
    results.zeros();
    
    if (constant_IRF == 0) {
        for (arma::uword h = 0; h < horizon; ++h) {
            // Pre-compute common terms for this horizon
            const arma::mat& L = chol_cache.L.slice(h);
            
            // Compute kstar - allocate once per horizon, not per item
            arma::mat kstar = K(theta.col(h), theta_star, beta_prior_sds.col(0));
            arma::mat kstarT = kstar.t();
            
            // Compute tmp_common = L^{-1} * kstar (shared across items)
            arma::mat tmp_common = arma::solve(arma::trimatl(L), kstar);
            
            // Compute posterior covariance (shared across items)
            arma::mat K_post = K(theta_star, theta_star, beta_prior_sds.col(0));
            K_post -= tmp_common.t() * tmp_common;  // In-place subtraction
            K_post.diag() += 1e-6;
            arma::mat L_post = arma::chol(K_post, "lower");
            
            // Pre-allocate alpha vector for this horizon (reused across items)
            arma::vec alpha(n);
            arma::vec draw_mean(N);
            
            // Parallelize over items
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic) firstprivate(alpha, draw_mean)
            #endif
            for (arma::uword j = 0; j < m; ++j) {
                int tid = get_thread_id();
                Workspace& ws = ws_pool.get(tid);
                
                // Compute item-specific mean
                alpha = double_solve(L, f.slice(h).col(j));
                draw_mean = kstarT * alpha + mu_star.slice(h).col(j);
                
                // Draw from posterior
                results.slice(h).col(j) = draw_mean + rmvnorm_threadsafe(L_post, ws.rng);
            }
        }
    } else {
        // Constant IRF case with inducing points
        arma::uword n_total = n * horizon;
        
        // Build combined data - allocate once
        arma::mat f_constant_all(n_total, m);
        arma::vec theta_constant_all(n_total);
        
        for (arma::uword h = 0; h < horizon; h++) {
            theta_constant_all.subvec(h*n, (h+1)*n-1) = theta.col(h);
            for (arma::uword j = 0; j < m; ++j) {
                f_constant_all.col(j).subvec(h*n, (h+1)*n-1) = f.slice(h).col(j);
            }
        }
        
        // Use inducing points for efficiency
        int n_induced_points = 100;
        arma::vec theta_constant = arma::linspace(theta.min(), theta.max(), n_induced_points);
        arma::mat f_constant(n_induced_points, m);
        
        for (arma::uword j = 0; j < m; ++j) {
            arma::vec points;
            arma::interp1(theta_constant_all, f_constant_all.col(j),
                         theta_constant, points, "linear");
            f_constant.col(j) = points;
        }
        
        // Compute covariance matrices once
        arma::mat S_constant = K(theta_constant, theta_constant, beta_prior_sds.col(0));
        S_constant.diag() += 1e-6;
        arma::mat L_constant = arma::chol(S_constant, "lower");
        
        // Pre-compute common terms
        arma::mat kstar = K(theta_constant, theta_star, beta_prior_sds.col(0));
        arma::mat kstarT = kstar.t();
        arma::mat tmp_common = arma::solve(arma::trimatl(L_constant), kstar);
        arma::mat K_post = K(theta_star, theta_star, beta_prior_sds.col(0));
        K_post -= tmp_common.t() * tmp_common;
        K_post.diag() += 1e-6;
        arma::mat L_post = arma::chol(K_post, "lower");
        
        // Pre-allocate for parallel loop
        arma::vec alpha(n_induced_points);
        arma::vec draw_mean(N);
        
        // Draw f_star once (shared across horizons)
        arma::mat f_star(N, m);
        
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic) firstprivate(alpha, draw_mean)
        #endif
        for (arma::uword j = 0; j < m; ++j) {
            int tid = get_thread_id();
            Workspace& ws = ws_pool.get(tid);
            
            // Compute item-specific mean
            alpha = double_solve(L_constant, f_constant.col(j));
            draw_mean = kstarT * alpha + mu_star.slice(0).col(j);
            
            // Draw from posterior
            f_star.col(j) = draw_mean + rmvnorm_threadsafe(L_post, ws.rng);
        }

        // Copy to all horizons
        for (arma::uword h = 0; h < horizon; ++h) {
            results.slice(h) = f_star;
        }
    }
}