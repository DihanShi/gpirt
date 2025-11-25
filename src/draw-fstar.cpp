#include "gpirt.h"
#include "mvnormal.h"

arma::cube draw_fstar(const arma::cube& f, 
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
    arma::cube results = arma::zeros<arma::cube>(N, m, horizon);
    
    if (constant_IRF == 0) {
        for (arma::uword h = 0; h < horizon; ++h) {
            // Pre-compute common terms for this horizon
            const arma::mat& L = chol_cache.L.slice(h);
            arma::mat kstar = K(theta.col(h), theta_star, beta_prior_sds.col(0));
            arma::mat kstarT = kstar.t();
            
            // Compute tmp_common = L^{-1} * kstar (shared across items)
            arma::mat tmp_common = arma::solve(arma::trimatl(L), kstar);
            
            // Compute posterior covariance (shared across items)
            arma::mat K_post = K(theta_star, theta_star, beta_prior_sds.col(0)) - tmp_common.t() * tmp_common;
            K_post.diag() += 1e-6;
            arma::mat L_post = arma::chol(K_post);
            
            // Parallelize over items
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for (arma::uword j = 0; j < m; ++j) {
                int tid = get_thread_id();
                Workspace& ws = ws_pool.get(tid);
                
                // Compute item-specific mean
                ws.alpha.set_size(n);
                ws.alpha = double_solve(L, f.slice(h).col(j));
                ws.draw_mean.set_size(N);
                ws.draw_mean = kstarT * ws.alpha + mu_star.slice(h).col(j);
                
                // Draw from posterior
                results.slice(h).col(j) = ws.draw_mean + rmvnorm_threadsafe(L_post, ws.rng);
            }
        }
    } else {
        // Constant IRF case with inducing points
        arma::mat f_constant_all(n*horizon, m);
        arma::vec theta_constant_all(n*horizon);
        
        for (arma::uword h = 0; h < horizon; h++) {
            theta_constant_all.subvec(h*n, (h+1)*n-1) = theta.col(h);
            for (arma::uword j = 0; j < m; ++j) {
                f_constant_all.col(j).subvec(h*n, (h+1)*n-1) = f.slice(h).col(j);
            }
        }
        
        int n_induced_points = 100;
        arma::mat f_constant(n_induced_points, m);
        arma::vec theta_constant(n_induced_points);
        theta_constant = arma::linspace(theta.min(), theta.max(), n_induced_points);
        
        for (arma::uword j = 0; j < m; ++j) {
            arma::vec points;
            arma::interp1(theta_constant_all, f_constant_all.col(j).t(),
                         theta_constant, points, "linear");
            f_constant.col(j) = points;
        }
        
        arma::mat S_constant = K(theta_constant, theta_constant, beta_prior_sds.col(0));
        S_constant.diag() += 1e-6;
        arma::mat L_constant = arma::chol(S_constant, "lower");
        
        // Pre-compute common terms
        arma::mat kstar = K(theta_constant, theta_star, beta_prior_sds.col(0));
        arma::mat kstarT = kstar.t();
        arma::mat tmp_common = arma::solve(arma::trimatl(L_constant), kstar);
        arma::mat K_post = K(theta_star, theta_star, beta_prior_sds.col(0)) - tmp_common.t() * tmp_common;
        K_post.diag() += 1e-6;
        arma::mat L_post = arma::chol(K_post);
        
        // Parallelize over items
        arma::mat f_star(N, m);
        
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (arma::uword j = 0; j < m; ++j) {
            int tid = get_thread_id();
            Workspace& ws = ws_pool.get(tid);
            
            // Compute item-specific mean
            ws.alpha.set_size(n_induced_points);
            ws.alpha = double_solve(L_constant, f_constant.col(j));
            ws.draw_mean.set_size(N);
            ws.draw_mean = kstarT * ws.alpha + mu_star.slice(0).col(j);
            
            // Draw from posterior
            f_star.col(j) = ws.draw_mean + rmvnorm_threadsafe(L_post, ws.rng);
        }

        for (arma::uword h = 0; h < horizon; ++h) {
            results.slice(h) = f_star;
        }
    }
    
    return results;
}