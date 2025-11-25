#include "gpirt.h"
#include "mvnormal.h"

inline arma::mat draw_fstar_ws(const arma::mat& f, const arma::vec& theta,
                               const arma::vec& theta_star, const arma::mat& beta_prior_sds,
                               const arma::mat& L, const arma::mat& mu_star,
                               Workspace& ws) {
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::uword N = theta_star.n_elem;
    arma::mat result(N, m);
    
    // Compute k* once for all items
    arma::mat kstar = K(theta, theta_star, beta_prior_sds.col(0));
    arma::mat kstarT = kstar.t();
    
    // Use workspace for tmp computation
    ws.tmp_mat.set_size(n, N);
    ws.tmp_mat = arma::solve(arma::trimatl(L), kstar);
    
    // Compute posterior covariance once
    arma::mat K_post = K(theta_star, theta_star, beta_prior_sds.col(0)) - ws.tmp_mat.t() * ws.tmp_mat;
    K_post.diag() += 1e-6;
    arma::mat L_post = arma::chol(K_post);
    
    // Use pre-allocated workspace
    ws.alpha.set_size(n);
    ws.draw_mean.set_size(N);
    
    for (arma::uword j = 0; j < m; ++j) {
        ws.alpha = double_solve(L, f.col(j));
        ws.draw_mean = kstarT * ws.alpha + mu_star.col(j);
        result.col(j) = ws.draw_mean + rmvnorm(L_post);
    }
    return result;
}

arma::cube draw_fstar(const arma::cube& f, 
                      const arma::mat& theta,
                      const arma::vec& theta_star, 
                      const arma::mat& beta_prior_sds,
                      CholeskyCache& chol_cache,
                      const arma::cube& mu_star,
                      const int constant_IRF,
                      Workspace& ws) {
    arma::uword n = f.n_rows;
    arma::uword horizon = f.n_slices;
    arma::uword m = f.n_cols;
    arma::uword N = theta_star.n_elem;
    arma::cube results = arma::zeros<arma::cube>(N, m, horizon);
    
    if (constant_IRF == 0) {
        for (arma::uword h = 0; h < horizon; ++h) {
            // Use cached Cholesky factor
            results.slice(h) = draw_fstar_ws(f.slice(h), theta.col(h), 
                                            theta_star, beta_prior_sds, 
                                            chol_cache.L.slice(h), mu_star.slice(h), ws);
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
        arma::vec theta_constant(f_constant.n_rows);
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
        
        arma::mat f_star = draw_fstar_ws(f_constant, theta_constant, 
                                        theta_star, beta_prior_sds, 
                                        L_constant, mu_star.slice(0), ws);

        for (arma::uword h = 0; h < horizon; ++h) {
            results.slice(h) = f_star;
        }
    }
    
    return results;
}