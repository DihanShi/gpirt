#include "gpirt.h"
#include "mvnormal.h"
#include "iterative_solvers.h"
#include <time.h>

inline arma::mat draw_fstar_(const arma::mat& f, const arma::vec& theta,
                             const arma::vec& theta_star, const arma::mat& beta_prior_sds,
                             const arma::mat& S, const arma::mat& mu_star,
                             IterativeWorkspace& ws) {
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::uword N = theta_star.n_elem;
    arma::mat result(N, m);
    
    arma::mat kstar = K(theta, theta_star, beta_prior_sds.col(0));
    arma::mat kstarT = kstar.t();
    
    // Use PCG instead of Cholesky solve
    arma::vec M_inv_diag = 1.0 / S.diag();  // Diagonal preconditioner
    
    for (arma::uword j = 0; j < m; ++j) {
        // Solve S * alpha = f.col(j) using PCG
        arma::vec alpha = pcg_solve(S, f.col(j), M_inv_diag, 1e-6, 50);
        arma::vec draw_mean = kstarT * alpha + mu_star.col(j);
        
        // Compute posterior covariance
        arma::mat tmp(n, N);
        for(arma::uword k = 0; k < N; ++k) {
            tmp.col(k) = pcg_solve(S, kstar.col(k), M_inv_diag, 1e-6, 30);
        }
        
        arma::mat K_post = K(theta_star, theta_star, beta_prior_sds.col(0)) - kstarT * tmp;
        K_post.diag() += 1e-6;
        
        // Create and initialize workspace for posterior sampling if needed
        IterativeWorkspace ws_post;
        ws_post.init(N, 20);
        arma::vec sample = lanczos_mvn_sample(K_post, ws_post.z, ws_post.Q, 
                                              ws_post.alpha, ws_post.beta, 20);
        
        result.col(j) = draw_mean + sample;
    }
    return result;
}

arma::cube draw_fstar(const arma::cube& f, 
                      const arma::mat& theta,
                      const arma::vec& theta_star, 
                      const arma::mat& beta_prior_sds,
                      const arma::cube& S,  // Now S instead of L
                      const arma::cube& mu_star,
                      const int constant_IRF,
                      IterativeWorkspace& workspace) {
    arma::uword n = f.n_rows;
    arma::uword horizon = f.n_slices;
    arma::uword m = f.n_cols;
    arma::uword N = theta_star.n_elem;
    arma::cube results = arma::zeros<arma::cube>(N, m, horizon);
    
    if(constant_IRF==0){
        for (arma::uword h = 0; h < horizon; ++h){
            results.slice(h) = draw_fstar_(f.slice(h), theta.col(h), 
                                          theta_star, beta_prior_sds, 
                                          S.slice(h), mu_star.slice(h),
                                          workspace);
        }
    }
    else{
        // Constant IRF case with inducing points
        arma::mat f_constant_all(n*horizon, m);
        arma::vec theta_constant_all(n*horizon);
        for (arma::uword h = 0; h < horizon; h++){
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
        
        // Create and initialize workspace for inducing points
        IterativeWorkspace ws_induced;
        ws_induced.init(n_induced_points, 30);
        arma::mat f_star = draw_fstar_(f_constant, theta_constant, 
                                       theta_star, beta_prior_sds, 
                                       S_constant, mu_star.slice(0),
                                       ws_induced);

        for (arma::uword h = 0; h < horizon; ++h){
            results.slice(h) = f_star;
        }
    }
    
    return results;
}