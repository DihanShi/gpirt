#include "gpirt.h"
#include "iterative_solvers.h"

inline arma::mat draw_fstar_(const arma::mat& f, const arma::vec& theta,
                            const arma::vec& theta_star, const arma::mat& beta_prior_sds,
                            const arma::mat& K, const arma::mat& mu_star,
                            arma::field<IterativeWorkspace>& workspaces) {
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::uword N = theta_star.n_elem;
    arma::mat result(N, m);
    
    // Compute cross-covariance
    arma::mat kstar = K(theta, theta_star, beta_prior_sds.col(0));
    arma::mat kstarT = kstar.t();
    
    // For each item, solve K * alpha = f using PCG
    #pragma omp parallel for schedule(dynamic)
    for (arma::uword j = 0; j < m; ++j) {
        // Use PCG to solve K * alpha = f(:,j)
        arma::vec alpha = pcg_solve(K, f.col(j), workspaces(j), 1e-5, 30);
        
        // Predictive mean
        arma::vec draw_mean = kstarT * alpha + mu_star.col(j);
        
        // For predictive covariance, we need K** - K*' K^-1 K*
        // Approximate using low-rank update
        arma::mat tmp = kstarT;
        for(arma::uword i = 0; i < n; ++i) {
            arma::vec k_i = kstar.col(i);
            arma::vec v = pcg_solve(K, k_i, workspaces(j), 1e-5, 20);
            tmp.col(i) = v;
        }
        
        arma::mat K_post = K(theta_star, theta_star, beta_prior_sds.col(0));
        K_post -= kstarT * tmp;
        K_post.diag() += 1e-6;
        
        // Sample from posterior using Lanczos
        IterativeWorkspace work_post(N, 30);
        arma::vec sample = lanczos_mvn_sample(K_post, work_post, 30);
        result.col(j) = draw_mean + sample;
    }
    
    return result;
}

arma::cube draw_fstar(const arma::cube& f, 
                      const arma::mat& theta,
                      const arma::vec& theta_star, 
                      const arma::mat& beta_prior_sds,
                      const arma::cube& K,  // Now K instead of L
                      const arma::cube& mu_star,
                      const int constant_IRF,
                      arma::field<IterativeWorkspace>& workspaces) {
    arma::uword n = f.n_rows;
    arma::uword horizon = f.n_slices;
    arma::uword m = f.n_cols;
    arma::uword N = theta_star.n_elem;
    arma::cube results = arma::zeros<arma::cube>(N, m, horizon);
    
    if(constant_IRF == 0) {
        // Non-constant IRF
        for (arma::uword h = 0; h < horizon; ++h) {
            results.slice(h) = draw_fstar_(f.slice(h), theta.col(h), 
                                          theta_star, beta_prior_sds, 
                                          K.slice(h), mu_star.slice(h),
                                          workspaces);
        }
    } else {
        // Constant IRF - use inducing points
        arma::mat f_constant_all(n * horizon, m);
        arma::vec theta_constant_all(n * horizon);
        
        for (arma::uword h = 0; h < horizon; h++) {
            theta_constant_all.subvec(h*n, (h+1)*n-1) = theta.col(h);
            for (arma::uword j = 0; j < m; ++j) {
                f_constant_all.col(j).subvec(h*n, (h+1)*n-1) = f.slice(h).col(j);
            }
        }
        
        // Use inducing points for efficiency
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
        
        // Build kernel for inducing points
        arma::mat K_constant = K(theta_constant, theta_constant, beta_prior_sds.col(0));
        K_constant.diag() += 1e-6;
        
        // Larger workspace for inducing points
        arma::field<IterativeWorkspace> workspaces_induced(m);
        for(arma::uword j = 0; j < m; ++j) {
            workspaces_induced(j) = IterativeWorkspace(n_induced_points, 30);
        }
        
        arma::mat f_star = draw_fstar_(f_constant, theta_constant, 
                                       theta_star, beta_prior_sds, 
                                       K_constant, mu_star.slice(0),
                                       workspaces_induced);
        
        for (arma::uword h = 0; h < horizon; ++h) {
            results.slice(h) = f_star;
        }
    }
    
    return results;
}
