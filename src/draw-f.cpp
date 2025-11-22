#include "gpirt.h"
#include "iterative_solvers.h"

// ESS with iterative methods and pre-allocated workspace
arma::vec ess_iterative(const arma::vec& f, const arma::vec& y, 
                        const arma::mat& K, const arma::vec& mu, 
                        const arma::vec& thresholds, const arma::uvec& obs_idx,
                        IterativeWorkspace& work) {
    arma::uword n = f.n_elem;
    
    // Use Lanczos for sampling from N(0, K)
    work.nu = lanczos_mvn_sample(K, work, 40);  // 40 Lanczos iterations usually sufficient
    
    double u = R::runif(0.0, 1.0);
    double log_y = ll_bar_sparse(f, y, mu, thresholds, obs_idx) + std::log(u);
    
    bool reject = true;
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = R::runif(epsilon_min, epsilon_max);
    epsilon_min = epsilon - M_2PI;
    
    while (reject) {
        // Use pre-allocated workspace
        work.f_prime = f * std::cos(epsilon) + work.nu * std::sin(epsilon);
        
        if (ll_bar_sparse(work.f_prime, y, mu, thresholds, obs_idx) > log_y) {
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
    return work.f_prime;
}

// Updated draw_f_ to use iterative ESS
inline arma::mat draw_f_(const arma::mat& f, const arma::mat& y, 
                        const arma::mat& K, const arma::mat& mu, 
                        const arma::mat& thresholds,
                        const arma::field<arma::uvec>& obs_persons_h,
                        arma::field<IterativeWorkspace>& workspaces) {
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::mat result(n, m);
    
    // Parallel loop over items (if OpenMP enabled)
    #pragma omp parallel for schedule(dynamic)
    for (arma::uword j = 0; j < m; ++j) {
        result.col(j) = ess_iterative(f.col(j), y.col(j), K, mu.col(j), 
                                     thresholds.row(j).t(), obs_persons_h(j),
                                     workspaces(j));
    }
    return result;
}

arma::cube draw_f(const arma::cube& f, const arma::mat& theta, const arma::cube& y, 
                  const arma::cube& K,  // Now K instead of cholS
                  const arma::mat& beta_prior_sds, 
                  const arma::cube& mu, const arma::cube& thresholds, 
                  const int constant_IRF,
                  const arma::field<arma::uvec>& obs_persons,
                  arma::field<IterativeWorkspace>& workspaces) {
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::uword horizon = f.n_slices;
    arma::cube result(n, m, horizon);

    if(constant_IRF == 0) {
        // Non-constant IRF
        for (arma::uword h = 0; h < horizon; ++h) {
            arma::field<arma::uvec> obs_persons_h(m);
            for(arma::uword j = 0; j < m; ++j) {
                obs_persons_h(j) = obs_persons(j, h);
            }
            
            result.slice(h) = draw_f_(f.slice(h), y.slice(h), K.slice(h),
                                     mu.slice(h), thresholds.slice(h), 
                                     obs_persons_h, workspaces);
        }
    } else {
        // Constant IRF case
        arma::mat f_constant(n * horizon, m);
        arma::mat y_constant(n * horizon, m);
        arma::mat mu_constant(n * horizon, m);
        arma::vec theta_constant(n * horizon);
        
        arma::field<arma::uvec> obs_persons_constant(m);
        for(arma::uword j = 0; j < m; ++j) {
            arma::uvec all_obs;
            for(arma::uword h = 0; h < horizon; ++h) {
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
        
        // Build combined kernel matrix
        arma::mat K_constant = K(theta_constant, theta_constant, beta_prior_sds.col(0));
        K_constant.diag() += 1e-6;
        
        // Need larger workspace for combined data
        arma::field<IterativeWorkspace> workspaces_large(m);
        for(arma::uword j = 0; j < m; ++j) {
            workspaces_large(j) = IterativeWorkspace(n * horizon, 50);
        }
        
        arma::mat f_prime = draw_f_(f_constant, y_constant, K_constant, 
                                   mu_constant, thresholds.slice(0), 
                                   obs_persons_constant, workspaces_large);

        for (arma::uword h = 0; h < horizon; ++h) {
            for (arma::uword j = 0; j < m; ++j) {
                result.slice(h).col(j) = f_prime.col(j).subvec(h*n, (h+1)*n-1);
            }
        }
    }
    
    return result;
}
