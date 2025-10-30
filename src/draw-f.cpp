#include "gpirt.h"
#include "mvnormal.h"

// Sparse ESS function
arma::vec ess_sparse(const arma::vec& f, const arma::vec& y, const arma::mat& cholS,
                     const arma::vec& mu, const arma::vec& thresholds,
                     const arma::uvec& obs_idx) {
    arma::uword n = f.n_elem;
    arma::vec nu = rmvnorm(cholS);
    double u = R::runif(0.0, 1.0);
    
    // Only compute likelihood for observed data
    double log_y = ll_bar_sparse(f, y, mu, thresholds, obs_idx) + std::log(u);
    
    bool reject = true;
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = R::runif(epsilon_min, epsilon_max);
    epsilon_min = epsilon - M_2PI;
    arma::vec f_prime(n);
    int iter = 0;
    
    while ( reject ) {
        iter += 1;
        f_prime = f * std::cos(epsilon) + nu * std::sin(epsilon);
        // Only compute likelihood for observed data in acceptance check
        if ( ll_bar_sparse(f_prime, y, mu, thresholds, obs_idx) > log_y ) {
            reject = false;
        } else {
            if ( epsilon < 0.0 ) {
                epsilon_min = epsilon;
            } else {
                epsilon_max = epsilon;
            }
            epsilon = R::runif(epsilon_min, epsilon_max);
        }
    }
    return f_prime;
}

// Updated draw_f_ to use sparse ESS
inline arma::mat draw_f_(const arma::mat& f, const arma::mat& y, const arma::mat& cholS,
                        const arma::mat& mu, const arma::mat& thresholds,
                        const arma::field<arma::uvec>& obs_persons_h) {
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::mat result(n, m);
    for ( arma::uword j = 0; j < m; ++j) {
        // Use sparse ESS with only observed respondents for this item
        result.col(j) = ess_sparse(f.col(j), y.col(j), cholS, mu.col(j), 
                                  thresholds.row(j).t(), obs_persons_h(j));
    }
    return result;
}

arma::cube draw_f(const arma::cube& f, const arma::mat& theta, const arma::cube& y, 
                  const arma::cube& cholS, const arma::mat& beta_prior_sds, 
                  const arma::cube& mu, const arma::cube& thresholds, 
                  const int constant_IRF,
                  const arma::field<arma::uvec>& obs_persons) {
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::uword horizon = f.n_slices;
    arma::cube result(n, m, horizon);

    if(constant_IRF==0){
        // draw f separately for non-constant IRF
        for ( arma::uword h = 0; h < horizon; ++h){
            // Extract obs_persons for this horizon
            arma::field<arma::uvec> obs_persons_h(m);
            for(arma::uword j = 0; j < m; ++j) {
                obs_persons_h(j) = obs_persons(j, h);
            }
            
            result.slice(h) = draw_f_(f.slice(h), y.slice(h), cholS.slice(h),
                                     mu.slice(h), thresholds.slice(h), obs_persons_h);
        }
    }
    else{
        // For constant IRF, still use sparse handling
        arma::mat f_constant(n*horizon, m);
        arma::mat y_constant(n*horizon, m);
        arma::mat mu_constant(n*horizon, m);
        arma::vec theta_constant(n*horizon);
        
        // Build combined data with sparsity tracking
        arma::field<arma::uvec> obs_persons_constant(m);
        for(arma::uword j = 0; j < m; ++j) {
            arma::uvec all_obs;
            for(arma::uword h = 0; h < horizon; ++h) {
                arma::uvec h_obs = obs_persons(j, h);
                all_obs = arma::join_cols(all_obs, h_obs + h*n);
            }
            obs_persons_constant(j) = all_obs;
        }
        
        for (arma::uword h = 0; h < horizon; h++){
            theta_constant.subvec(h*n, (h+1)*n-1) = theta.col(h);
            for ( arma::uword j = 0; j < m; ++j ) {
                f_constant.col(j).subvec(h*n, (h+1)*n-1) = f.slice(h).col(j);
                y_constant.col(j).subvec(h*n, (h+1)*n-1) = y.slice(h).col(j);
                mu_constant.col(j).subvec(h*n, (h+1)*n-1) = mu.slice(h).col(j);
            }
        }
        
        arma::mat X_constant(n*horizon, 3);
        X_constant.col(0) = arma::ones<arma::vec>(n*horizon);
        X_constant.col(1) = theta_constant;
        X_constant.col(2) = arma::pow(theta_constant, 2);
        
        arma::mat S_constant = K(theta_constant, theta_constant, beta_prior_sds.col(0));
        S_constant.diag() += 1e-6;
        arma::mat L_constant = arma::chol(S_constant, "lower");
        
        arma::mat f_prime = draw_f_(f_constant, y_constant, L_constant, 
                                   mu_constant, thresholds.slice(0), 
                                   obs_persons_constant);

        for ( arma::uword h = 0; h < horizon; ++h ){
            for ( arma::uword j = 0; j < m; ++j ){
                result.slice(h).col(j) = f_prime.col(j).subvec(h*n, (h+1)*n-1);
            }
        }
    }
    
    return result;
}
