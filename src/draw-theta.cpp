#include "gpirt.h"
#include "mvnormal.h"

inline double compute_ll_sparse(const double theta,
                                const arma::vec& y,
                                const arma::mat& fstar,
                                const arma::mat& mu_star,
                                const arma::mat& thresholds,
                                const arma::uvec& obs_idx){
    // round nu to the nearest index grid - ADJUSTED FOR 0.1 GRID
    int theta_index = round((theta+5)/0.1);  // CHANGED FROM 0.01 TO 0.1
    if(theta_index<0){
        theta_index = 0;
    }else if (theta_index>101)  // CHANGED FROM 1001 TO 101
    {
        theta_index=101;  // CHANGED FROM 1001 TO 101
    }
    
    // Extract only observed items
    arma::vec f_obs = fstar.row(theta_index).t();
    arma::vec mu_obs = mu_star.row(theta_index).t();
    f_obs = f_obs(obs_idx);
    mu_obs = mu_obs(obs_idx);
    arma::vec y_obs = y(obs_idx);
    
    // Extract relevant thresholds for observed items
    double result = 0.0;
    for(arma::uword idx = 0; idx < obs_idx.n_elem; ++idx) {
        arma::uword j = obs_idx(idx);
        int c = int(y_obs(idx));
        double g = f_obs(idx) + mu_obs(idx);
        double z1 = thresholds(j, c-1) - g;
        double z2 = thresholds(j, c) - g;
        result += std::log(R::pnorm(z2, 0, 1, 1, 0) - 
                          R::pnorm(z1, 0, 1, 1, 0) + 1e-6);
    }
    return result;
}

inline arma::vec draw_theta_ess_sparse(const arma::vec& theta,
                                       const arma::mat& y,
                                       const arma::mat& L,
                                       const arma::cube& fstar,
                                       const arma::cube& mu_star,
                                       const arma::cube& thresholds,
                                       const arma::field<arma::uvec>& obs_items_i){
    arma::uword horizon = y.n_cols;
    arma::vec nu = rmvnorm(L);
    double u = R::runif(0.0,1.0);
    double log_y = std::log(u);
    
    for (arma::uword h = 0; h < horizon; h++) {
        log_y += compute_ll_sparse(theta(h), y.col(h), fstar.slice(h),
                                  mu_star.slice(h), thresholds.slice(h),
                                  obs_items_i(h));
    }

    bool reject = true;
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = R::runif(epsilon_min, epsilon_max);
    epsilon_min = epsilon - M_2PI;
    arma::vec theta_prime(horizon, arma::fill::zeros);

    while ( reject ) {
        theta_prime = theta * std::cos(epsilon) + nu * std::sin(epsilon);
        theta_prime.clamp(-5.0, 5.0);
        
        double log_y_prime = 0;
        for (arma::uword h = 0; h < horizon; h++) {
            log_y_prime += compute_ll_sparse(theta_prime(h), y.col(h),
                                            fstar.slice(h), mu_star.slice(h), 
                                            thresholds.slice(h), obs_items_i(h));
        }

        if ( log_y_prime > log_y ) {
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
    return theta_prime;
}

arma::mat draw_theta(const arma::vec& theta_star,
                     const arma::cube& y, const arma::mat& theta,
                     const arma::mat& theta_prior_sds,
                     const arma::cube& fstar, const arma::cube& mu_star,
                     const arma::cube& thresholds,
                     const double& os,
                     const double& ls, const std::string& KERNEL,
                     const arma::field<arma::uvec>& obs_items) {

    arma::uword n = y.n_rows;
    arma::uword m = y.n_cols;
    arma::uword horizon = y.n_slices;
    arma::uword N = theta_star.n_elem;
    arma::mat result(n, horizon);
    arma::vec ts = arma::linspace<arma::vec>(0, horizon-1, horizon);
    arma::mat V;
    
    if(ls>=3*horizon){
        // CST: constant theta across horizon
        V.ones(1, 1);
        for ( arma::uword i = 0; i < n; ++i ){
            // Combine all observed items across horizons
            arma::field<arma::uvec> obs_items_i(1);
            arma::uvec all_obs;
            for(arma::uword h = 0; h < horizon; ++h) {
                arma::uvec h_obs = obs_items(i, h);
                all_obs = arma::join_cols(all_obs, h_obs + h*m);
            }
            obs_items_i(0) = all_obs;
            
            // Reshape data for constant case
            arma::mat y_(m*horizon,1);
            arma::cube fstar_(N, m*horizon,1);
            arma::cube mu_star_(N, m*horizon,1);
            
            for(arma::uword h = 0; h < horizon; ++h) {
                y_.col(0).subvec(h*m, (h+1)*m-1) = y.slice(h).row(i).t();
                for(arma::uword k = 0; k < N; ++k) {
                    fstar_.slice(0).row(k).subvec(h*m, (h+1)*m-1) = fstar.slice(h).row(k);
                    mu_star_.slice(0).row(k).subvec(h*m, (h+1)*m-1) = mu_star.slice(h).row(k);
                }
            }
            
            arma::vec raw_theta_ess = draw_theta_ess_sparse(
                arma::vec(1, arma::fill::value(theta(i,0))), y_,
                arma::chol(V+std::pow(theta_prior_sds(0,i),2), "lower"), 
                fstar_, mu_star_, thresholds, obs_items_i);
                
            // ADJUSTED FOR 0.1 GRID
            for ( arma::uword h = 0; h < horizon; ++h ){
                result(i, h) = theta_star[round((raw_theta_ess(0)+5)/0.1)];  // CHANGED FROM 0.01 TO 0.1
            }
        }
    }else if(ls<=0.1){
        // RDM: independent theta
        V.ones(1, 1);
        for ( arma::uword i = 0; i < n; ++i ){
            for ( arma::uword h = 0; h < horizon; ++h ){
                arma::field<arma::uvec> obs_items_i(1);
                obs_items_i(0) = obs_items(i, h);
                
                arma::mat y_(m, 1);
                y_.col(0) = y.slice(h).row(i).t();
                arma::cube fstar_(N, m, 1);
                arma::cube mu_star_(N, m, 1);
                fstar_.slice(0) = fstar.slice(h);
                mu_star_.slice(0) = mu_star.slice(h);
                
                arma::vec raw_theta_ess = draw_theta_ess_sparse(
                    arma::vec(1, arma::fill::value(theta(i,h))), y_,
                    arma::chol(V+std::pow(theta_prior_sds(0,i),2), "lower"), 
                    fstar_, mu_star_, thresholds, obs_items_i);
                    
                // ADJUSTED FOR 0.1 GRID    
                result(i, h) = theta_star[round((raw_theta_ess(0)+5)/0.1)];  // CHANGED FROM 0.01 TO 0.1
            }
        }
    }else{
        // Regular GP case
        for ( arma::uword i = 0; i < n; ++i ){
            // Extract observed items for this respondent across all horizons
            arma::field<arma::uvec> obs_items_i(horizon);
            for(arma::uword h = 0; h < horizon; ++h) {
                obs_items_i(h) = obs_items(i, h);
            }
            
            V = K_time(ts, ts, os, ls, theta_prior_sds.col(i), KERNEL);
            V.diag() += 1e-6;
            arma::mat L = arma::chol(V, "lower");
            
            arma::vec raw_theta_ess = draw_theta_ess_sparse(theta.row(i).t(), y.row(i), 
                                                           L, fstar, mu_star, thresholds,
                                                           obs_items_i);
            // ADJUSTED FOR 0.1 GRID
            for ( arma::uword h = 0; h < horizon; ++h ){
                result(i, h) = theta_star[round((raw_theta_ess(h)+5)/0.1)];  // CHANGED FROM 0.01 TO 0.1
            }
        }
    }
    
    return result;
}
