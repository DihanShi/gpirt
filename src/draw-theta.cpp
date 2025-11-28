#include "gpirt.h"
#include "mvnormal.h"

inline double compute_ll_sparse(const double theta,
                                const arma::vec& y,
                                const arma::mat& fstar,
                                const arma::mat& mu_star,
                                const arma::mat& thresholds,
                                const arma::uvec& obs_idx){
    // Coarse 0.1 grid spacing: multiply by 10 to index into original theta_star array
    int theta_index = static_cast<int>(round((theta + 5.0) / 0.1)) * 10;
    if(theta_index < 0){
        theta_index = 0;
    } else if (theta_index > 1000) {
        theta_index = 1000;
    }
    
    double result = 0.0;
    for(arma::uword idx = 0; idx < obs_idx.n_elem; ++idx) {
        arma::uword j = obs_idx(idx);
        int c = static_cast<int>(y(j));
        double g = fstar(theta_index, j) + mu_star(theta_index, j);
        double z1 = thresholds(j, c-1) - g;
        double z2 = thresholds(j, c) - g;
        result += std::log(R::pnorm(z2, 0, 1, 1, 0) - 
                          R::pnorm(z1, 0, 1, 1, 0) + 1e-6);
    }
    return result;
}

inline void draw_theta_ess_sparse_threadsafe(arma::vec& out,
                                             const arma::vec& theta,
                                             const arma::mat& y,
                                             const arma::mat& L,
                                             const arma::cube& fstar,
                                             const arma::cube& mu_star,
                                             const arma::cube& thresholds,
                                             const arma::field<arma::uvec>& obs_items_i,
                                             Workspace& ws){
    arma::uword horizon = y.n_cols;
    
    // Use thread-safe RNG with pre-allocated workspace
    ws.nu.head(horizon) = rmvnorm_threadsafe(L, ws.rng);
    
    double u = ws.rng.runif();
    double log_y = std::log(u);
    
    for (arma::uword h = 0; h < horizon; h++) {
        log_y += compute_ll_sparse(theta(h), y.col(h), fstar.slice(h),
                                  mu_star.slice(h), thresholds.slice(h),
                                  obs_items_i(h));
    }

    bool reject = true;
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = ws.rng.runif(epsilon_min, epsilon_max);
    epsilon_min = epsilon - M_2PI;

    while (reject) {
        for (arma::uword h = 0; h < horizon; ++h) {
            ws.theta_prime(h) = theta(h) * std::cos(epsilon) + ws.nu(h) * std::sin(epsilon);
            // Clamp in-place
            if (ws.theta_prime(h) < -5.0) ws.theta_prime(h) = -5.0;
            if (ws.theta_prime(h) > 5.0) ws.theta_prime(h) = 5.0;
        }
        
        double log_y_prime = 0;
        for (arma::uword h = 0; h < horizon; h++) {
            log_y_prime += compute_ll_sparse(ws.theta_prime(h), y.col(h),
                                            fstar.slice(h), mu_star.slice(h), 
                                            thresholds.slice(h), obs_items_i(h));
        }

        if (log_y_prime > log_y) {
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
    
    out = ws.theta_prime.head(horizon);
}

void draw_theta(arma::mat& result, const arma::vec& theta_star,
                const arma::cube& y, const arma::mat& theta,
                const arma::mat& theta_prior_sds,
                const arma::cube& fstar, const arma::cube& mu_star,
                const arma::cube& thresholds,
                const double& os,
                const double& ls, const std::string& KERNEL,
                const arma::field<arma::uvec>& obs_items,
                CholeskyCache& chol_cache,
                WorkspacePool& ws_pool) {

    arma::uword n = y.n_rows;
    arma::uword m = y.n_cols;
    arma::uword horizon = y.n_slices;
    arma::uword N = theta_star.n_elem;
    
    // Update time Cholesky factors if needed
    if (ls > 0.1 && ls < 3*horizon && chol_cache.needs_update) {
        arma::vec ts = arma::linspace<arma::vec>(0, horizon-1, horizon);
        for (arma::uword i = 0; i < n; ++i) {
            arma::mat V = K_time(ts, ts, os, ls, theta_prior_sds.col(i), KERNEL);
            V.diag() += 1e-6;
            chol_cache.L_time.slice(i) = arma::chol(V, "lower");
        }
    }
    
    if(ls >= 3*horizon){
        // CST: constant theta
        arma::mat V(1, 1, arma::fill::ones);
        
        // Pre-allocate thread-local storage outside the parallel region
        // FIX: Moved allocation outside parallel loop
        
        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            int tid = get_thread_id();
            Workspace& ws = ws_pool.get(tid);
            
            // Thread-local pre-allocated storage
            arma::field<arma::uvec> obs_items_i_local(1);
            arma::mat y_local(m*horizon, 1);
            arma::cube fstar_local(N, m*horizon, 1);
            arma::cube mu_star_local(N, m*horizon, 1);
            arma::uvec all_obs_local(m*horizon); // Max possible size
            arma::vec raw_theta_ess(1);
            
            #ifdef _OPENMP
            #pragma omp for schedule(dynamic)
            #endif
            for (arma::uword i = 0; i < n; ++i){
                // Count total observations for this respondent
                arma::uword total_obs = 0;
                for(arma::uword h = 0; h < horizon; ++h) {
                    total_obs += obs_items(i, h).n_elem;
                }
                
                // Build observation indices using pre-allocated storage
                arma::uword pos = 0;
                for(arma::uword h = 0; h < horizon; ++h) {
                    const arma::uvec& h_obs = obs_items(i, h);
                    for(arma::uword k = 0; k < h_obs.n_elem; ++k) {
                        all_obs_local(pos++) = h_obs(k) + h*m;
                    }
                }
                obs_items_i_local(0) = all_obs_local.head(total_obs);
                
                // Build combined data
                for(arma::uword h = 0; h < horizon; ++h) {
                    y_local.col(0).subvec(h*m, (h+1)*m-1) = y.slice(h).row(i).t();
                    for(arma::uword k = 0; k < N; ++k) {
                        fstar_local.slice(0).row(k).subvec(h*m, (h+1)*m-1) = fstar.slice(h).row(k);
                        mu_star_local.slice(0).row(k).subvec(h*m, (h+1)*m-1) = mu_star.slice(h).row(k);
                    }
                }
                
                double prior_var = V(0,0) + std::pow(theta_prior_sds(0,i), 2);
                arma::mat L_i(1, 1);
                L_i(0, 0) = std::sqrt(prior_var);
                
                draw_theta_ess_sparse_threadsafe(raw_theta_ess,
                    arma::vec(1, arma::fill::value(theta(i,0))), y_local,
                    L_i, fstar_local, mu_star_local, thresholds, obs_items_i_local, ws);
                    
                for (arma::uword h = 0; h < horizon; ++h){
                    int idx = static_cast<int>(round((raw_theta_ess(0)+5)/0.1)) * 10;
                    if(idx < 0) idx = 0;
                    if(idx > 1000) idx = 1000;
                    result(i, h) = theta_star(idx);
                }
            }
        }
    } else if(ls <= 0.1){
        // RDM: independent theta
        arma::mat V(1, 1, arma::fill::ones);
        
        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            int tid = get_thread_id();
            Workspace& ws = ws_pool.get(tid);
            
            // Thread-local pre-allocated storage
            arma::field<arma::uvec> obs_items_i_local(1);
            arma::mat y_local(m, 1);
            arma::cube fstar_local(N, m, 1);
            arma::cube mu_star_local(N, m, 1);
            arma::vec raw_theta_ess(1);
            
            #ifdef _OPENMP
            #pragma omp for schedule(dynamic)
            #endif
            for (arma::uword i = 0; i < n; ++i){
                for (arma::uword h = 0; h < horizon; ++h){
                    obs_items_i_local(0) = obs_items(i, h);
                    
                    y_local.col(0) = y.slice(h).row(i).t();
                    fstar_local.slice(0) = fstar.slice(h);
                    mu_star_local.slice(0) = mu_star.slice(h);
                    
                    double prior_var = V(0,0) + std::pow(theta_prior_sds(0,i), 2);
                    arma::mat L_i(1, 1);
                    L_i(0, 0) = std::sqrt(prior_var);
                    
                    draw_theta_ess_sparse_threadsafe(raw_theta_ess,
                        arma::vec(1, arma::fill::value(theta(i,h))), y_local,
                        L_i, fstar_local, mu_star_local, thresholds, obs_items_i_local, ws);
                        
                    int idx = static_cast<int>(round((raw_theta_ess(0)+5)/0.1)) * 10;
                    if(idx < 0) idx = 0;
                    if(idx > 1000) idx = 1000;
                    result(i, h) = theta_star(idx);
                }
            }
        }
    } else {
        // Regular GP case - use cached Cholesky
        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            int tid = get_thread_id();
            Workspace& ws = ws_pool.get(tid);
            
            // Thread-local pre-allocated storage
            arma::field<arma::uvec> obs_items_i_local(horizon);
            arma::vec raw_theta_ess(horizon);
            
            #ifdef _OPENMP
            #pragma omp for schedule(dynamic)
            #endif
            for (arma::uword i = 0; i < n; ++i){
                for(arma::uword h = 0; h < horizon; ++h) {
                    obs_items_i_local(h) = obs_items(i, h);
                }
                
                // Use cached L_time
                draw_theta_ess_sparse_threadsafe(raw_theta_ess,
                    theta.row(i).t(), y.row(i), 
                    chol_cache.L_time.slice(i), fstar, mu_star, thresholds,
                    obs_items_i_local, ws);
                    
                for (arma::uword h = 0; h < horizon; ++h){
                    int idx = static_cast<int>(round((raw_theta_ess(h)+5)/0.1)) * 10;
                    if(idx < 0) idx = 0;
                    if(idx > 1000) idx = 1000;
                    result(i, h) = theta_star(idx);
                }
            }
        }
    }
}