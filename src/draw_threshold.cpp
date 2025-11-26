#include "gpirt.h"
#include "mvnormal.h"

inline void ess_threshold_sparse_threadsafe(arma::vec& out,
                                            const arma::vec& delta, const arma::cube& f,
                                            const arma::cube& y, const arma::cube& mu,
                                            const arma::uvec& obs_idx,
                                            arma::uword j_idx,
                                            Workspace& ws) {
    arma::uword C = delta.n_elem + 1;
    arma::uword m = y.n_cols;
    arma::uword horizon = y.n_slices;
    arma::uword n = y.n_rows;
    
    arma::vec v(C-1, arma::fill::ones);
    arma::mat S = arma::diagmat(v);
    arma::mat cholS = arma::chol(S, "lower");
    
    // Use thread-safe RNG with pre-allocated workspace
    ws.nu.head(C-1) = rmvnorm_threadsafe(cholS, ws.rng);
    
    double u = ws.rng.runif();
    double log_y = std::log(u);
    arma::vec thresholds = delta_to_threshold(delta);
    
    // Compute log likelihood over all observed data for this item
    for (arma::uword h = 0; h < horizon; h++) {
        for (arma::uword i = 0; i < n; ++i) {
            // Check if this person-horizon is in obs_idx
            arma::uword combined_idx = h * n + i;
            bool is_observed = false;
            for (arma::uword k = 0; k < obs_idx.n_elem; ++k) {
                if (obs_idx(k) == combined_idx) {
                    is_observed = true;
                    break;
                }
            }
            if (!is_observed) continue;
            
            double f_val = f(i, j_idx, h);
            double y_val = y(i, j_idx, h);
            double mu_val = mu(i, j_idx, h);
            
            if (!std::isnan(y_val)) {
                int c = int(y_val);
                double g = f_val + mu_val;
                double z1 = thresholds(c-1) - g;
                double z2 = thresholds(c) - g;
                log_y += std::log(R::pnorm(z2, 0, 1, 1, 0) - 
                                 R::pnorm(z1, 0, 1, 1, 0) + 1e-6);
            }
        }
    }

    bool reject = true;
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = ws.rng.runif(epsilon_min, epsilon_max);
    epsilon_min = epsilon - M_2PI;

    while (reject) {
        for (arma::uword i = 0; i < C-1; ++i) {
            ws.delta_prime(i) = delta(i) * std::cos(epsilon) + ws.nu(i) * std::sin(epsilon);
        }
        
        double log_y_prime = 0;
        arma::vec thresholds_prime = delta_to_threshold(ws.delta_prime.head(C-1));
        
        for (arma::uword h = 0; h < horizon; h++){
            for (arma::uword i = 0; i < n; ++i) {
                arma::uword combined_idx = h * n + i;
                bool is_observed = false;
                for (arma::uword k = 0; k < obs_idx.n_elem; ++k) {
                    if (obs_idx(k) == combined_idx) {
                        is_observed = true;
                        break;
                    }
                }
                if (!is_observed) continue;
                
                double f_val = f(i, j_idx, h);
                double y_val = y(i, j_idx, h);
                double mu_val = mu(i, j_idx, h);
                
                if (!std::isnan(y_val)) {
                    int c = int(y_val);
                    double g = f_val + mu_val;
                    double z1 = thresholds_prime(c-1) - g;
                    double z2 = thresholds_prime(c) - g;
                    log_y_prime += std::log(R::pnorm(z2, 0, 1, 1, 0) - 
                                           R::pnorm(z1, 0, 1, 1, 0) + 1e-6);
                }
            }
        }
        
        if (log_y_prime > log_y) {
            reject = false;
        }
        else {
            if (epsilon < 0.0) {
                epsilon_min = epsilon;
            }
            else {
                epsilon_max = epsilon;
            }
            epsilon = ws.rng.runif(epsilon_min, epsilon_max);
        }
    }
    
    out = ws.delta_prime.head(C-1);
}

// Simpler version for non-constant IRF case (single horizon, single item)
inline void ess_threshold_single_threadsafe(arma::vec& out,
                                            const arma::vec& delta, 
                                            const arma::vec& f_col,
                                            const arma::vec& y_col, 
                                            const arma::vec& mu_col,
                                            const arma::uvec& obs_idx,
                                            Workspace& ws) {
    arma::uword C = delta.n_elem + 1;
    
    arma::vec v(C-1, arma::fill::ones);
    arma::mat S = arma::diagmat(v);
    arma::mat cholS = arma::chol(S, "lower");
    
    ws.nu.head(C-1) = rmvnorm_threadsafe(cholS, ws.rng);
    
    double u = ws.rng.runif();
    double log_y = std::log(u);
    arma::vec thresholds = delta_to_threshold(delta);
    
    // Compute likelihood for observed indices
    for (arma::uword k = 0; k < obs_idx.n_elem; ++k) {
        arma::uword i = obs_idx(k);
        int c = int(y_col(i));
        double g = f_col(i) + mu_col(i);
        double z1 = thresholds(c-1) - g;
        double z2 = thresholds(c) - g;
        log_y += std::log(R::pnorm(z2, 0, 1, 1, 0) - 
                         R::pnorm(z1, 0, 1, 1, 0) + 1e-6);
    }

    bool reject = true;
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = ws.rng.runif(epsilon_min, epsilon_max);
    epsilon_min = epsilon - M_2PI;

    while (reject) {
        for (arma::uword i = 0; i < C-1; ++i) {
            ws.delta_prime(i) = delta(i) * std::cos(epsilon) + ws.nu(i) * std::sin(epsilon);
        }
        
        double log_y_prime = 0;
        arma::vec thresholds_prime = delta_to_threshold(ws.delta_prime.head(C-1));
        
        for (arma::uword k = 0; k < obs_idx.n_elem; ++k) {
            arma::uword i = obs_idx(k);
            int c = int(y_col(i));
            double g = f_col(i) + mu_col(i);
            double z1 = thresholds_prime(c-1) - g;
            double z2 = thresholds_prime(c) - g;
            log_y_prime += std::log(R::pnorm(z2, 0, 1, 1, 0) - 
                                   R::pnorm(z1, 0, 1, 1, 0) + 1e-6);
        }
        
        if (log_y_prime > log_y) {
            reject = false;
        }
        else {
            if (epsilon < 0.0) {
                epsilon_min = epsilon;
            }
            else {
                epsilon_max = epsilon;
            }
            epsilon = ws.rng.runif(epsilon_min, epsilon_max);
        }
    }
    
    out = ws.delta_prime.head(C-1);
}

void draw_threshold(arma::cube& result, const arma::cube& thresholds, const arma::cube& y,
                    const arma::cube& f, const arma::cube& mu, 
                    const int constant_IRF,
                    const arma::field<arma::uvec>& obs_persons,
                    const arma::field<arma::uvec>& obs_persons_combined,
                    WorkspacePool& ws_pool){
    arma::uword m = thresholds.n_rows;
    arma::uword C = thresholds.n_cols - 1;
    arma::uword horizon = thresholds.n_slices;
    
    // Zero out result
    result.zeros();
    
    if(constant_IRF == 1){
        // Parallelize over items
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (arma::uword j = 0; j < m; ++j){
            int tid = get_thread_id();
            Workspace& ws = ws_pool.get(tid);
            
            // Use pre-computed combined observation indices (const reference)
            const arma::uvec& obs_idx = obs_persons_combined(j, 0);
            
            arma::vec delta = threshold_to_delta(thresholds.slice(0).row(j).t());
            arma::vec delta_prime(C-1);
            ess_threshold_sparse_threadsafe(delta_prime, delta, f, y, mu, obs_idx, j, ws);
            arma::vec new_thresholds = delta_to_threshold(delta_prime);
            
            result.slice(0).row(j) = new_thresholds.t();
            
            for(arma::uword h = 1; h < horizon; ++h){
                result.slice(h).row(j) = result.slice(0).row(j);
            }
        }
    }
    else{
        for (arma::uword h = 0; h < horizon; ++h){
            // Parallelize over items
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for (arma::uword j = 0; j < m; ++j){
                int tid = get_thread_id();
                Workspace& ws = ws_pool.get(tid);
                
                // Use const reference to avoid copy
                const arma::uvec& obs_idx = obs_persons(j, h);
                
                arma::vec delta = threshold_to_delta(thresholds.slice(h).row(j).t());
                arma::vec delta_prime(C-1);
                ess_threshold_single_threadsafe(delta_prime, delta, 
                                               f.slice(h).col(j),
                                               y.slice(h).col(j),
                                               mu.slice(h).col(j),
                                               obs_idx, ws);
                result.slice(h).row(j) = delta_to_threshold(delta_prime).t();
            }
        }
    }
}