#include "gpirt.h"
#include "mvnormal.h"

arma::vec ess_threshold_sparse_ws(const arma::vec& delta, const arma::cube& f,
                                  const arma::cube& y, const arma::cube& mu,
                                  const arma::field<arma::uvec>& obs_persons,
                                  Workspace& ws) {
    arma::uword C = delta.n_elem + 1;
    arma::uword m = y.n_cols;
    arma::uword horizon = y.n_slices;
    
    arma::vec v(C-1, arma::fill::ones);
    arma::mat S = arma::diagmat(v);
    arma::mat cholS = arma::chol(S, "lower");
    
    // Use pre-allocated workspace
    ws.nu.set_size(C-1);
    ws.nu = rmvnorm(cholS);
    
    double u = R::runif(0.0,1.0);
    double log_y = std::log(u);
    arma::vec thresholds = delta_to_threshold(delta);
    
    for (arma::uword h = 0; h < horizon; h++) {
        for (arma::uword j = 0; j < m; j++){
            arma::uvec obs_idx = obs_persons(j, h);
            if(obs_idx.n_elem > 0) {
                // Use workspace for temporary storage
                ws.f_obs.set_size(obs_idx.n_elem);
                ws.y_obs.set_size(obs_idx.n_elem);
                ws.mu_obs.set_size(obs_idx.n_elem);
                
                // Correct syntax: first get the column as a vector, then subset
                arma::vec f_col = f.slice(h).col(j);
                arma::vec y_col = y.slice(h).col(j);
                arma::vec mu_col = mu.slice(h).col(j);
                
                ws.f_obs = f_col.elem(obs_idx);
                ws.y_obs = y_col.elem(obs_idx);
                ws.mu_obs = mu_col.elem(obs_idx);
                
                arma::uvec valid_idx = arma::linspace<arma::uvec>(0, obs_idx.n_elem-1, obs_idx.n_elem);
                
                log_y += ll_bar_sparse(ws.f_obs, ws.y_obs, ws.mu_obs, thresholds, valid_idx);
            }
        }
    }

    bool reject = true;
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = R::runif(epsilon_min, epsilon_max);
    epsilon_min = epsilon - M_2PI;
    
    // Use pre-allocated workspace
    ws.delta_prime.set_size(C-1);

    while (reject) {
        ws.delta_prime = delta * std::cos(epsilon) + ws.nu * std::sin(epsilon);
        double log_y_prime = 0;
        arma::vec thresholds_prime = delta_to_threshold(ws.delta_prime);
        
        for (arma::uword h = 0; h < horizon; h++){
            for (arma::uword j = 0; j < m; j++) {
                arma::uvec obs_idx = obs_persons(j, h);
                if(obs_idx.n_elem > 0) {
                    ws.f_obs.set_size(obs_idx.n_elem);
                    ws.y_obs.set_size(obs_idx.n_elem);
                    ws.mu_obs.set_size(obs_idx.n_elem);
                    
                    // Correct syntax: first get the column as a vector, then subset
                    arma::vec f_col = f.slice(h).col(j);
                    arma::vec y_col = y.slice(h).col(j);
                    arma::vec mu_col = mu.slice(h).col(j);
                    
                    ws.f_obs = f_col.elem(obs_idx);
                    ws.y_obs = y_col.elem(obs_idx);
                    ws.mu_obs = mu_col.elem(obs_idx);
                    
                    arma::uvec valid_idx = arma::linspace<arma::uvec>(0, obs_idx.n_elem-1, obs_idx.n_elem);
                    
                    log_y_prime += ll_bar_sparse(ws.f_obs, ws.y_obs, ws.mu_obs, thresholds_prime, valid_idx);
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
            epsilon = R::runif(epsilon_min, epsilon_max);
        }
    }
    return ws.delta_prime;
}

arma::cube draw_threshold(const arma::cube& thresholds, const arma::cube& y,
                         const arma::cube& f, const arma::cube& mu, 
                         const int constant_IRF,
                         const arma::field<arma::uvec>& obs_persons,
                         Workspace& ws){
    arma::uword m = thresholds.n_rows;
    arma::uword C = thresholds.n_cols - 1;
    arma::uword horizon = thresholds.n_slices;
    arma::cube thresholds_prime(m, C+1, horizon, arma::fill::zeros);
    
    if(constant_IRF==1){
        arma::field<arma::uvec> obs_persons_combined(m, 1);
        for(arma::uword j = 0; j < m; ++j) {
            arma::uvec all_obs;
            for(arma::uword h = 0; h < horizon; ++h) {
                arma::uvec h_obs = obs_persons(j, h);
                all_obs = arma::join_cols(all_obs, h_obs + h*f.n_rows);
            }
            obs_persons_combined(j, 0) = all_obs;
        }
        
        for ( arma::uword j = 0; j < m; ++j ){
            arma::field<arma::uvec> obs_j(1, 1);
            obs_j(0, 0) = obs_persons_combined(j, 0);
            
            arma::vec delta = threshold_to_delta(thresholds.slice(0).row(j).t());
            arma::vec delta_prime = ess_threshold_sparse_ws(delta, f, y, mu, obs_j, ws);
            thresholds_prime.slice(0).row(j) = delta_to_threshold(delta_prime).t();
            
            for(arma::uword h = 1; h < horizon; ++h){
                thresholds_prime.slice(h).row(j) = thresholds_prime.slice(0).row(j);
            }
        }
    }
    else{
        for ( arma::uword h = 0; h < horizon; ++h){
            for ( arma::uword j = 0; j < m; ++j ){
                arma::field<arma::uvec> obs_j(1, 1);
                obs_j(0, 0) = obs_persons(j, h);
                
                arma::cube f_h(f.n_rows, 1, 1);
                arma::cube y_h(y.n_rows, 1, 1);
                arma::cube mu_h(mu.n_rows, 1, 1);
                
                f_h.slice(0).col(0) = f.slice(h).col(j);
                y_h.slice(0).col(0) = y.slice(h).col(j);
                mu_h.slice(0).col(0) = mu.slice(h).col(j);
                
                arma::vec delta = threshold_to_delta(thresholds.slice(h).row(j).t());
                arma::vec delta_prime = ess_threshold_sparse_ws(delta, f_h, y_h, mu_h, obs_j, ws);
                thresholds_prime.slice(h).row(j) = delta_to_threshold(delta_prime).t();
            }
        }
    }
    
    return thresholds_prime;
}