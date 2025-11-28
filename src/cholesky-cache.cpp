#include "gpirt.h"

// Solve L * L^T * x = b using forward and back substitution
arma::mat double_solve(const arma::mat& L, const arma::mat& X) {
    // First solve L * tmp = X (forward substitution)
    arma::mat tmp = arma::solve(arma::trimatl(L), X);
    // Then solve L^T * result = tmp (back substitution)
    return arma::solve(arma::trimatu(L.t()), tmp);
}

void update_cholesky_cache(CholeskyCache& cache, const arma::mat& theta,
                          const arma::mat& beta_prior_sds,
                          const double& os, const double& ls,
                          const std::string& KERNEL) {
    // Check if theta has changed significantly
    double theta_change = arma::norm(cache.theta_hash - theta, "fro");
    
    // Only update if theta has changed by more than threshold
    // Using 0.01 threshold for performance balance
    if (theta_change > 0.01 || cache.needs_update) {
        arma::uword n = theta.n_rows;
        arma::uword horizon = theta.n_cols;
        
        // Update spatial Cholesky factors
        for (arma::uword h = 0; h < horizon; h++) {
            arma::mat S = K(theta.col(h), theta.col(h), beta_prior_sds.col(0));
            S.diag() += 1e-6;
            cache.L.slice(h) = arma::chol(S, "lower");
        }
        
        // Update time Cholesky factors if needed (GP over time)
        if (ls > 0.1 && ls < 3.0 * static_cast<double>(horizon)) {
            arma::vec ts = arma::linspace<arma::vec>(0, horizon-1, horizon);
            for (arma::uword i = 0; i < n; ++i) {
                arma::mat V = K_time(ts, ts, os, ls, 
                                    arma::vec(2, arma::fill::zeros), KERNEL);
                V.diag() += 1e-6;
                cache.L_time.slice(i) = arma::chol(V, "lower");
            }
        }
        
        // Update hash and clear flag
        cache.theta_hash = theta;
        cache.needs_update = false;
    }
}