#include <RcppArmadillo.h>

// Function to set seed state
void set_seed_state(Rcpp::NumericVector seed_state);
void set_seed(int seed);
Rcpp::NumericVector get_seed_state();

// Cache structure for Cholesky decompositions
struct CholeskyCache {
    arma::cube L;           // Cholesky factors
    arma::cube L_time;      // Time covariance Cholesky factors
    arma::mat theta_hash;   // Hash of theta values used for L
    bool needs_update;       // Flag for update needed
    
    CholeskyCache(arma::uword n, arma::uword horizon) : 
        L(n, n, horizon), 
        L_time(horizon, horizon, n),
        theta_hash(n, horizon), 
        needs_update(true) {}
};

// Memory workspace for avoiding allocations
struct Workspace {
    // For ESS functions
    arma::vec nu;
    arma::vec f_prime;
    arma::vec theta_prime;
    arma::vec beta_prime;
    arma::vec delta_prime;
    
    // For likelihood computations
    arma::vec g;
    arma::vec f_obs;
    arma::vec mu_obs;
    arma::vec y_obs;
    
    // For matrix operations
    arma::mat tmp_mat;
    arma::vec tmp_vec;
    arma::vec alpha;
    arma::vec draw_mean;
    
    // Initialize with maximum expected sizes
    Workspace(arma::uword max_n, arma::uword max_m, arma::uword horizon) :
        nu(max_n), f_prime(max_n), theta_prime(horizon),
        beta_prime(3), delta_prime(10), // assuming max 10 categories
        g(max_n), f_obs(max_m), mu_obs(max_m), y_obs(max_m),
        tmp_mat(max_n, max_n), tmp_vec(max_n),
        alpha(max_n), draw_mean(1001) {}  // 1001 for theta_star grid
};

// Function to draw f with sparsity support and workspace
arma::cube draw_f(const arma::cube& f, const arma::mat& theta, const arma::cube& y, 
                  CholeskyCache& chol_cache, const arma::mat& beta_prior_sds, 
                  const arma::cube& mu, const arma::cube& thresholds, 
                  const int constant_IRF,
                  const arma::field<arma::uvec>& obs_items,
                  Workspace& ws);

// Function to draw fstar with workspace
arma::cube draw_fstar(const arma::cube& f, 
                      const arma::mat& theta,
                      const arma::vec& theta_star, 
                      const arma::mat& beta_prior_sds,
                      CholeskyCache& chol_cache,
                      const arma::cube& mu_star,
                      const int constant_IRF,
                      Workspace& ws);

// Function to draw theta with sparsity support and workspace
arma::mat draw_theta(const arma::vec& theta_star,
                     const arma::cube& y, const arma::mat& theta,
                     const arma::mat& theta_prior_sds,
                     const arma::cube& fstar, const arma::cube& mu_star,
                     const arma::cube& thresholds,
                     const double& os,
                     const double& ls, const std::string& KERNEL,
                     const arma::field<arma::uvec>& obs_items,
                     CholeskyCache& chol_cache,
                     Workspace& ws);

// Function to draw beta with sparsity support and workspace
arma::cube draw_beta(arma::cube& beta, const arma::cube& X,
                    const arma::cube& y, const arma::cube& f,
                    const arma::mat& prior_means, const arma::mat& prior_sds,
                    const arma::cube& thresholds,
                    const arma::field<arma::uvec>& obs_persons,
                    Workspace& ws);

// Function to draw thresholds with sparsity support and workspace
arma::cube draw_threshold(const arma::cube& thresholds, const arma::cube& y,
                    const arma::cube& f, const arma::cube& mu, 
                    const int constant_IRF,
                    const arma::field<arma::uvec>& obs_persons,
                    Workspace& ws);

// Utility function to update Cholesky cache if needed
void update_cholesky_cache(CholeskyCache& cache, const arma::mat& theta,
                          const arma::mat& beta_prior_sds,
                          const double& os, const double& ls,
                          const std::string& KERNEL);

// Covariance function
arma::mat K(const arma::vec& x1, const arma::vec& x2, const arma::vec& beta_prior_sds);
arma::mat K_time(const arma::vec& x1, const arma::vec& x2,
                 const double& os, const double& ls,
                 const arma::vec& theta_prior_sds, const std::string& KERNEL);

// Likelihood function for ordinal regression
double ll(const arma::vec& f, const arma::vec& y, const arma::mat& thresholds);
double ll_bar(const arma::vec& f, const arma::vec& y, const arma::vec& mu, const arma::vec& thresholds);

// Sparse likelihood functions
double ll_sparse(const arma::vec& f, const arma::vec& y, 
                 const arma::mat& thresholds, const arma::uvec& obs_idx);
double ll_bar_sparse(const arma::vec& f, const arma::vec& y, 
                     const arma::vec& mu, const arma::vec& thresholds,
                     const arma::uvec& obs_idx);

// convertion between thresholds and delta thresholds
arma::vec delta_to_threshold(const arma::vec& deltas);
arma::vec threshold_to_delta(const arma::vec& thresholds);

// cholesky decomposition
arma::mat double_solve(const arma::mat& L, const arma::mat& X);
arma::mat compress_toeplitz(arma::mat& T);
arma::mat toep_cholesky_lower(arma::mat& T);