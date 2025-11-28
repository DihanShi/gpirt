#ifndef GPIRT_H
#define GPIRT_H

#include <RcppArmadillo.h>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Thread-safe random number generator wrapper
class ThreadRNG {
private:
    std::mt19937 gen;
    std::normal_distribution<double> norm_dist;
    std::uniform_real_distribution<double> unif_dist;
    
public:
    ThreadRNG() : norm_dist(0.0, 1.0), unif_dist(0.0, 1.0) {}
    
    void seed(unsigned int s) {
        gen.seed(s);
    }
    
    double rnorm() {
        return norm_dist(gen);
    }
    
    double runif() {
        return unif_dist(gen);
    }
    
    double rnorm(double mean, double sd) {
        return mean + sd * norm_dist(gen);
    }
    
    double runif(double min, double max) {
        return min + (max - min) * unif_dist(gen);
    }
};

// Function to set seed state
void set_seed_state(Rcpp::NumericVector seed_state);
void set_seed(int seed);
Rcpp::NumericVector get_seed_state();

// Cache structure for Cholesky decompositions
struct CholeskyCache {
    arma::cube L;           // Cholesky factors for spatial covariance
    arma::cube L_time;      // Time covariance Cholesky factors
    arma::mat theta_hash;   // Hash of theta values used for L
    bool needs_update;       // Flag for update needed
    
    CholeskyCache(arma::uword n, arma::uword horizon) : 
        L(n, n, horizon, arma::fill::zeros), 
        L_time(horizon, horizon, n, arma::fill::zeros),
        theta_hash(n, horizon, arma::fill::zeros), 
        needs_update(true) {}
    
    // Prevent accidental copies
    CholeskyCache(const CholeskyCache&) = delete;
    CholeskyCache& operator=(const CholeskyCache&) = delete;
    
    // Allow moves
    CholeskyCache(CholeskyCache&&) = default;
    CholeskyCache& operator=(CholeskyCache&&) = default;
};

// Memory workspace for avoiding allocations - per thread
// All vectors are pre-allocated to maximum expected sizes
struct Workspace {
    // For ESS functions - pre-allocated to max sizes
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
    
    // Pre-allocated temporaries for draw functions
    arma::vec result_col;
    arma::mat X_obs;
    arma::mat cholS_small;
    
    // Thread-safe RNG
    ThreadRNG rng;
    
    // Maximum dimensions stored for bounds checking
    arma::uword max_n;
    arma::uword max_m;
    arma::uword max_horizon;
    
    // Initialize with maximum expected sizes
    Workspace(arma::uword n, arma::uword m, arma::uword horizon) :
        nu(n * horizon, arma::fill::zeros), 
        f_prime(n * horizon, arma::fill::zeros), 
        theta_prime(horizon, arma::fill::zeros),
        beta_prime(3, arma::fill::zeros), 
        delta_prime(20, arma::fill::zeros), // assuming max 20 categories
        g(n * horizon, arma::fill::zeros), 
        f_obs(n * horizon, arma::fill::zeros), 
        mu_obs(n * horizon, arma::fill::zeros), 
        y_obs(n * horizon, arma::fill::zeros),
        tmp_mat(n * horizon, 1001, arma::fill::zeros), 
        tmp_vec(n * horizon, arma::fill::zeros),
        alpha(n * horizon, arma::fill::zeros), 
        draw_mean(1001, arma::fill::zeros),
        result_col(n * horizon, arma::fill::zeros),
        X_obs(n * horizon, 3, arma::fill::zeros),
        cholS_small(3, 3, arma::fill::zeros),
        max_n(n), max_m(m), max_horizon(horizon) {}
        
    // Deleted copy constructor to prevent accidental copies
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    
    // Move constructor
    Workspace(Workspace&& other) noexcept :
        nu(std::move(other.nu)), 
        f_prime(std::move(other.f_prime)), 
        theta_prime(std::move(other.theta_prime)),
        beta_prime(std::move(other.beta_prime)), 
        delta_prime(std::move(other.delta_prime)),
        g(std::move(other.g)), 
        f_obs(std::move(other.f_obs)), 
        mu_obs(std::move(other.mu_obs)), 
        y_obs(std::move(other.y_obs)),
        tmp_mat(std::move(other.tmp_mat)), 
        tmp_vec(std::move(other.tmp_vec)),
        alpha(std::move(other.alpha)), 
        draw_mean(std::move(other.draw_mean)),
        result_col(std::move(other.result_col)),
        X_obs(std::move(other.X_obs)),
        cholS_small(std::move(other.cholS_small)),
        max_n(other.max_n), max_m(other.max_m), max_horizon(other.max_horizon) {}
    
    Workspace& operator=(Workspace&&) = default;
};

// Thread workspace pool - manages per-thread workspaces
class WorkspacePool {
private:
    std::vector<std::unique_ptr<Workspace>> workspaces;
    arma::uword max_n, max_m, horizon;
    
public:
    WorkspacePool(arma::uword n, arma::uword m, arma::uword h, int num_threads) 
        : max_n(n), max_m(m), horizon(h) {
        workspaces.reserve(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            workspaces.push_back(std::make_unique<Workspace>(n, m, h));
        }
    }
    
    // Prevent copies
    WorkspacePool(const WorkspacePool&) = delete;
    WorkspacePool& operator=(const WorkspacePool&) = delete;
    
    void seed_all(unsigned int base_seed) {
        for (size_t i = 0; i < workspaces.size(); ++i) {
            workspaces[i]->rng.seed(base_seed + static_cast<unsigned int>(i) * 1000);
        }
    }
    
    Workspace& get(int thread_id) {
        return *workspaces[thread_id];
    }
    
    Workspace& get_single() {
        return *workspaces[0];
    }
    
    int size() const {
        return static_cast<int>(workspaces.size());
    }
};

// Get number of threads
inline int get_num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Get current thread ID
inline int get_thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// Function to draw f with sparsity support and workspace pool (writes to output reference)
void draw_f(arma::cube& result, const arma::cube& f, const arma::mat& theta, const arma::cube& y, 
            CholeskyCache& chol_cache, const arma::mat& beta_prior_sds, 
            const arma::cube& mu, const arma::cube& thresholds, 
            const int constant_IRF,
            const arma::field<arma::uvec>& obs_persons,
            const arma::field<arma::uvec>& obs_persons_combined,
            WorkspacePool& ws_pool);

// Function to draw fstar with workspace pool (writes to output reference)
void draw_fstar(arma::cube& results, const arma::cube& f, 
                const arma::mat& theta,
                const arma::vec& theta_star, 
                const arma::mat& beta_prior_sds,
                CholeskyCache& chol_cache,
                const arma::cube& mu_star,
                const int constant_IRF,
                WorkspacePool& ws_pool);

// Function to draw theta with sparsity support and workspace pool (writes to output reference)
void draw_theta(arma::mat& result, const arma::vec& theta_star,
                const arma::cube& y, const arma::mat& theta,
                const arma::mat& theta_prior_sds,
                const arma::cube& fstar, const arma::cube& mu_star,
                const arma::cube& thresholds,
                const double& os,
                const double& ls, const std::string& KERNEL,
                const arma::field<arma::uvec>& obs_items,
                CholeskyCache& chol_cache,
                WorkspacePool& ws_pool);

// Function to draw beta with sparsity support and workspace pool (writes to output reference)
void draw_beta(arma::cube& result, const arma::cube& beta, const arma::cube& X,
               const arma::cube& y, const arma::cube& f,
               const arma::mat& prior_means, const arma::mat& prior_sds,
               const arma::cube& thresholds,
               const arma::field<arma::uvec>& obs_persons,
               WorkspacePool& ws_pool);

// Function to draw thresholds with sparsity support and workspace pool (writes to output reference)
void draw_threshold(arma::cube& result, const arma::cube& thresholds, const arma::cube& y,
                    const arma::cube& f, const arma::cube& mu, 
                    const int constant_IRF,
                    const arma::field<arma::uvec>& obs_persons,
                    const arma::field<arma::uvec>& obs_persons_combined,
                    WorkspacePool& ws_pool);

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

// Conversion between thresholds and delta thresholds
arma::vec delta_to_threshold(const arma::vec& deltas);
arma::vec threshold_to_delta(const arma::vec& thresholds);

// Cholesky decomposition utilities
arma::mat double_solve(const arma::mat& L, const arma::mat& X);

#endif // GPIRT_H