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

// Memory workspace for avoiding allocations - per thread
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
    
    // Pre-allocated temporaries for draw functions
    arma::vec result_col;
    arma::mat X_obs;
    arma::mat cholS_small;
    arma::cube f_h;
    arma::cube y_h;
    arma::cube mu_h;
    
    // Thread-safe RNG
    ThreadRNG rng;
    
    // Initialize with maximum expected sizes
    Workspace(arma::uword max_n, arma::uword max_m, arma::uword horizon) :
        nu(max_n * horizon), f_prime(max_n * horizon), theta_prime(horizon),
        beta_prime(3), delta_prime(10), // assuming max 10 categories
        g(max_n * horizon), f_obs(max_n * horizon), mu_obs(max_n * horizon), y_obs(max_n * horizon),
        tmp_mat(max_n * horizon, 1001), tmp_vec(max_n * horizon),
        alpha(max_n * horizon), draw_mean(1001),
        result_col(max_n * horizon),
        X_obs(max_n * horizon, 3),
        cholS_small(3, 3),
        f_h(max_n, 1, 1),
        y_h(max_n, 1, 1),
        mu_h(max_n, 1, 1) {}
        
    // Copy constructor for creating thread-local copies
    Workspace(const Workspace& other) :
        nu(other.nu.n_elem), f_prime(other.f_prime.n_elem), 
        theta_prime(other.theta_prime.n_elem),
        beta_prime(other.beta_prime.n_elem), delta_prime(other.delta_prime.n_elem),
        g(other.g.n_elem), f_obs(other.f_obs.n_elem), 
        mu_obs(other.mu_obs.n_elem), y_obs(other.y_obs.n_elem),
        tmp_mat(other.tmp_mat.n_rows, other.tmp_mat.n_cols), 
        tmp_vec(other.tmp_vec.n_elem),
        alpha(other.alpha.n_elem), draw_mean(other.draw_mean.n_elem),
        result_col(other.result_col.n_elem),
        X_obs(other.X_obs.n_rows, other.X_obs.n_cols),
        cholS_small(other.cholS_small.n_rows, other.cholS_small.n_cols),
        f_h(other.f_h.n_rows, other.f_h.n_cols, other.f_h.n_slices),
        y_h(other.y_h.n_rows, other.y_h.n_cols, other.y_h.n_slices),
        mu_h(other.mu_h.n_rows, other.mu_h.n_cols, other.mu_h.n_slices) {}
};

// Thread workspace pool
class WorkspacePool {
private:
    std::vector<Workspace> workspaces;
    arma::uword max_n, max_m, horizon;
    
public:
    WorkspacePool(arma::uword n, arma::uword m, arma::uword h, int num_threads) 
        : max_n(n), max_m(m), horizon(h) {
        workspaces.reserve(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            workspaces.emplace_back(n, m, h);
        }
    }
    
    void seed_all(unsigned int base_seed) {
        for (size_t i = 0; i < workspaces.size(); ++i) {
            workspaces[i].rng.seed(base_seed + static_cast<unsigned int>(i) * 1000);
        }
    }
    
    Workspace& get(int thread_id) {
        return workspaces[thread_id];
    }
    
    Workspace& get_single() {
        return workspaces[0];
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

// convertion between thresholds and delta thresholds
arma::vec delta_to_threshold(const arma::vec& deltas);
arma::vec threshold_to_delta(const arma::vec& thresholds);

// cholesky decomposition
arma::mat double_solve(const arma::mat& L, const arma::mat& X);
arma::mat compress_toeplitz(arma::mat& T);
arma::mat toep_cholesky_lower(arma::mat& T);

#endif // GPIRT_H