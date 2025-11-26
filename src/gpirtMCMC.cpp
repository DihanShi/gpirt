#include "gpirt.h"
#include "mvnormal.h"
#include <Rcpp.h>
#include <time.h>

using namespace Rcpp;

void set_seed(int seed) {
    Environment base_env("package:base");
    Function set_seed_r = base_env["set.seed"];
    set_seed_r(seed);
}

NumericVector get_seed_state(){
    Rcpp::Environment global_env(".GlobalEnv");
    return global_env[".Random.seed"];
}

void set_seed_state(NumericVector seed_state){
    Rcpp::Environment global_env(".GlobalEnv");
    global_env[".Random.seed"] = seed_state;
}

// [[Rcpp::export(.gpirtMCMC)]]
Rcpp::List gpirtMCMC(const arma::cube& y, arma::mat theta,
                     const int sample_iterations, const int burn_iterations, 
                     const int THIN,
                     const arma::mat& beta_prior_means,
                     const arma::mat& beta_prior_sds,
                     const arma::mat& theta_prior_means,
                     const arma::mat& theta_prior_sds,
                     const double& theta_os,
                     const double& theta_ls,
                     const std::string& KERNEL,
                     arma::cube thresholds,
                     const int constant_IRF) {

    // Bookkeeping variables
    arma::uword n = y.n_rows;
    arma::uword m = y.n_cols;
    arma::uword horizon = y.n_slices;
    int total_iterations = sample_iterations + burn_iterations;

    // Get number of threads
    int num_threads = get_num_threads();
    Rcpp::Rcout << "Using " << num_threads << " OpenMP thread(s)\n";

    // Initialize Cholesky cache and workspace pool
    CholeskyCache chol_cache(n, horizon);
    WorkspacePool ws_pool(n, m, horizon, num_threads);

    // Create sparsity masks for efficient NA handling
    arma::field<arma::uvec> obs_items(n, horizon);
    arma::field<arma::uvec> obs_persons(m, horizon);
    arma::umat n_obs(n, horizon);

    // Pre-compute indices of non-missing data
    for (arma::uword h = 0; h < horizon; ++h) {
        for (arma::uword i = 0; i < n; ++i) {
            arma::uvec valid = arma::find_finite(y.slice(h).row(i));
            obs_items(i, h) = valid;
            n_obs(i, h) = valid.n_elem;
        }
        for (arma::uword j = 0; j < m; ++j) {
            obs_persons(j, h) = arma::find_finite(y.slice(h).col(j));
        }
    }

    // Pre-compute combined observation indices for constant_IRF case
    // This avoids repeated allocation/deallocation in the MCMC loop
    arma::field<arma::uvec> obs_persons_combined(m, 1);
    for (arma::uword j = 0; j < m; ++j) {
        // Count total observations
        arma::uword total_obs = 0;
        for (arma::uword h = 0; h < horizon; ++h) {
            total_obs += obs_persons(j, h).n_elem;
        }
        // Pre-allocate and fill
        arma::uvec all_obs(total_obs);
        arma::uword pos = 0;
        for (arma::uword h = 0; h < horizon; ++h) {
            const arma::uvec& h_obs = obs_persons(j, h);
            for (arma::uword k = 0; k < h_obs.n_elem; ++k) {
                all_obs(pos++) = h_obs(k) + h * n;
            }
        }
        obs_persons_combined(j, 0) = all_obs;
    }

    double avg_obs = arma::mean(arma::vectorise(n_obs));
    Rcpp::Rcout << "Sparsity: Average " << avg_obs << " out of " 
                << m << " items observed per respondent ("
                << (avg_obs/m*100) << "% density)\n";

    // clamp theta
    theta.clamp(-5.0, 5.0);

    // Initialize and cache Cholesky decompositions
    update_cholesky_cache(chol_cache, theta, beta_prior_sds, theta_os, theta_ls, KERNEL);

    arma::cube X(n, 3, horizon);
    X.col(0) = arma::ones<arma::mat>(n, horizon);
    X.col(1) = theta;
    X.col(2) = arma::pow(theta,2);
    arma::cube f(n, m, horizon);
    arma::cube beta(3, m, horizon);
    arma::cube mu(n,m,horizon);
    
    Rcpp::Rcout << "Setting up gpirtMCMC...\n";

    // Setup initial values
    if(constant_IRF==0){
        for(arma::uword h = 0; h < horizon; ++h){
            for ( arma::uword j = 0; j < m; ++j ) {
                for ( arma::uword p = 0; p < 3; ++p ) {
                    beta.slice(h).col(j).row(p) = R::rnorm(beta_prior_means(p, j), beta_prior_sds(p, j));
                }
                mu.slice(h) = X.slice(h) * beta.slice(h);
            }
        }
        for (arma::uword h = 0; h < horizon; h++){
            for ( arma::uword j = 0; j < m; ++j ) {
                // Use cached Cholesky
                f.slice(h).col(j) = rmvnorm(chol_cache.L.slice(h));
            }
        }
    } 
    else{
        arma::vec theta_constant(n*horizon);
        for (arma::uword h = 0; h < horizon; h++){
            theta_constant.subvec(h*n, (h+1)*n-1) = theta.col(h);
        }

        arma::mat X_constant(n*horizon, 3);
        X_constant.col(0) = arma::ones<arma::vec>(n*horizon);
        X_constant.col(1) = theta_constant;
        X_constant.col(2) = arma::pow(theta_constant,2);
        
        arma::mat S_constant = K(theta_constant, theta_constant, beta_prior_sds.col(0));
        S_constant.diag() += 1e-6;

        arma::mat f_constant(n*horizon, m);
        arma::mat cholS_constant = arma::chol(S_constant, "lower");
        arma::mat mu_constant(n*horizon, m);
        
        for ( arma::uword j = 0; j < m; ++j ) {
            for ( arma::uword p = 0; p < 3; ++p ) {
                beta.slice(0).col(j).row(p) = R::rnorm(beta_prior_means(p, j), beta_prior_sds(p, j));
            }
        }
        mu_constant = X_constant * beta.slice(0);
        
        for ( arma::uword j = 0; j < m; ++j ) {
            f_constant.col(j).subvec(0, n-1) = rmvnorm(chol_cache.L.slice(0));
            for (arma::uword h = 0; h < horizon; h++){
                f_constant.col(j).subvec(h*n, (h+1)*n-1) = f_constant.col(j).subvec(0, n-1);
            }
        }

        for (arma::uword h = 0; h < horizon; h++){
            for ( arma::uword j = 0; j < m; ++j ) {
                f.slice(h).col(j) = f_constant.col(j).subvec(h*n, (h+1)*n-1);
                mu.slice(h).col(j) = mu_constant.col(j).subvec(h*n, (h+1)*n-1);
            }
        }
    }
    
    // Setup theta_star grid
    arma::vec theta_star = arma::regspace<arma::vec>(-5.0, 0.01, 5.0);
    arma::uword N = theta_star.n_elem;
    arma::mat Xstar(N, 3);
    Xstar.col(0) = arma::ones<arma::vec>(N);
    Xstar.col(1) = theta_star;
    Xstar.col(2) = arma::pow(theta_star,2);
    arma::cube mu_star(N, m, horizon);
    for (arma::uword h = 0; h < horizon; h++){
        mu_star.slice(h) = Xstar * beta.slice(h);
    }
    
    // Pre-allocate working arrays that will be reused each iteration
    arma::cube f_star(N, m, horizon);
    arma::cube f_new(n, m, horizon);
    arma::mat theta_new(n, horizon);
    arma::cube beta_new(3, m, horizon);
    arma::cube thresholds_new(m, thresholds.n_cols, horizon);
    
    // Initial f_star draw
    draw_fstar(f_star, f, theta, theta_star, beta_prior_sds, chol_cache, mu_star, constant_IRF, ws_pool);
    Rcpp::Rcout << "start running gpirtMCMC...\n";

    // Setup results storage
    arma::cube              theta_draws(int(sample_iterations/THIN), n, y.n_slices);
    arma::field<arma::cube> beta_draws(int(sample_iterations/THIN));
    arma::field<arma::cube> f_draws(int(sample_iterations/THIN));
    arma::field<arma::cube> fstar_draws(int(sample_iterations/THIN));
    arma::field<arma::cube> threshold_draws(int(sample_iterations/THIN));
    arma::vec               ll_draws(int(sample_iterations/THIN));

    double progress_increment = (1.0 / total_iterations) * 100.0;
    double progress = 0.0;

    // Start sampling loop
    for ( int iter = 0; iter < total_iterations; ++iter ) {
        Rprintf("\r%6.3f %% complete", progress);
        progress += progress_increment;
        Rcpp::checkUserInterrupt();

        // Seed all thread RNGs with iteration-based seeds for reproducibility
        ws_pool.seed_all(static_cast<unsigned int>(iter * 10000));

        // Draw with cached Cholesky and workspace pool - using output references
        draw_f(f_new, f, theta, y, chol_cache, beta_prior_sds, mu, thresholds, 
               constant_IRF, obs_persons, obs_persons_combined, ws_pool);
        f = f_new;  // Copy result back
        
        draw_fstar(f_star, f, theta, theta_star, beta_prior_sds, chol_cache, mu_star, constant_IRF, ws_pool);
        
        draw_theta(theta_new, theta_star, y, theta, theta_prior_sds, f_star, 
                   mu_star, thresholds, theta_os, theta_ls, KERNEL, obs_items, chol_cache, ws_pool);
        theta = theta_new;  // Copy result back
        
        // update X from theta
        X.col(1) = theta;
        X.col(2) = arma::pow(theta, 2);

        // Update f for new theta
        arma::mat idx = (theta+5)/0.01;
        for (arma::uword k = 0; k < n; ++k){
            for (arma::uword h = 0; h < horizon; ++h){
                f.slice(h).row(k) = f_star.slice(h).row(round(idx(k, h)));
            }
        }
        
        draw_beta(beta_new, beta, X, y, f, beta_prior_means, beta_prior_sds, 
                  thresholds, obs_persons, ws_pool);
        beta = beta_new;  // Copy result back
        
        // Update mu
        for (arma::uword h = 0; h < horizon; h++){
            mu.slice(h) = X.slice(h) * beta.slice(h);
            mu_star.slice(h) = Xstar * beta.slice(h);
        }
        
        // Mark cache for update (theta has changed)
        chol_cache.needs_update = true;
        update_cholesky_cache(chol_cache, theta, beta_prior_sds, theta_os, theta_ls, KERNEL);

        draw_threshold(thresholds_new, thresholds, y, f, mu, constant_IRF, 
                       obs_persons, obs_persons_combined, ws_pool);
        thresholds = thresholds_new;  // Copy result back
        
        // Compute log likelihood
        double ll = 0;
        for (arma::uword h = 0; h < horizon; h++){
            for (arma::uword j = 0; j < m; j++) {
                ll += ll_bar_sparse(f.slice(h).col(j), y.slice(h).col(j),
                                   mu.slice(h).col(j), thresholds.slice(h).row(j).t(),
                                   obs_persons(j, h));
            }
        }

        if (iter>=burn_iterations && iter%THIN == 0){
            int store_idx = int((iter-burn_iterations)/THIN);
            theta_draws.row(store_idx)     = theta;
            f_draws[store_idx]             = f;
            beta_draws[store_idx]          = beta;
            threshold_draws[store_idx]     = thresholds;
            fstar_draws[store_idx]         = f_star;
            ll_draws[store_idx]            = ll;
        }
    }
    Rprintf("\r100.000 %% complete\n");

    Rcpp::List result = Rcpp::List::create(Rcpp::Named("theta", theta_draws),
                                           Rcpp::Named("f", f_draws),
                                           Rcpp::Named("beta", beta_draws),
                                           Rcpp::Named("fstar", fstar_draws),
                                           Rcpp::Named("threshold", threshold_draws),
                                           Rcpp::Named("ll", ll_draws));
    return result;
}