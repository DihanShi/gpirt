#include "gpirt.h"
#include "mvnormal.h"
#include "iterative_solvers.h"
#include <Rcpp.h>
#include <time.h>

using namespace Rcpp;

// utility functions to set seed
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

    double avg_obs = arma::mean(arma::vectorise(n_obs));
    Rcpp::Rcout << "Sparsity: Average " << avg_obs << " out of " 
                << m << " items observed per respondent ("
                << (avg_obs/m*100) << "% density)\n";

    // PRE-ALLOCATE WORKSPACES for iterative methods
    arma::field<IterativeWorkspace> workspaces(m);
    for(arma::uword j = 0; j < m; ++j) {
        workspaces(j) = IterativeWorkspace(n, 30);
    }
    IterativeWorkspace workspace_fstar(1001, 30);  // For fstar computation

    // clamp theta
    theta.clamp(-5.0, 5.0);

    // Initialize S instead of computing Cholesky
    arma::cube S = arma::zeros<arma::cube>(n, n, horizon);
    for (arma::uword h = 0; h < horizon; h++) {   
        S.slice(h) = K(theta.col(h), theta.col(h), beta_prior_sds.col(0));
        S.slice(h).diag() += 1e-6;
    }
    
    arma::cube X(n, 3, horizon);
    X.col(0) = arma::ones<arma::mat>(n, horizon);
    X.col(1) = theta;
    X.col(2) = arma::pow(theta,2);
    arma::cube f(n, m, horizon);
    arma::cube beta(3, m, horizon);
    arma::cube mu(n,m,horizon);
    
    Rcpp::Rcout << "Setting up gpirtMCMC with iterative solvers...\n";

    // Setup each horizon separately for non-constant IRFs
    if(constant_IRF==0){
        for(arma::uword h = 0; h < horizon; ++h){
            for (arma::uword j = 0; j < m; ++j) {
                for (arma::uword p = 0; p < 3; ++p) {
                    beta.slice(h).col(j).row(p) = R::rnorm(beta_prior_means(p, j), beta_prior_sds(p, j));
                }
                mu.slice(h) = X.slice(h) * beta.slice(h);
            }
        }
        // Initialize f using Lanczos sampling
        for (arma::uword h = 0; h < horizon; h++){
            for (arma::uword j = 0; j < m; ++j) {
                f.slice(h).col(j) = lanczos_mvn_sample(S.slice(h), 
                                                       workspaces(j).z,
                                                       workspaces(j).Q,
                                                       workspaces(j).alpha,
                                                       workspaces(j).beta, 30);
            }
        }
    } 
    else{
        // Setup IRF object jointly using thetas across all horizons
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
        arma::mat mu_constant(n*horizon, m);
        
        for (arma::uword j = 0; j < m; ++j) {
            for (arma::uword p = 0; p < 3; ++p) {
                beta.slice(0).col(j).row(p) = R::rnorm(beta_prior_means(p, j), beta_prior_sds(p, j));
            }
        }
        mu_constant = X_constant * beta.slice(0);
        
        // Initialize with Lanczos
        IterativeWorkspace ws_init(n*horizon, 30);
        for (arma::uword j = 0; j < m; ++j) {
            arma::vec sample = lanczos_mvn_sample(S.slice(0), ws_init.z, ws_init.Q,
                                                  ws_init.alpha, ws_init.beta, 30);
            f_constant.col(j).subvec(0, n-1) = sample.head(n);
            for (arma::uword h = 0; h < horizon; h++){
                f_constant.col(j).subvec(h*n, (h+1)*n-1) = f_constant.col(j).subvec(0, n-1);
            }
        }

        for (arma::uword h = 0; h < horizon; h++){
            for (arma::uword j = 0; j < m; ++j) {
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
    
    arma::cube f_star = draw_fstar(f, theta, theta_star, beta_prior_sds, S, 
                                  mu_star, constant_IRF, workspace_fstar);
    Rcpp::Rcout << "Starting gpirtMCMC with iterative methods...\n";

    // Setup results storage
    arma::cube              theta_draws(int(sample_iterations/THIN), n, y.n_slices);
    arma::field<arma::cube> beta_draws(int(sample_iterations/THIN));
    arma::field<arma::cube> f_draws(int(sample_iterations/THIN));
    arma::field<arma::cube> fstar_draws(int(sample_iterations/THIN));
    arma::field<arma::cube> threshold_draws(int(sample_iterations/THIN));
    arma::vec               ll_draws(int(sample_iterations/THIN));

    // Pre-allocate temporary vectors used in the main loop
    arma::mat idx(n, horizon);
    
    double progress_increment = (1.0 / total_iterations) * 100.0;
    double progress = 0.0;

    // Start sampling loop
    for (int iter = 0; iter < total_iterations; ++iter) {
        // Update progress and check for user interrupt
        Rprintf("\r%6.3f %% complete", progress);
        progress += progress_increment;
        Rcpp::checkUserInterrupt();

        // set seed
        set_seed(iter);

        // Draw new parameter values with iterative methods
        f = draw_f(f, theta, y, S, beta_prior_sds, mu, thresholds, 
                  constant_IRF, obs_persons, workspaces);
        
        f_star = draw_fstar(f, theta, theta_star, beta_prior_sds, S, 
                           mu_star, constant_IRF, workspace_fstar);
        
        theta = draw_theta(theta_star, y, theta, theta_prior_sds, f_star, 
                          mu_star, thresholds, theta_os, theta_ls, KERNEL, obs_items);
        
        // update X from theta
        X.col(1) = theta;
        X.col(2) = arma::pow(theta, 2);

        // Update f for new theta
        idx = (theta+5)/0.01;
        for (arma::uword k = 0; k < n; ++k){
            for (arma::uword h = 0; h < horizon; ++h){
                f.slice(h).row(k) = f_star.slice(h).row(round(idx(k, h)));
            }
        }
        
        // draw beta with sparse handling
        beta = draw_beta(beta, X, y, f, beta_prior_means, beta_prior_sds, 
                        thresholds, obs_persons);
        
        // update S, mu from theta/beta (no Cholesky needed)
        for (arma::uword h = 0; h < horizon; h++){
            mu.slice(h) = X.slice(h) * beta.slice(h);
            mu_star.slice(h) = Xstar * beta.slice(h);
            S.slice(h) = K(theta.col(h), theta.col(h), beta_prior_sds.col(0));
            S.slice(h).diag() += 1e-6;
        }

        // draw thresholds with sparse handling
        thresholds = draw_threshold(thresholds, y, f, mu, constant_IRF, obs_persons);
        
        // compute current log likelihood (only for observed data)
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
            theta_draws.row(store_idx) = theta;
            f_draws[store_idx] = f;
            beta_draws[store_idx] = beta;
            threshold_draws[store_idx] = thresholds;
            fstar_draws[store_idx] = f_star;
            ll_draws[store_idx] = ll;
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