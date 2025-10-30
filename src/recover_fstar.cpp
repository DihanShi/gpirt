#include "gpirt.h"
#include "mvnormal.h"
#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export(.recover_fstar)]]
Rcpp::List recover_fstar(int seed, 
                         arma::cube f,
                         const arma::cube& y,
                         const arma::mat& theta,
                         const arma::cube& beta,
                         const arma::cube& thresholds,
                         const arma::mat& beta_prior_means,
                         const arma::mat& beta_prior_sds,
                         const int constant_IRF){
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::uword horizon = f.n_slices;

    // Create sparsity masks for this function
    arma::field<arma::uvec> obs_persons(m, horizon);
    for (arma::uword h = 0; h < horizon; ++h) {
        for (arma::uword j = 0; j < m; ++j) {
            obs_persons(j, h) = arma::find_finite(y.slice(h).col(j));
        }
    }

    // set up S
    arma::cube S = arma::zeros<arma::cube>(n, n, horizon);
    for (arma::uword h = 0; h < horizon; h++)
    {
        S.slice(h) = K(theta.col(h), theta.col(h), beta_prior_sds.col(0));
        S.slice(h).diag() += 1e-6;
    }

    // set up X
    arma::cube X(n, 2, horizon);
    X.col(0) = arma::ones<arma::mat>(n, horizon);
    X.col(1) = theta;

    // set up mu
    arma::cube mu(n,m,horizon);
    for (arma::uword h = 0; h < horizon; h++){
        mu.slice(h) = X.slice(h) * beta.slice(h);
    }

    // set up mu_star
    arma::vec theta_star = arma::regspace<arma::vec>(-5.0, 0.01, 5.0);
    arma::uword N = theta_star.n_elem;
    arma::mat Xstar(N, 2);
    Xstar.col(0) = arma::ones<arma::vec>(N);
    Xstar.col(1) = theta_star;
    arma::cube mu_star(N, m, horizon);
    for (arma::uword h = 0; h < horizon; h++){
        mu_star.slice(h) = Xstar * beta.slice(h);
    }

    // set up cholS
    arma::cube cholS(n, n, horizon);
    for (arma::uword h = 0; h < horizon; h++){
        cholS.slice(h) = arma::chol(S.slice(h), "lower");
    }
    
    // restore seed
    set_seed(seed);
    
    // Use sparse version of draw_f
    f = draw_f(f, theta, y, cholS, beta_prior_sds, mu, thresholds, constant_IRF, obs_persons);
    arma::cube f_star = draw_fstar(f, theta, theta_star, beta_prior_sds, cholS, mu_star, constant_IRF);
    
    Rcpp::List result = Rcpp::List::create(Rcpp::Named("fstar", f_star));
    return result;
}
