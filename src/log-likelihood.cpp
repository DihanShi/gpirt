#include <RcppArmadillo.h>
#include "gpirt.h"

// Original likelihood functions (kept for compatibility)
double ll(const arma::vec& f, const arma::vec& y, const arma::mat& thresholds) {
    arma::uword m = f.n_elem;
    double result = 0.0;
    for ( arma::uword j = 0; j < m; ++j ) {
        if ( !std::isnan(y[j]) ) {
            int c = int(y[j]);
            double z1 = thresholds(j % thresholds.n_rows, c-1) - f[j];
            double z2 = thresholds(j % thresholds.n_rows, c) - f[j];
            result += std::log(R::pnorm(z2, 0, 1, 1, 0)-R::pnorm(z1, 0, 1, 1, 0)+1e-6);
        }
    }
    return result;
}

double ll_bar(const arma::vec& f, const arma::vec& y, const arma::vec& mu, const arma::vec& thresholds) {
    arma::uword n = f.n_elem;
    double result = 0.0;
    arma::vec g = f + mu;
    for ( arma::uword i = 0; i < n; ++i ) {
        if ( std::isnan(y[i]) ) {
            continue;
        }
        int c = int(y[i]);
        double z1 = thresholds[c-1] - g[i];
        double z2 = thresholds[c] - g[i];
        result += std::log(R::pnorm(z2, 0, 1, 1, 0)-R::pnorm(z1, 0, 1, 1, 0)+1e-6);
    }
    return result;
}

// NEW: Sparse versions that only process observed data
double ll_sparse(const arma::vec& f, const arma::vec& y, 
                 const arma::mat& thresholds, const arma::uvec& obs_idx) {
    double result = 0.0;
    for ( arma::uword idx = 0; idx < obs_idx.n_elem; ++idx ) {
        arma::uword j = obs_idx(idx);
        int c = int(y[j]);
        double z1 = thresholds(j % thresholds.n_rows, c-1) - f[j];
        double z2 = thresholds(j % thresholds.n_rows, c) - f[j];
        result += std::log(R::pnorm(z2, 0, 1, 1, 0) - 
                          R::pnorm(z1, 0, 1, 1, 0) + 1e-6);
    }
    return result;
}

double ll_bar_sparse(const arma::vec& f, const arma::vec& y, 
                     const arma::vec& mu, const arma::vec& thresholds,
                     const arma::uvec& obs_idx) {
    double result = 0.0;
    arma::vec g = f + mu;
    for ( arma::uword idx = 0; idx < obs_idx.n_elem; ++idx ) {
        arma::uword i = obs_idx(idx);
        int c = int(y[i]);
        double z1 = thresholds[c-1] - g[i];
        double z2 = thresholds[c] - g[i];
        result += std::log(R::pnorm(z2, 0, 1, 1, 0) - 
                          R::pnorm(z1, 0, 1, 1, 0) + 1e-6);
    }
    return result;
}

arma::vec delta_to_threshold(const arma::vec& deltas){
    arma::uword C = deltas.n_elem + 1;
    arma::vec thresholds(C+1, arma::fill::zeros);
    thresholds[0] = -std::numeric_limits<double>::infinity();
    thresholds[1] = deltas[0];
    thresholds[C] = std::numeric_limits<double>::infinity();
    for (arma::uword i = 1; i < C-1; i++)
    {
        thresholds[i+1] = thresholds[i] + std::exp(deltas[i]);
    }
    return thresholds;
}

arma::vec threshold_to_delta(const arma::vec& thresholds){
    arma::uword C = thresholds.n_elem - 1;
    arma::vec deltas(C-1, arma::fill::zeros);
    deltas[0] = thresholds[1];
    for (arma::uword i = 1; i < C-1; i++)
    {
        deltas[i] = std::log(thresholds[i+1]-thresholds[i]);
    }
    return deltas;
}