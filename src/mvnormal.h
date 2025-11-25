#ifndef GPIRT_MVNORMAL_H
#define GPIRT_MVNORMAL_H

#include <RcppArmadillo.h>

// Original version using R's RNG (for single-threaded use)
inline arma::vec rmvnorm(const arma::mat& cholS) {
    arma::uword m = cholS.n_cols;
    arma::vec res(m);
    for (arma::uword i = 0; i < m; ++i ) {
        res[i] = R::rnorm(0.0, 1.0);
    }
    return cholS * res;
}

// Thread-safe version using ThreadRNG
inline arma::vec rmvnorm_threadsafe(const arma::mat& cholS, ThreadRNG& rng) {
    arma::uword m = cholS.n_cols;
    arma::vec res(m);
    for (arma::uword i = 0; i < m; ++i) {
        res[i] = rng.rnorm();
    }
    return cholS * res;
}

#endif