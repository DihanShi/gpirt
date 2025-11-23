#ifndef GPIRT_ITERATIVE_SOLVERS_H
#define GPIRT_ITERATIVE_SOLVERS_H

#include <RcppArmadillo.h>

// Preconditioned Conjugate Gradient solver with diagonal (Jacobi) preconditioner
inline arma::vec pcg_solve(const arma::mat& K, const arma::vec& b, 
                           const arma::vec& M_inv_diag,
                           double tol = 1e-6, int max_iter = 50) {
    arma::uword n = b.n_elem;
    arma::vec x(n, arma::fill::zeros);
    arma::vec r = b;  // Since x = 0, r = b - K*x = b
    arma::vec z = M_inv_diag % r;
    arma::vec p = z;
    double rsold = arma::dot(r, z);
    
    for(int iter = 0; iter < max_iter; ++iter) {
        arma::vec Ap = K * p;
        double alpha = rsold / arma::dot(p, Ap);
        x += alpha * p;
        r -= alpha * Ap;
        
        double r_norm = arma::norm(r);
        if(r_norm < tol) break;
        
        z = M_inv_diag % r;
        double rsnew = arma::dot(r, z);
        double beta = rsnew / rsold;
        p = z + beta * p;
        rsold = rsnew;
    }
    return x;
}

// Lanczos method for sampling from MVN(0, K)
inline arma::vec lanczos_mvn_sample(const arma::mat& K, 
                                    arma::vec& z,  // Pre-allocated random vector
                                    arma::mat& Q,  // Pre-allocated basis matrix
                                    arma::vec& alpha,  // Pre-allocated
                                    arma::vec& beta,   // Pre-allocated
                                    int num_lanczos = 30) {
    arma::uword n = K.n_rows;
    
    // Generate random vector
    for(arma::uword i = 0; i < n; ++i) {
        z(i) = R::rnorm(0.0, 1.0);
    }
    
    // Build Lanczos basis
    arma::vec q_prev(n, arma::fill::zeros);
    arma::vec q = z / arma::norm(z);
    Q.col(0) = q;
    
    for(int j = 0; j < num_lanczos; ++j) {
        arma::vec v = K * q;
        alpha(j) = arma::dot(q, v);
        
        if(j == 0) {
            v = v - alpha(j) * q;
        } else {
            v = v - alpha(j) * q - beta(j-1) * q_prev;
        }
        
        if(j < num_lanczos-1) {
            beta(j) = arma::norm(v);
            if(beta(j) < 1e-10) {
                // Early termination if Krylov subspace exhausted
                num_lanczos = j + 1;
                break;
            }
            q_prev = q;
            q = v / beta(j);
            Q.col(j+1) = q;
        }
    }
    
    // Build tridiagonal matrix
    arma::mat T(num_lanczos, num_lanczos, arma::fill::zeros);
    T.diag() = alpha.head(num_lanczos);
    for(int j = 0; j < num_lanczos-1; ++j) {
        T(j, j+1) = beta(j);
        T(j+1, j) = beta(j);
    }
    
    // Eigendecomposition of tridiagonal matrix
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, T);
    
    // Ensure all eigenvalues are positive (numerical stability)
    eigval = arma::abs(eigval);
    
    // Sample from approximate distribution
    arma::vec y(num_lanczos);
    for(int i = 0; i < num_lanczos; ++i) {
        y(i) = R::rnorm(0.0, 1.0);
    }
    y = eigvec * (arma::sqrt(eigval) % y);
    
    return Q.cols(0, num_lanczos-1) * y;
}

// Structure to hold pre-allocated workspace for iterative methods
struct IterativeWorkspace {
    // For Lanczos
    arma::vec z;
    arma::mat Q;
    arma::vec alpha;
    arma::vec beta;
    arma::vec q_prev;
    arma::vec v;
    
    // For PCG
    arma::vec x;
    arma::vec r;
    arma::vec z_pcg;
    arma::vec p;
    arma::vec Ap;
    
    // Default constructor (required for arma::field)
    IterativeWorkspace() {}
    
    // Initialization function to set up workspace
    void init(arma::uword n, int num_lanczos = 30) {
        // Lanczos workspace
        z = arma::vec(n);
        Q = arma::mat(n, num_lanczos);
        alpha = arma::vec(num_lanczos);
        beta = arma::vec(num_lanczos-1);
        q_prev = arma::vec(n);
        v = arma::vec(n);
        
        // PCG workspace
        x = arma::vec(n);
        r = arma::vec(n);
        z_pcg = arma::vec(n);
        p = arma::vec(n);
        Ap = arma::vec(n);
    }
    
    // Constructor with initialization
    IterativeWorkspace(arma::uword n, int num_lanczos = 30) {
        init(n, num_lanczos);
    }
};

#endif