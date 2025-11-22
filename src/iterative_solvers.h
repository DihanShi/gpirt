#ifndef GPIRT_ITERATIVE_SOLVERS_H
#define GPIRT_ITERATIVE_SOLVERS_H

#include <RcppArmadillo.h>

// Workspace structure to avoid repeated allocations
struct IterativeWorkspace {
    // For PCG
    arma::vec pcg_r, pcg_z, pcg_p, pcg_Ap;
    arma::vec M_inv;  // Preconditioner
    
    // For Lanczos
    arma::mat Q;
    arma::vec alpha, beta;
    arma::vec q, q_prev, v;
    
    // For ESS
    arma::vec f_prime;
    arma::vec nu;
    
    // Constructor to pre-allocate
    IterativeWorkspace(arma::uword n, arma::uword lanczos_k = 50) {
        // PCG workspace
        pcg_r.set_size(n);
        pcg_z.set_size(n);
        pcg_p.set_size(n);
        pcg_Ap.set_size(n);
        M_inv.set_size(n);
        
        // Lanczos workspace
        Q.set_size(n, lanczos_k);
        alpha.set_size(lanczos_k);
        beta.set_size(lanczos_k - 1);
        q.set_size(n);
        q_prev.set_size(n);
        v.set_size(n);
        
        // ESS workspace
        f_prime.set_size(n);
        nu.set_size(n);
    }
};

// Preconditioned Conjugate Gradient with pre-allocated workspace
inline arma::vec pcg_solve(const arma::mat& K, const arma::vec& b,
                           IterativeWorkspace& work,
                           double tol = 1e-6, int max_iter = 50) {
    arma::uword n = b.n_elem;
    arma::vec x(n, arma::fill::zeros);
    
    // Setup Jacobi preconditioner (diagonal)
    work.M_inv = 1.0 / (K.diag() + 1e-6);
    
    // Initialize
    work.pcg_r = b;  // r = b - K*x, but x = 0
    work.pcg_z = work.M_inv % work.pcg_r;
    work.pcg_p = work.pcg_z;
    double rsold = arma::dot(work.pcg_r, work.pcg_z);
    
    for(int iter = 0; iter < max_iter; ++iter) {
        work.pcg_Ap = K * work.pcg_p;
        double alpha = rsold / arma::dot(work.pcg_p, work.pcg_Ap);
        x += alpha * work.pcg_p;
        work.pcg_r -= alpha * work.pcg_Ap;
        
        // Check convergence
        double r_norm = arma::norm(work.pcg_r);
        if(r_norm < tol) break;
        
        work.pcg_z = work.M_inv % work.pcg_r;
        double rsnew = arma::dot(work.pcg_r, work.pcg_z);
        double beta = rsnew / rsold;
        work.pcg_p = work.pcg_z + beta * work.pcg_p;
        rsold = rsnew;
    }
    return x;
}

// Lanczos method for sampling from N(0, K) with pre-allocated workspace
inline arma::vec lanczos_mvn_sample(const arma::mat& K, IterativeWorkspace& work,
                                    int num_lanczos = 50) {
    arma::uword n = K.n_rows;
    
    // Random initial vector
    work.q.randn();
    work.q /= arma::norm(work.q);
    work.Q.col(0) = work.q;
    work.q_prev.zeros();
    
    // Lanczos iteration to build tridiagonal matrix
    for(int j = 0; j < num_lanczos; ++j) {
        work.v = K * work.q;
        work.alpha(j) = arma::dot(work.q, work.v);
        
        if(j == 0) {
            work.v -= work.alpha(j) * work.q;
        } else {
            work.v -= work.alpha(j) * work.q + work.beta(j-1) * work.q_prev;
        }
        
        if(j < num_lanczos - 1) {
            work.beta(j) = arma::norm(work.v);
            if(work.beta(j) < 1e-10) {
                // Early termination if breakdown
                num_lanczos = j + 1;
                break;
            }
            work.q_prev = work.q;
            work.q = work.v / work.beta(j);
            work.Q.col(j + 1) = work.q;
        }
    }
    
    // Build tridiagonal matrix
    arma::mat T(num_lanczos, num_lanczos, arma::fill::zeros);
    T.diag() = work.alpha.head(num_lanczos);
    for(int j = 0; j < num_lanczos - 1; ++j) {
        T(j, j + 1) = work.beta(j);
        T(j + 1, j) = work.beta(j);
    }
    
    // Eigendecomposition of tridiagonal matrix
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, T);
    
    // Handle negative eigenvalues (shouldn't happen with positive definite K)
    eigval = arma::abs(eigval);
    
    // Sample in eigenspace
    arma::vec y(num_lanczos);
    y.randn();
    y = eigvec * (arma::sqrt(eigval) % y);
    
    // Project back to original space
    return work.Q.cols(0, num_lanczos - 1) * y;
}

// Matrix-vector multiply using structure when possible
inline arma::vec structured_matvec(const arma::mat& K, const arma::vec& x) {
    // For now, just standard multiply
    // Could be optimized for specific kernel structures
    return K * x;
}

#endif
