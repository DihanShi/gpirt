#include "gpirt.h"
#include "mvnormal.h"

namespace {

// Armadillo's interp1 assumes x is sorted (and behaves badly with unsorted x).
// In constant_IRF mode we stack theta across horizons, which is not sorted and
// can contain duplicates (due to grid clamping). This helper sorts x and
// collapses duplicates by averaging y within each duplicated x value.
static inline void make_sorted_unique_xy(const arma::vec& x_in,
                                         const arma::vec& y_in,
                                         arma::vec& x_out,
                                         arma::vec& y_out) {
    arma::uword n = x_in.n_elem;
    arma::uvec ord = arma::sort_index(x_in);
    arma::vec x = x_in(ord);
    arma::vec y = y_in(ord);

    arma::vec x_u(n);
    arma::vec y_u(n);
    arma::uword k = 0;

    arma::uword i = 0;
    while (i < n) {
        double xv = x(i);
        double sum = 0.0;
        arma::uword cnt = 0;
        while (i < n && x(i) == xv) {
            sum += y(i);
            ++cnt;
            ++i;
        }
        x_u(k) = xv;
        y_u(k) = sum / static_cast<double>(cnt);
        ++k;
    }

    x_out = x_u.head(k);
    y_out = y_u.head(k);
}

} // namespace

void draw_fstar(arma::cube& results, const arma::cube& f,
                const arma::mat& theta,
                const arma::vec& theta_star,
                const arma::mat& beta_prior_sds,
                CholeskyCache& chol_cache,
                const arma::cube& mu_star,
                const int constant_IRF,
                WorkspacePool& ws_pool) {
    (void)mu_star; // fstar is the GP deviation; the likelihood adds mu_star separately.

    arma::uword n = f.n_rows;
    arma::uword horizon = f.n_slices;
    arma::uword m = f.n_cols;
    arma::uword N = theta_star.n_elem;

    // Zero out results (reusing pre-allocated memory)
    results.zeros();

    if (constant_IRF == 0) {
        for (arma::uword h = 0; h < horizon; ++h) {
            // Pre-compute common terms for this horizon
            const arma::mat& L = chol_cache.L.slice(h);

            // Compute kstar - allocate once per horizon, not per item
            arma::mat kstar = K(theta.col(h), theta_star, beta_prior_sds.col(0));
            arma::mat kstarT = kstar.t();

            // Compute tmp_common = L^{-1} * kstar (shared across items)
            arma::mat tmp_common = arma::solve(arma::trimatl(L), kstar);

            // Compute posterior covariance (shared across items)
            arma::mat K_post = K(theta_star, theta_star, beta_prior_sds.col(0));
            K_post -= tmp_common.t() * tmp_common; // In-place subtraction
            K_post.diag() += 1e-6;
            arma::mat L_post = arma::chol(K_post, "lower");

            // Pre-allocate alpha vector for this horizon (reused across items)
            arma::vec alpha(n);
            arma::vec draw_mean(N);

            // Parallelize over items
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic) firstprivate(alpha, draw_mean)
            #endif
            for (arma::uword j = 0; j < m; ++j) {
                int tid = get_thread_id();
                Workspace& ws = ws_pool.get(tid);

                // Posterior mean for GP deviation f*(theta_star)
                alpha = double_solve(L, f.slice(h).col(j));
                draw_mean = kstarT * alpha;

                // Draw from posterior
                results.slice(h).col(j) = draw_mean + rmvnorm_threadsafe(L_post, ws.rng);
            }
        }
    } else {
        // Constant IRF case with inducing points
        arma::uword n_total = n * horizon;

        // Build combined data - allocate once
        arma::mat f_constant_all(n_total, m);
        arma::vec theta_constant_all(n_total);

        for (arma::uword h = 0; h < horizon; h++) {
            theta_constant_all.subvec(h * n, (h + 1) * n - 1) = theta.col(h);
            for (arma::uword j = 0; j < m; ++j) {
                f_constant_all.col(j).subvec(h * n, (h + 1) * n - 1) = f.slice(h).col(j);
            }
        }

        // Use inducing points for efficiency
        const arma::uword n_induced_points = 100;
        arma::vec theta_constant = arma::linspace(theta.min(), theta.max(), n_induced_points);
        arma::mat f_constant(n_induced_points, m);

        // Interpolate each item's stacked f onto the inducing grid.
        // IMPORTANT: theta_constant_all must be sorted for interp1.
        for (arma::uword j = 0; j < m; ++j) {
            arma::vec x_u, y_u;
            make_sorted_unique_xy(theta_constant_all, f_constant_all.col(j), x_u, y_u);

            arma::vec points;
            if (x_u.n_elem >= 2) {
                arma::interp1(x_u, y_u, theta_constant, points, "linear");
            } else {
                // All theta are identical; f is constant under this approximation.
                points.set_size(theta_constant.n_elem);
                points.fill(y_u(0));
            }
            f_constant.col(j) = points;
        }

        // Compute covariance matrices once
        arma::mat S_constant = K(theta_constant, theta_constant, beta_prior_sds.col(0));
        S_constant.diag() += 1e-6;
        arma::mat L_constant = arma::chol(S_constant, "lower");

        // Pre-compute common terms
        arma::mat kstar = K(theta_constant, theta_star, beta_prior_sds.col(0));
        arma::mat kstarT = kstar.t();
        arma::mat tmp_common = arma::solve(arma::trimatl(L_constant), kstar);
        arma::mat K_post = K(theta_star, theta_star, beta_prior_sds.col(0));
        K_post -= tmp_common.t() * tmp_common;
        K_post.diag() += 1e-6;
        arma::mat L_post = arma::chol(K_post, "lower");

        // Pre-allocate for parallel loop
        arma::vec alpha(n_induced_points);
        arma::vec draw_mean(N);

        // Draw f_star once (shared across horizons)
        arma::mat f_star(N, m);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic) firstprivate(alpha, draw_mean)
        #endif
        for (arma::uword j = 0; j < m; ++j) {
            int tid = get_thread_id();
            Workspace& ws = ws_pool.get(tid);

            // Posterior mean for GP deviation f*(theta_star)
            alpha = double_solve(L_constant, f_constant.col(j));
            draw_mean = kstarT * alpha;

            // Draw from posterior
            f_star.col(j) = draw_mean + rmvnorm_threadsafe(L_post, ws.rng);
        }

        // Copy to all horizons
        for (arma::uword h = 0; h < horizon; ++h) {
            results.slice(h) = f_star;
        }
    }
}
