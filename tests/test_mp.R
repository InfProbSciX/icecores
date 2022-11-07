
library(rstan)
library(magrittr)
library(testthat)
rstan_options(auto_write = TRUE)

functions_code = "
functions {
    matrix diag_rate_to_full(vector rate_mat_diag) {
        int n = size(rate_mat_diag);
        matrix[n, n] full_mat = -diag_matrix(rate_mat_diag);

        for (i in 1:(n - 1)) {
            full_mat[i, i + 1] = rate_mat_diag[i];
        }
        full_mat[n, n] = 0.0;
        return full_mat;
    }

    real hmm_marginal_cts(matrix log_omega,
                          vector Gamma_diag,
                          vector rho,
                          vector ts) {
        int K = dims(log_omega)[1];
        int N = dims(log_omega)[2];

        vector[K] log_alpha;
        vector[K] inner_sum;
        vector[K] inner_vec;
        vector[K] log_Gamma_diag = log(Gamma_diag);
        matrix[K, K] log_trans_mat;

        int min_i; int max_i;

        log_alpha = log_omega[, 1] + log(rho);

        if (N > 1) {
            for (n in 2:N) {
                log_trans_mat = log(matrix_exp((ts[n] - ts[n - 1]) * diag_rate_to_full(Gamma_diag)) + 1e-10);
                for (i in 1:K) {
                    inner_vec = log_alpha + log_trans_mat[, i];
                    inner_sum[i] = log_sum_exp(inner_vec);
                }
                log_alpha = log_omega[, n] + inner_sum;
            }
        }

        return log_sum_exp(log_alpha);
    }
    real stan_hmm_marginal(matrix log_omega,
                           matrix Gamma,
                           vector rho) {
        return hmm_marginal(log_omega, Gamma, rho);
    }
}"

stan_model(model_code=functions_code) %>% expose_stan_functions

test_that('hmm_marginal computations match with native Stan implementation.', {

    num_states = 30; num_data = 180

    # transition matrix
    rates_diag = rexp(num_states)
    rates = diag_rate_to_full(rates_diag)
    probs = as.matrix(Matrix::expm(rates))

    # likelihoods
    log_omega = matrix(dnorm(rnorm(num_data*num_states), log=T), num_states, num_data)

    # initial probs
    rho = c(5, 5, runif(num_states - 2))
    rho = exp(rho) / sum(exp(rho))
    ts = 1:num_data

    expect_equal(
        hmm_marginal_cts(log_omega, rates_diag, rho, ts),
        stan_hmm_marginal(log_omega, probs, rho),
    tolerance=1e-3)

})
