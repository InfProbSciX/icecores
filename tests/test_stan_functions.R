
library(rstan)
library(magrittr)
library(testthat)
rstan_options(auto_write = TRUE)

hmm_marginal_banded_code = readLines('../stan/functions.stan') %>%
    paste(collapse='\n')

extra_functions_code = "
functions {
    real stan_hmm_marginal(matrix log_omega,
                           matrix Gamma,
                           vector rho) {
        return hmm_marginal(log_omega, Gamma, rho);
    }
}"

stan_model(model_code=hmm_marginal_banded_code) %>% expose_stan_functions
stan_model(model_code=extra_functions_code) %>% expose_stan_functions

test_that('hmm_marginal computations match with native Stan implementation.', {

    num_states = 2000; num_data = 130

    # transition matrix
    Gamma = diag(runif(num_states))
    Gamma[row(Gamma) == col(Gamma) - 1] = 1 - diag(Gamma)[-num_states]
    Gamma[num_states, num_states] = 1

    # likelihoods
    log_omega = matrix(dnorm(rnorm(num_data*num_states), log=T), num_states, num_data)

    # initial probs
    rho = runif(num_states)
    rho = exp(rho) / sum(exp(rho))

    expect_equal(
        hmm_marginal_banded(log_omega, diag(Gamma), rho),
        stan_hmm_marginal(log_omega, Gamma, rho)
    )

    expect_lt(
        system.time(hmm_marginal_banded(log_omega, diag(Gamma), rho))['elapsed'],
        system.time(stan_hmm_marginal(log_omega, Gamma, rho))['elapsed']
    )

})

test_that('tile works as expected.', {
    expect_equal(
        tile(1:5, 3),
        rep(1:5, 3)
    )
})
