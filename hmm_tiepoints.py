
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from cmdstanpy import CmdStanModel

from utils import *

##############################################################
# Define model

model_code = \
    Path('stan/functions.stan').read_text() + \
    Path('stan/data.stan').read_text() + \
"""
parameters {
    vector<lower=0, upper=1>[n_st] p_diag;
    real<lower=-3, upper=3> mu[n];
    real<lower=0, upper=1> sigma[n];
    real<lower=0, upper=2> scale[n];

    vector<lower=0, upper=10>[s] p_a;
    vector<lower=0, upper=10>[s] p_b;
    real<lower=-3, upper=3> mu_m;
    real<lower=0, upper=3> mu_s;
    real<lower=0, upper=3> sg_s;
    real<lower=0, upper=4> cs_s;
}
transformed parameters {
    vector[s] cosine_term = cos(2.0*pi()*(year_fractions + 0.5));
    matrix[n_st, n] log_omega;

    for(i in 1:num_years){
        for(j in 1:s) {
            for(k in 1:n) {
                log_omega[(i-1)*s + j, k] =
                    normal_lpdf(y[k] | mu[k] + cosine_term[j]*scale[k], sigma[k]);
            }
        }
    }

    log_omega[1:(n_st - s*3), n] = rep_vector(-999.9, n_st - s*3);
    log_omega[(n_st - s*3 + 1):n_st, n] = rep_vector(100.0, s*3);
}
model {
    vector[n_st] p_a_transformed = tile(p_a, num_years);
    vector[n_st] p_b_transformed = tile(p_b, num_years);

    mu ~ normal(mu_m, mu_s);
    sigma ~ exponential(1/sg_s);
    scale ~ exponential(1/cs_s);
    p_diag ~ beta(p_a_transformed, p_b_transformed);

    target += hmm_marginal_banded(log_omega, p_diag, rho);
}
generated quantities {
    matrix[n_st, n_st] p_full = diag_trans_to_full(p_diag);
    matrix[n_st, n] posterior = hmm_hidden_state_prob(log_omega, p_full, rho);
    array[n] int sampled_states = hmm_latent_rng(log_omega, p_full, rho);
}
"""

with open('hmm.stan', 'w') as f:
    f.write(model_code)

model = CmdStanModel(stan_file='hmm.stan')

##############################################################
# Data preperation

s = 6  # number of states in every year
depth, data, expert_labels, eruptions = load_data()

for tie in range(len(eruptions) - 1):

    idx = np.arange(
        np.argmin(abs(depth - eruptions[tie, 0].item())),
        np.argmin(abs(depth - eruptions[tie + 1, 0].item()))
    )
    n = len(idx)  # length of data
    num_years = int((eruptions[tie, 1] - eruptions[tie + 1, 1]).item()) + 1

    initial_probs = get_initial_probs(num_years, s)

    save_data(n, s, num_years, initial_probs, data[idx], depth[idx])

    ##############################################################
    # Inference

    fit = model.variational(data='data.json', seed=42, show_console=True, output_samples=1, grad_samples=20)

    # Post hoc analysis of time given depth
    posterior = fit.stan_variable('posterior')
    sampled_states = fit.stan_variable('sampled_states')
    year_estimates = eruptions[tie, 1].item() - sampled_states/s  # this can be replaced with viterbi - would be more accurate

    np.save(f'single_pass_{tie}_idx.npy', idx)
    np.save(f'single_pass_{tie}_post.npy', posterior)
    np.save(f'single_pass_{tie}.npy', year_estimates)
