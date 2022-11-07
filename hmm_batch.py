
import torch
import numpy as np
import pandas as pd
from tqdm import trange
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
    vector<lower=0, upper=1>[s] p_diag;
    real<lower=-3, upper=3> mu;
    real<lower=0, upper=1> sigma;
    real<lower=0, upper=2> scale;
}
transformed parameters {
    vector[s] cosine_term = cos(2.0*pi()*(year_fractions + 0.5));
    matrix[n_st, n] log_omega;

    for(i in 1:num_years){
        for(j in 1:s) {
            for(k in 1:n) {
                log_omega[(i-1)*s + j, k] =
                      normal_lpdf(y[k] | mu + cosine_term[j]*scale, sigma);
            }
        }
    }
}
model {
    target += hmm_marginal_banded(log_omega, tile(p_diag, num_years), rho);
}
generated quantities {
    matrix[n_st, n_st] p_full = diag_trans_to_full(tile(p_diag, num_years));
    matrix[n_st, n] posterior = hmm_hidden_state_prob(log_omega, p_full, rho);
    array[n] int sampled_states = hmm_latent_rng(log_omega, p_full, rho);
}
"""

with open('hmm.stan', 'w') as f:
    f.write(model_code)

model = CmdStanModel(stan_file='hmm.stan')

##############################################################
# Data preperation

depth, data, expert_labels, eruptions = load_data()

s = 6  # number of states in every year
batch_size = 180  # run each batch using data that is of this length
num_years = 150

initial_probs = get_initial_probs(num_years, s)  # initial dist of t|d

def get_posterior(n, s, num_years, initial_probs, data, depth):
    save_data(n, s, num_years, initial_probs, data, depth)
    fit = model.optimize(data='data.json', seed=42, show_console=False)
    posterior = fit.stan_variable('posterior')
    sampled_states = fit.stan_variable('sampled_states')
    return posterior, 2012 - sampled_states/s

year_estimates = batch_run(batch_size, s, num_years, initial_probs, depth, data, get_posterior)

np.save('dis_batch.npy', year_estimates)
