
import torch
import numpy as np
import pandas as pd
from tqdm import trange

def load_data():
    eruptions = np.array([
        [  0.33, 2012],  # latest date
        [ 69.9 , 1963],  # agung
        [133.59, 1883],  # krakatoa
    ])

    depth, log_msa, d18o = torch.load('snow.data')  # loads depth and two proxies
    depth_age_obs = torch.tensor(np.loadtxt('depth_age_obs.txt'))
    data = log_msa.reshape(-1).numpy() #  torch.cat([log_msa, d18o], axis=1).numpy()
    depth = depth[:, 0].numpy()

    # interpolate expert labels so that they're on the same depth scales as 
    # data - convenient for plotting
    expert_labels = np.interp(depth, depth_age_obs[:, 0], depth_age_obs[:, 1]).round()
    
    return (
        depth,  # np.array size n
        data,   # np.array size n
        expert_labels,  # np.array size n
        eruptions,  # np.array size n_ties * 2 (depth; tie point)
    )

def get_initial_probs(num_years, s):
    return np.concatenate([
        np.ones(s)/s,  # uniform over the first year
        np.zeros((num_years - 1) * s)  # zero for other states
    ], axis=0)

def save_data(n, s, num_years, initial_probs, data, depth):
    pd.Series(dict(
        n=n, s=s, num_years=num_years, initial_probs=initial_probs, y=data, depth=depth
    )).to_json('data.json')

def batch_run(batch_size, s, num_years, initial_probs, depth, data, get_posterior):
    n_full = len(data)
    current_data_idx = np.arange(batch_size)  # this will keep account of the current data indices
    our_year_estimates = np.array([])
    posterior_concat = np.empty((s*num_years, 0))

    for yr in trange(15):
        # as part of our chunking process, we utilize one extra data point than the batch size
        # so that the posterior for this data point can serve as the initial distribution for the next
        data_idx_plus_one = np.hstack([current_data_idx, current_data_idx.max() + 1])

        # at the end of the loop, we don't need any extra points
        data_idx_plus_one = data_idx_plus_one[data_idx_plus_one < len(depth)]

        # Mini-batch data
        n = len(data_idx_plus_one)

        posterior, current_year_estimates = \
            get_posterior(n, s, num_years, initial_probs, data[data_idx_plus_one], depth[data_idx_plus_one])

        # Set up for next iteration
        initial_probs = posterior[:, -1]  # last bit of posterior becomes initial state of next chunk

        end_of_loop: bool = max(data_idx_plus_one) + 1 == len(depth)  # data_idx_plus_one will be n - 1 at the end
        if not end_of_loop:
            posterior = posterior[:, :-1]  # just select the posterior without the extra point
            current_year_estimates = current_year_estimates[:-1]

        if np.isnan(posterior).any():
            print(yr)

        our_year_estimates = np.hstack((
            our_year_estimates,
            current_year_estimates
        ))
        current_data_idx += batch_size
        current_data_idx = current_data_idx[current_data_idx < n_full]

    return our_year_estimates
