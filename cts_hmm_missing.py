
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from utils import *

import matplotlib.pyplot as plt
plt.ion(); plt.style.use('seaborn-pastel')
# tf.test.is_gpu_available()

def variable(size):
    return tf.Variable(tf.ones([size]))

def log_prior(prior_dist, y):
    return tf.reduce_sum(prior_dist.log_prob(y))

s = 4; num_years = 5  # number of states in a year and max number of years for trans mat
d, y, _, _ = load_data()

y = np.hstack([y[:50], y[110:190]])
d = np.hstack([d[:50], d[110:190]])
n = len(y)
# plt.plot(d, y)

sinusoidal_basis = tf.abs(tf.linspace(-1 + 2/s, 1, s))
sinusoidal_basis = tf.cast(sinusoidal_basis, tf.float32)

obs_mean_prior_dist = tfd.Normal(0, 10)
obs_log_sd_prior_dist = tfd.Exponential(1)
transition_rates_prior = tfd.Exponential(0.01)

log_scale = variable(1)
bias = variable(1)
log_sigma = variable(s)
log_transition_rates = variable(s)

initial_probs = np.zeros(s*num_years) + 1e-6
initial_probs[:s] += 1/s
initial_probs /= initial_probs.sum()
initial_probs = tf.constant(initial_probs.astype('f'))
initial_dist = tfd.Categorical(probs=initial_probs)

n_states = s*num_years
Q_jitter = np.ones((n_states, n_states))*1e-6
Q_jitter[range(n_states), range(n_states)] = -Q_jitter[0, 1:].sum()
Q_jitter = tf.constant(Q_jitter.astype('f'))

def transition_rates_to_transition_matrix():
    """ Converts the rate matrix Q to a set of transition matrices P. """
    q_diag = tf.tile(tf.exp(log_transition_rates), (num_years,))
    Q = -tf.linalg.diag(q_diag)
    Q += tf.linalg.diag(q_diag[:-1], k=1)
    Q = Q + tf.reverse(tf.linalg.diag(q_diag[-1:], k=s*num_years-1), (0,))
    Q = Q + Q_jitter
    Q = tf.repeat(Q[None, ...], n - 1, axis=0)
    trans_mat = tf.linalg.expm((d[1:] - d[:-1])[..., None, None] * Q)
    return trans_mat

def get_obs_mean_given_time():
    means = tf.exp(log_scale)*sinusoidal_basis + bias
    return tf.tile(means, (num_years,))

def get_obs_std_dev_given_time():
    return tf.tile(tf.exp(log_sigma), (num_years,))

def get_time_given_depth_model():
    obs_dist = tfd.Normal(
        loc=get_obs_mean_given_time(),
        scale=get_obs_std_dev_given_time()
    )

    transition_dist = tfd.Categorical(probs=transition_rates_to_transition_matrix())

    return tfd.HiddenMarkovModel(
        initial_distribution=initial_dist,
        transition_distribution=transition_dist,
        observation_distribution=obs_dist,
        num_steps=n,
        time_varying_transition_distribution=True
    )

def log_posterior_with_time_marginalised():
    hmm = get_time_given_depth_model()

    return hmm.log_prob(y) + \
        log_prior(obs_mean_prior_dist, log_scale)  + \
        log_prior(obs_mean_prior_dist, bias)  + \
        log_prior(obs_log_sd_prior_dist, get_obs_std_dev_given_time())  + \
        log_prior(transition_rates_prior, tf.exp(log_transition_rates))

losses = tfp.math.minimize(
    lambda: -log_posterior_with_time_marginalised(),
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    num_steps=300)

hmm = get_time_given_depth_model()
posterior = hmm.posterior_marginals(y).probs_parameter().numpy().T

years = posterior.reshape(num_years, s, n).sum(axis=1)

posterior_plot = np.empty((n_states, len(np.arange(d[0], d[-1], np.median(np.diff(d))))))*np.nan
posterior_plot[:, :50] = posterior[:, :50]
posterior_plot[:, -(n - 50):] = posterior[:, -(n - 50):]

fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3, 3))
ax_a.plot(d[:50], y[:50], c='#17BEBB')
ax_a.plot(d[50:], y[50:], c='#17BEBB')
ax_a.set_ylabel('concentration')
ax_a.get_xaxis().set_visible(False)
ax_a.set_ylim(-2, 3)
ax_a.get_yaxis().set_ticks([])

ax_b.imshow(posterior_plot, extent=[d[0], d[-1], num_years, 0])
ax_b.set_ylabel('year')
ax_b.set_xlabel('depth')
plt.tight_layout()
