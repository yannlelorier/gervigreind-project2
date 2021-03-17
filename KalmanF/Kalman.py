import numpy as np
import pylab as pl
from pykalman import KalmanFilter

# specify parameters
random_state = np.random.RandomState(0)
#interval for measurements
delta_t = 10

#Transition matrix F
transition_matrix = [[1, delta_t],
                     [0, 1]]
# transition_matrix = [[1, 0, delta_t, 0],
#                      [0, 1, 0, delta_t],
#                      [0, 0, 1, 0],
#                      [0, 0, 0, 1]]
transition_offset = [-0.1, 0.1] #TODO what is this

observation_matrix = np.eye(2)# + random_state.randn(2, 2) * 0.1
# Observation matrix H
# observation_matrix = [[1, 0, 0, 0],
#                       [0, 1, 0, 0]]
observation_offset = [1.0, -1.0] #TODO what is this

transition_covariance = np.eye(2)
# transition_covariance = np.eye(4)
observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1
# observation_covariance = np.eye(4) #+ random_state.randn(4, 4) * 0.1
initial_state_mean = [5, -5]
# initial_state_mean = [[5, -5],
#                       [-5, 5]]
initial_state_covariance = [[1, 0.1], [-0.1, 1]]

# sample from model
kf = KalmanFilter(
    transition_matrix, observation_matrix, transition_covariance,
    observation_covariance, transition_offset, observation_offset,
    initial_state_mean, initial_state_covariance,
    random_state=random_state
)
states, observations = kf.sample(
    n_timesteps=50,
    initial_state=initial_state_mean
)

# estimate state with filtering and smoothing
filtered_state_estimates = kf.filter(observations)[0]
smoothed_state_estimates = kf.smooth(observations)[0]

# draw estimates
pl.figure()
lines_true = pl.plot(states, color='b')
lines_filt = pl.plot(filtered_state_estimates, color='r')
lines_smooth = pl.plot(smoothed_state_estimates, color='g')
pl.legend((lines_true[0], lines_filt[0], lines_smooth[0]),
          ('true', 'filt', 'smooth'),
          loc='lower right'
)
pl.show()