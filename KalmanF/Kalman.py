import numpy as np
import pylab as pl
from pykalman import KalmanFilter

from traffic.data.samples import quickstart

flight = quickstart["AFR27GH"].data
flight = flight[['timestamp', 'latitude', 'longitude']]

# specify parameters
random_state = np.random.RandomState(0)
#interval for measurements
delta_t = 10

#Transition matrix F
# transition_matrix = [[1, delta_t],
#                      [0, 1]]
transition_matrix = [[1, 0, delta_t, 0],
                     [0, 1, 0, delta_t],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]
# transition_offset = [-0.1, 0.1] #TODO what is this
transition_offset = np.zeros((4,1))

# observation_matrix = np.eye(2)# + random_state.randn(2, 2) * 0.1
# Observation matrix H
observation_matrix = [[1, 0, 0, 0],
                      [0, 1, 0, 0]]
# observation_offset = [1.0, -1.0] #TODO what is this
observation_offset = np.zeros((4,1))

# transition_covariance = np.eye(2)
transition_covariance = np.eye(4)
# observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1
observation_covariance = np.eye(4) #+ random_state.randn(4, 4) * 0.1
# initial_state_mean = [5, -5]
initial_state_mean = [flight[['longitude', 'latitude']].values[0],0,0]
# initial_state_covariance = [[1, 0.1], [-0.1, 1]]
initial_state_covariance = np.eye(4)

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

state_means  = kf.filter(flight[['latitude', 'longitude']].values)[0]
# state_means    = meas.flatten()

# estimate state with filtering and smoothing
# filtered_state_estimates = kf.filter(meas)[0]
# smoothed_state_estimates = kf.smooth(meas)[0]

# draw estimates
# pl.figure()
# lines_true = pl.plot(states, color='b')
# lines_filt = pl.plot(meas, color='r')
# lines_smooth = pl.plot(smoothed_state_estimates, color='g')
# pl.legend((lines_true[0], lines_filt[0], lines_smooth[0]),
#           ('true', 'filt', 'smooth'),
#           loc='lower right'
# )

res = [[ i for i, j in state_means ], 
       [ j for i, j in state_means ]]
# print(res) 

pl.figure()
lines_true = pl.scatter(x = flight[['longitude']].values, y= flight[['latitude']].values, color='b')
lines_filt = pl.scatter(x = res[0], y = res[1], color='r')

pl.show()