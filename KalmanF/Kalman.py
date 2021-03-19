import numpy as np
import pylab as pl
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

from traffic.data.samples import quickstart

#flight = quickstart["AFR27GH"].data
flight = quickstart["RYR3YM"].data
flight = flight[['timestamp', 'latitude', 'longitude']]
flight = flight.iloc[::10, :]

# specify parameters

#interval for measurements
delta_t = 10
#Variances
sigma_p = 1.5
sigma_o = 50

#Transition matrix F
transition_matrix = [[1, 0, delta_t, 0],
                     [0, 1, 0, delta_t],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]

# Observation matrix H
observation_matrix = [[1, 0, 0, 0],
                      [0, 1, 0, 0]]

transition_covariance = [
    [0.25*(delta_t**4)*(sigma_p**2), 0, 0.5*(delta_t**3)*(sigma_p**2), 0],
    [0, 0.25*(delta_t**4)*(sigma_p**2), 0, 0.5*(delta_t**3)*(sigma_p**2)],
    [0, 0, (delta_t**2)*(sigma_p**2), 0],
    [0, 0, 0, (delta_t**2)*(sigma_p**2)]
]

observation_covariance = np.eye(2)*sigma_o**2

initial_state_mean = [flight[['longitude']].values[0],
    flight[['latitude']].values[0], 
    0, 
    0]

initial_state_covariance = np.eye(4)*sigma_o**2

# sample from model
kf = KalmanFilter(
    transition_matrices=transition_matrix,
    observation_matrices=observation_matrix,
    transition_covariance=transition_covariance,
    observation_covariance=observation_covariance,
    initial_state_mean=initial_state_mean,
    initial_state_covariance=initial_state_covariance
)

state_means  = kf.filter(flight[['longitude', 'latitude']].values)[0]

res = [[ i for i, j, k, l in state_means ], 
       [ j for i, j, k, l in state_means ],
       [ k for i, j, k, l in state_means ],
       [ l for i, j, k, l in state_means ]]

plt.figure()
lines_true = plt.scatter(x = flight[['longitude']].values, y= flight[['latitude']].values, color='b')
lines_filt = plt.scatter(x = res[0], y = res[1], color='r')

plt.show()