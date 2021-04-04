from traffic.core import flight
from traffic.data import samples
from geopy import distance
from geopy.distance import geodesic
from numpy.random import default_rng
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import copy
import sys

# returns a list of flights with the original GPS data
def get_ground_truth_data(names):
    #names=['liguria', 'pixair_toulouse', 'indiana', 'texas', 'georeal_fyn_island', 'ign_mercantour', 'ign_fontainebleau', 'mecsek_mountains', 'ign_lot_et_garonne', 'inflight_refuelling', 'aircraft_carrier', 'luberon', 'alto_adige', 'franconia', 'danube_valley', 'cevennes', 'oxford_cambridge', 'alpi_italiane', 'rega_zh', 'samu31', 'rega_sg', 'monastir', 'guatemala', 'london_heathrow', 'cardiff', 'sydney', 'brussels_ils', 'ajaccio', 'toulouse', 'noumea', 'london_gatwick', 'perth', 'kota_kinabalu', 'montreal', 'funchal', 'nice', 'munich', 'vancouver', 'lisbon', 'liege_sprimont', 'kiruna', 'bornholm', 'kingston', 'brussels_vor', 'vienna', 'border_control', 'dreamliner_boeing', 'texas_longhorn', 'zero_gravity', 'qantas747', 'turkish_flag', 'airbus_tree', 'easter_rabbit', 'belevingsvlucht', 'anzac_day', 'thankyou', 'vasaloppet']
    # names=['munich']
    return [samples.__getattr__(x) for x in names]

# needed for set_lat_lon_from_x_y below
# is set by get_radar_data()
projection_for_flight = {}

# Returns the same list of flights as get_ground_truth_data(), but with the position data modified as if it was a reading from a radar
# i.e., the data is less accurate and with fewer points than the one from get_ground_truth_data()
# The flights in this list will have x, y coordinates set to suitable 2d projection of the lat/lon positions.
# You can access these coordinates in the Flight.data attribute, which is a Pandas DataFrame.
def get_radar_data(names):
    rng = default_rng()
    radar_error = 0.1 # in kilometers
    radar_altitude_error = 330 # in feet ( ~ 100 meters)
    gt = get_ground_truth_data(names)
    radar_data = []

    for flight in gt:
        # print("flight: %s" % (str(flight)))
        flight_radar = flight.resample("10s")
        # print(flight_radar.data[['longitude', 'latitude']])
        flight_radar.data['longitude_true'] = flight_radar.data['longitude']
        flight_radar.data['latitude_true'] = flight_radar.data['latitude']


        for i in range(len(flight_radar.data)):
            point = geodesic(kilometers=rng.normal()*radar_error).destination((flight_radar.data.at[i,"latitude"], flight_radar.data.at[i,"longitude"]), rng.random()*360)
            (flight_radar.data.at[i,"latitude"], flight_radar.data.at[i,"longitude"]) = (point.latitude, point.longitude)
            flight_radar.data.at[i,"altitude"] += rng.normal()*radar_altitude_error
            # print("after: %f, %f" % (flight_radar.data.at[i,"latitude"], flight_radar.data.at[i,"longitude"]))

        projection = pyproj.Proj(proj="lcc", ellps="WGS84", lat_1=flight_radar.data.latitude.min(), lat_2=flight_radar.data.latitude.max(), lat_0=flight_radar.data.latitude.mean(), lon_0=flight_radar.data.longitude.mean())
        flight_radar = flight_radar.compute_xy(projection)
        flightid = flight_radar.callsign + str(flight_radar.start)

        if flightid in projection_for_flight:
            print("ERROR: duplicate flight ids: %s" % (flightid))
        projection_for_flight[flight_radar.callsign + str(flight_radar.start)]=projection
        radar_data.append(flight_radar)
    return radar_data

# returns the same flight with latitude and longitude changed to reflect the x, y positions in the data
# The intended use of this function is to:
#  1. make a copy of a flight that you got from get_radar_data
#  2. use a kalman filter on that flight and set the x, y columns of the data to the filtered positions
#  3. call set_lat_lon_from_x_y() on that flight to set its latitude and longitude columns according to the filitered x,y positions
# Step 3 is necessary, if you want to plot the data, because plotting is based on the lat/lon coordinates.
def set_lat_lon_from_x_y(flight):
    flightid = flight.callsign + str(flight.start)
    projection = projection_for_flight[flightid]
    if projection is None:
        print("No projection found for flight %s. You probably did not get this flight from get_radar_data()." % (flightid))
    
    lons, lats = projection(flight.data["x"], flight.data["y"], inverse=True)
    flight.data["longitude"] = lons
    flight.data["latitude"] = lats
    return flight

#shows a scatter plot (kalman vs noise vs true)
def show_kalman_plot(flight_original, flight_filtered, flight_smoothed):
    plt.figure()
    plt.title('Kalman filter flight tracking')
    plt.scatter(x=flight_original.data[['latitude']].values, y=flight_original.data[['longitude']].values, color='r', alpha=0.8, label='unfiltered')
    plt.scatter(x=flight_filtered.data[['latitude']].values, y=flight_filtered.data[['longitude']].values, color='g', alpha=0.8, label='filtered')
    plt.scatter(x=flight_smoothed.data[['latitude']].values, y=flight_smoothed.data[['longitude']].values, color='violet', alpha=0.8, label='smoothed')
    plt.scatter(x=flight_original.data[['latitude_true']].values, y=flight_original.data[['longitude_true']].values, color='gray', alpha=0.9, label='true')
    plt.legend(loc='upper left')
    #allows for non-blocking showing of window
    plt.draw()
    plt.pause(0.001)

def kalman_smooth(config_tuple, flight):
    kf = KalmanFilter(
    transition_matrices=config_tuple[0],
    observation_matrices=config_tuple[1],
    transition_covariance=config_tuple[2],
    observation_covariance=config_tuple[3],
    initial_state_mean=config_tuple[4],
    initial_state_covariance=config_tuple[5]
    )

    state_means = kf.smooth(flight[['x', 'y']].values)[0]

    res = [ [ i for i, _, _, _ in state_means ], 
            [ j for _, j, _, _ in state_means ],
            [ k for _, _, k, _ in state_means ],
            [ l for _, _, _, l in state_means ]]

    return res

def kalman_filter(config_tuple, flight):

    if not config_tuple:
        raise NotImplementedError("Config tuple is not initialized, maybe 3 dimensions implementation is missing?")

    kf = KalmanFilter(
    transition_matrices=config_tuple[0],
    observation_matrices=config_tuple[1],
    transition_covariance=config_tuple[2],
    observation_covariance=config_tuple[3],
    initial_state_mean=config_tuple[4],
    initial_state_covariance=config_tuple[5]
    )

    state_means  = kf.filter(flight[['x', 'y']].values)[0]

    res = [ [ i for i, _, _, _ in state_means ], 
            [ j for _, j, _, _ in state_means ],
            [ k for _, _, k, _ in state_means ],
            [ l for _, _, _, l in state_means ]]

    return res

#returns a tuple with the initial configurations of the Kalman Filter
def init_kalman(flight, delta_t=10, sigma_p=1.5, sigma_o=50, dim=2):
    if dim == 2:
        #F
        trans_matrix = [[1, 0, delta_t, 0],
                        [0, 1, 0, delta_t],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]

        # Observation matrix H
        obs_matrix = [[1, 0, 0, 0],
                      [0, 1, 0, 0]]
        #Q
        trans_cov = [
                    [0.25*(delta_t**4)*(sigma_p**2), 0, 0.5*(delta_t**3)*(sigma_p**2), 0],
                    [0, 0.25*(delta_t**4)*(sigma_p**2), 0, 0.5*(delta_t**3)*(sigma_p**2)],
                    [(0.5*delta_t**3)*(sigma_p**2), 0, (delta_t**2)*(sigma_p**2), 0],
                    [0, (0.5*delta_t**3)*(sigma_p**2), 0, (delta_t**2)*(sigma_p**2)]
                    ]
        #R
        obs_cov = [
                   [(0.25*delta_t**4)*(sigma_p**2), 0],
                   [0, (0.25*delta_t**4)*(sigma_p**2)]
                  ]

        #TODO change init velocities
        init_st_mean = [flight['x'].values[0], flight['y'].values[0], 0, 0]

        init_st_cov = np.eye(4)*sigma_o**2

        return (trans_matrix, obs_matrix, trans_cov, obs_cov, init_st_mean, init_st_cov)
    else: return None #TODO 3 dimensions (bonus)

############## Main ################
if __name__ == "__main__":
    # names=['liguria', 'pixair_toulouse', 'indiana', 'texas', 'georeal_fyn_island', 'ign_mercantour', 'ign_fontainebleau', 'mecsek_mountains', 'ign_lot_et_garonne', 'inflight_refuelling', 'aircraft_carrier', 'luberon', 'alto_adige', 'franconia', 'danube_valley', 'cevennes', 'oxford_cambridge', 'alpi_italiane', 'rega_zh', 'samu31', 'rega_sg', 'monastir', 'guatemala', 'london_heathrow', 'cardiff', 'sydney', 'brussels_ils', 'ajaccio', 'toulouse', 'noumea', 'london_gatwick', 'perth', 'kota_kinabalu', 'montreal', 'funchal', 'nice', 'munich', 'vancouver', 'lisbon', 'liege_sprimont', 'kiruna', 'bornholm', 'kingston', 'brussels_vor', 'vienna', 'border_control', 'dreamliner_boeing', 'texas_longhorn', 'zero_gravity', 'qantas747', 'turkish_flag', 'airbus_tree', 'easter_rabbit', 'belevingsvlucht', 'anzac_day', 'thankyou', 'vasaloppet']
    names = [str(sys.argv[1])]
    # tot_maxi_mean_noised = 0
    # tot_mean_mean_noised = 0

    # tot_maxi_mean_filtered = 0
    # tot_mean_mean_filtered = 0

    radar_data = get_radar_data(names)

    #create a copy for the kalman filter
    # for i in range(len(names)):

    flight_original = radar_data[0]
    flight_filtered = copy.deepcopy(radar_data[0])
    flight_smoothed = copy.deepcopy(radar_data[0])


    config = init_kalman(flight_filtered.data, dim=2, sigma_o=80, sigma_p=1.5)
    res = kalman_filter(config, flight_filtered.data)
    res_smoothed = kalman_smooth(config, flight_smoothed.data)

    #save filtered and smoothed data
    flight_filtered.data['x'] = res[0] #x
    flight_filtered.data['y'] = res[1] #y

    flight_smoothed.data['x'] = res_smoothed[0]
    flight_smoothed.data['y'] = res_smoothed[1]
    #no velocity components ?
    #convert back the x and y to coordinates
    flight_filtered = set_lat_lon_from_x_y(flight_filtered)
    flight_smoothed = set_lat_lon_from_x_y(flight_smoothed)

    show_kalman_plot(flight_original, flight_filtered, flight_smoothed)
    
    mse_noised = ((flight_original.data[['latitude']].values - flight_original.data[['latitude_true']].values)**2 + (flight_original.data[['longitude']].values - flight_original.data[['longitude_true']].values)**2).mean()
    mse_filtered = ((flight_filtered.data[['latitude']].values - flight_original.data[['latitude_true']].values)**2 + (flight_filtered.data[['longitude']].values - flight_original.data[['longitude']].values)**2).mean()
    mse_smoothed = ((flight_smoothed.data[['latitude']].values - flight_original.data[['latitude_true']].values)**2 + (flight_smoothed.data[['longitude']].values - flight_original.data[['longitude']].values)**2).mean()
    #write the position data in a separate dataframe
    tmp_data = [
        flight_original.data[['latitude_true']], 
        flight_original.data[['longitude_true']], 
        flight_original.data[['latitude']].rename(columns={'latitude': 'latitude_noised'}), 
        flight_original.data[['longitude']].rename(columns={'longitude': 'longitude_noised'}), 
        flight_filtered.data[['latitude']].rename(columns={'latitude': 'latitude_filtered'}), 
        flight_filtered.data[['longitude']].rename(columns={'longitude': 'longitude_filtered'}),
        flight_smoothed.data[['latitude']].rename(columns={'latitude':'latitude_smoothed'}),
        flight_smoothed.data[['longitude']].rename(columns={'longitude':'longitude_smoothed'})
        ]
    position_df = pd.concat(tmp_data, axis=1)

    position_df['distance_filtered_true'] = np.nan
    position_df['distance_noised_true'] = np.nan
    position_df['distance_smoothed_true'] = np.nan

    for i in position_df.index:
        #distance betweeen noised data and the real data
        position_df['distance_noised_true'][i] = distance.distance((position_df['latitude_true'][i], position_df['longitude_true'][i]), (position_df['latitude_noised'][i], position_df['longitude_noised'][i])).m
        #distance between the filtered data and the real data
        position_df['distance_filtered_true'][i] = distance.distance((position_df['latitude_true'][i],position_df['longitude_true'][i]),(position_df['latitude_filtered'][i],position_df['longitude_filtered'][i])).m

        position_df['distance_smoothed_true'][i] = distance.distance((position_df['latitude_true'][i],position_df['longitude_true'][i]),(position_df['latitude_smoothed'][i],position_df['longitude_smoothed'][i])).m
    # tot_maxi_mean_noised += position_df['distance_noised_true'].max()
    # tot_mean_mean_noised += position_df['distance_noised_true'].mean()
    # tot_maxi_mean_filtered += position_df['distance_filtered_true'].max()
    # tot_mean_mean_filtered += position_df['distance_filtered_true'].mean()

    print(f"FLIGHT {names[0]}:")
    print('----------------------------------------------------')
    print('Unfiltered:')
    print(f"\t> Maxi Noised-true distance = {position_df['distance_noised_true'].max()} metres \n\t> Noised-true distance mean = {position_df['distance_noised_true'].mean()} metres")

    print('----------------------------------------------------')
    print('Filtered')
    print(f"\t> Maxi Filtered-true distance = {position_df['distance_filtered_true'].max()} metres\n\t> Filtered-true distance mean = {position_df['distance_filtered_true'].mean()} metres")

    print('----------------------------------------------------')
    print('Smoothed')
    print(f"\t> Maxi smooth-true distance = {position_df['distance_smoothed_true'].max()} metres\n\t> smoothed-true distance mean = {position_df['distance_smoothed_true'].mean()} metres")
    print('----------------------------------------------------')
    print(f"\t> MSE for noised and filtered data = {(mse_noised-mse_filtered)/mse_noised}")
    print('----------------------------------------------------')
    print(f"\t> MSE for smoothed and filtered data = {(mse_noised-mse_smoothed)/mse_noised}")
    
    # testing all flights
    
    # print("Results")
    # print(f"tot_maxi_mean_noised\n\t>{tot_maxi_mean_noised/len(names)}")
    # print(f"tot_mean_mean_noised\n\t>{tot_mean_mean_noised/len(names)}")
    # print(f"tot_maxi_mean_filtered\n\t>{tot_maxi_mean_filtered/len(names)}")
    # print(f"tot_mean_mean_filtered\n\t>{tot_mean_mean_filtered/len(names)}")
