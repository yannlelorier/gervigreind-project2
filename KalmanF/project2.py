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
# returns a list of flights with the original GPS data
def get_ground_truth_data():
    #names=['liguria', 'pixair_toulouse', 'indiana', 'texas', 'georeal_fyn_island', 'ign_mercantour', 'ign_fontainebleau', 'mecsek_mountains', 'ign_lot_et_garonne', 'inflight_refuelling', 'aircraft_carrier', 'luberon', 'alto_adige', 'franconia', 'danube_valley', 'cevennes', 'oxford_cambridge', 'alpi_italiane', 'rega_zh', 'samu31', 'rega_sg', 'monastir', 'guatemala', 'london_heathrow', 'cardiff', 'sydney', 'brussels_ils', 'ajaccio', 'toulouse', 'noumea', 'london_gatwick', 'perth', 'kota_kinabalu', 'montreal', 'funchal', 'nice', 'munich', 'vancouver', 'lisbon', 'liege_sprimont', 'kiruna', 'bornholm', 'kingston', 'brussels_vor', 'vienna', 'border_control', 'dreamliner_boeing', 'texas_longhorn', 'zero_gravity', 'qantas747', 'turkish_flag', 'airbus_tree', 'easter_rabbit', 'belevingsvlucht', 'anzac_day', 'thankyou', 'vasaloppet']
    names=['munich']
    return [samples.__getattr__(x) for x in names]

# needed for set_lat_lon_from_x_y below
# is set by get_radar_data()
projection_for_flight = {}

# Returns the same list of flights as get_ground_truth_data(), but with the position data modified as if it was a reading from a radar
# i.e., the data is less accurate and with fewer points than the one from get_ground_truth_data()
# The flights in this list will have x, y coordinates set to suitable 2d projection of the lat/lon positions.
# You can access these coordinates in the Flight.data attribute, which is a Pandas DataFrame.
def get_radar_data():
    rng = default_rng()
    radar_error = 0.1 # in kilometers
    radar_altitude_error = 330 # in feet ( ~ 100 meters)
    gt = get_ground_truth_data()
    radar_data = []

    for flight in gt:
        print("flight: %s" % (str(flight)))
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
        # print(flight_radar.data[['longitude', 'latitude']])

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

def kalman_filter(config_tuple, flight):

    if not config_tuple:
        raise NotImplementedError("Config tuple is not initialized, maybe 3 dimensions missing?")

    kf = KalmanFilter(
    transition_matrices=config_tuple[0],
    observation_matrices=config_tuple[1],
    transition_covariance=config_tuple[2],
    observation_covariance=config_tuple[3],
    initial_state_mean=config_tuple[4],
    initial_state_covariance=config_tuple[5]
    )

    state_means  = kf.filter(flight[['x', 'y']].values)[0]

    res = [ [ i for i, j, k, l in state_means ], 
            [ j for i, j, k, l in state_means ],
            [ k for i, j, k, l in state_means ],
            [ l for i, j, k, l in state_means ]]

    return res


#returns a tuple with the configurations
def init_kalman(flight, delta_t=10, sigma_p=1.5, sigma_o=50, dim=2):
    if dim == 2:
        trans_matrix = [[1, 0, delta_t, 0],
                        [0, 1, 0, delta_t],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]

        # Observation matrix H
        obs_matrix = [[1, 0, 0, 0],
                              [0, 1, 0, 0]]

        trans_cov = [
                    [0.25*(delta_t**4)*(sigma_p**2), 0, 0.5*(delta_t**3)*(sigma_p**2), 0],
                    [0, 0.25*(delta_t**4)*(sigma_p**2), 0, 0.5*(delta_t**3)*(sigma_p**2)],
                    [0.5*delta_t**3, 0, (delta_t**2)*(sigma_p**2), 0],
                    [0, 0.5*delta_t**3, 0, (delta_t**2)*(sigma_p**2)]
                    ]
        
        obs_cov = np.eye(2)*sigma_o**2

        #TODO change init velocities
        init_st_mean = [flight['x'].values[0], flight['y'].values[0], 0, 0]

        init_st_cov = np.eye(4)*sigma_o**2

        return (trans_matrix, obs_matrix, trans_cov, obs_cov, init_st_mean, init_st_cov)
    else: return None #TODO 3 dimensions

if __name__ == "__main__":

    radar_data = get_radar_data()

    #create a copy for the calman filter
    flight_original = radar_data[0]
    flight_filtered = copy.deepcopy(radar_data[0])
    
    config = init_kalman(flight_filtered.data)
    res = kalman_filter(config, flight_filtered.data)

    # plt.figure()
    lines_true = plt.scatter(x = flight_original.data[['latitude_true']].values, y= flight_original.data[['longitude_true']].values, color='b')
    lines_meas = plt.scatter(x = flight_original.data[['latitude']].values, y= flight_original.data[['longitude']].values, color='red')
    
    #write filtered data to flights
    flight_filtered.data['x'] = res[0]
    flight_filtered.data['y'] = res[1]
    #convert back the x and y to coordinates
    filtered_flight = set_lat_lon_from_x_y(flight_filtered)
    
    lines_filtered = plt.scatter(x = filtered_flight.data[['latitude']].values, y= filtered_flight.data[['longitude']].values, color='green')
    #plt.show()
    
    mse_noise = ((flight_original.data[['latitude']].values - flight_original.data[['latitude_true']].values)**2 + (flight_original.data[['longitude']].values - flight_original.data[['longitude_true']].values)**2).mean()
    mse_filtered = ((filtered_flight.data[['latitude']].values - flight_original.data[['latitude_true']].values)**2 + (filtered_flight.data[['longitude']].values - flight_original.data[['longitude']].values)**2).mean()

    
    #write the position data in a seperate dataframe
    data = [flight_original.data[['latitude_true']], flight_original.data[['longitude_true']], flight_original.data[['latitude']].rename(columns={'latitude': 'latitude_noise'}), flight_original.data[['longitude']].rename(columns={'longitude': 'longitude_noise'}), filtered_flight.data[['latitude']].rename(columns={'latitude': 'latitude_filtered'}), filtered_flight.data[['longitude']].rename(columns={'longitude': 'longitude_filtered'})]
    position_df = pd.concat(data, axis=1)

    #add the distances to the data frame    
    position_df['distance_filtered_true'] = np.nan
    position_df['distance_noise_true'] = np.nan
    # #for ind in df.index:
    # #print(df['Name'][ind], df['Stream'][ind])

    for i in position_df.index:
        position_df['distance_noise_true'][i] =   distance.distance((position_df['latitude_true'][i],position_df['longitude_true'][i]),(position_df['latitude_noise'][i],position_df['longitude_noise'][i])).m
        position_df['distance_filtered_true'][i] = distance.distance((position_df['latitude_true'][i],position_df['longitude_true'][i]),(position_df['latitude_filtered'][i],position_df['longitude_filtered'][i])).m
    
    print('----------------------------------------')
    print('unfiltered')
    print(position_df['distance_noise_true'].max())
    print(position_df['distance_noise_true'].mean())


    print('----------------------------------------')
    print('filtered')
    print(position_df['distance_filtered_true'].max())
    print(position_df['distance_filtered_true'].mean())


    print('----------------------------------------')
    print((mse_noise-mse_filtered)/mse_noise)