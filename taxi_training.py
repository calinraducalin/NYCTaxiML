import pandas as pd
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from math import radians, cos, sin, asin, sqrt
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import taxi_constants as const
import warnings
plt.rcParams['figure.figsize'] = [14, 8]
warnings.filterwarnings("ignore")



# Loading the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Initial Data Exploration
# print(train.head())
# print(train.describe())
train.info()

# Data Preparation

# Trip Duration Clean-up
# drop very short and too long trips
# keep only trips that are in +- 2 * std
m = np.mean(train[const.f_trip_duration])
s = np.std(train[const.f_trip_duration])
train = train[train[const.f_trip_duration] <= m + 2*s]
train = train[train[const.f_trip_duration] >= m - 2*s]

# Latitude and Longitude clean-up
# city_long_border = (-74.03, -73.75)
# city_lat_border = (40.63, 40.85)
train = train[train[const.f_pickup_longitude] <= -73.75]
train = train[train[const.f_pickup_longitude] >= -74.03]
train = train[train[const.f_pickup_latitude] <= 40.85]
train = train[train[const.f_pickup_latitude] >= 40.63]
train = train[train[const.f_dropoff_longitude] <= -73.75]
train = train[train[const.f_dropoff_longitude] >= -74.03]
train = train[train[const.f_dropoff_latitude] <= 40.85]
train = train[train[const.f_dropoff_latitude] >= 40.63]


# Date clean-up
# Change pickup & dropoff dates format
train[const.f_pickup_datetime] = pd.to_datetime(train.pickup_datetime)
test[const.f_pickup_datetime] = pd.to_datetime(test.pickup_datetime)
train.loc[:, const.f_pickup_date] = train[const.f_pickup_datetime].dt.date
test.loc[:, const.f_pickup_date] = test[const.f_pickup_datetime].dt.date
train[const.f_dropoff_datetime] = pd.to_datetime(train.dropoff_datetime) #Not in Test


# Data Visualisation & Analysis
# Initial Analysis
# plt.hist(train[const.f_trip_duration].values, bins=100)
# plt.xlabel(const.f_trip_duration)
# plt.ylabel('number of train records')


# apply log transformation over trip duration
# in order to get a bell-shaped like (Gaussian distribusion)
# train[const.f_log_trip_duration] = np.log(train[const.f_trip_duration].values + 1)
# plt.hist(train[const.f_log_trip_duration].values, bins=100)
# plt.xlabel('log(trip_duration)')
# plt.ylabel('number of train records')
# sns.distplot(train[const.f_log_trip_duration], bins =100)


# number of trips over time ->
# similar pattern between train and test
# some drops in trips numbars in late Jan and late May
# plt.plot(train.groupby(const.f_pickup_date).count()[[const.f_id]], 'o-', label='train')
# plt.plot(test.groupby(const.f_pickup_date).count()[[const.f_id]], 'o-', label='test')
# plt.title('Trips over Time.')
# plt.legend(loc=0)
# plt.ylabel('Trips')


# Time per vendor -> vendor_id == 2 slightly more average time
# plot_vendor = train.groupby(const.f_vendor_id)[const.f_trip_duration].mean()
# plt.subplots(1,1,figsize=(14,8))
# plt.ylim(ymin=800)
# plt.ylim(ymax=840)
# sns.barplot(plot_vendor.index,plot_vendor.values)
# plt.title('Time per Vendor')
# plt.legend(loc=0)
# plt.ylabel('Time in Seconds')


# Time per store_and_fwd_flag
# in average, about 200 s more when store_and_fwd_flag == 1
# snwflag = train.groupby(const.f_store_and_fwd_flag)[const.f_trip_duration].mean()
# plt.subplots(1,1,figsize=(14,8))
# plt.ylim(ymin=0)
# plt.ylim(ymax=1100)
# plt.title('Time per store_and_fwd_flag')
# plt.legend(loc=0)
# plt.ylabel('Time in Seconds')
# sns.barplot(snwflag.index,snwflag.values)


# Time per number of passengers
# when there is no passenger, the time is significantly lower
# for 1+ passengers, there is not really any difference
# pc = train.groupby(const.f_passenger_count)[const.f_trip_duration].mean()
# plt.subplots(1,1,figsize=(14,8))
# plt.ylim(ymin=0)
# plt.ylim(ymax=1100)
# plt.title('Time per passenger_count')
# plt.legend(loc=0)
# plt.ylabel('Time in Seconds')
# sns.barplot(pc.index,pc.values)

# trips by number of passengers
# print(train.groupby(const.f_passenger_count).size())
# print(test.groupby(const.f_passenger_count).size())


# Coordinate Mapping

city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)

#Pickup locations
# fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
# ax[0].scatter(train[const.f_pickup_longitude].values[:100000], train[const.f_pickup_latitude].values[:100000],
#               color='blue', s=1, label='train', alpha=0.1)
# ax[1].scatter(test[const.f_pickup_longitude].values[:100000], test[const.f_pickup_latitude].values[:100000],
#               color='green', s=1, label='test', alpha=0.1)
# fig.suptitle('Train and test area complete overlap.')
# ax[0].legend(loc=0)
# ax[0].set_ylabel('latitude')
# ax[0].set_xlabel('longitude')
# ax[1].set_xlabel('longitude')
# ax[1].legend(loc=0)
# plt.ylim(city_lat_border)
# plt.xlim(city_long_border)


# Distance and Directionality
# The haversine formula determines the great-circle distance between two points on a sphere given their longitudes and latitudes.

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


train.loc[:, const.f_distance_haversine] = haversine_array(train[const.f_pickup_latitude].values, train[const.f_pickup_longitude].values,
                                                     train[const.f_dropoff_latitude].values,
                                                     train[const.f_dropoff_longitude].values)
test.loc[:, const.f_distance_haversine] = haversine_array(test[const.f_pickup_latitude].values, test[const.f_pickup_longitude].values,
                                                    test[const.f_dropoff_latitude].values, test[const.f_dropoff_longitude].values)

train.loc[:, const.f_distance_dummy_manhattan] = dummy_manhattan_distance(train[const.f_pickup_latitude].values,
                                                                    train[const.f_pickup_longitude].values,
                                                                    train[const.f_dropoff_latitude].values,
                                                                    train[const.f_dropoff_longitude].values)
test.loc[:, const.f_distance_dummy_manhattan] = dummy_manhattan_distance(test[const.f_pickup_latitude].values,
                                                                   test[const.f_pickup_longitude].values,
                                                                   test[const.f_dropoff_latitude].values,
                                                                   test[const.f_dropoff_longitude].values)

train.loc[:, const.f_direction] = bearing_array(train[const.f_pickup_latitude].values, train[const.f_pickup_longitude].values,
                                          train[const.f_dropoff_latitude].values, train[const.f_dropoff_longitude].values)
test.loc[:, const.f_direction] = bearing_array(test[const.f_pickup_latitude].values, test[const.f_pickup_longitude].values,
                                         test[const.f_dropoff_latitude].values, test[const.f_dropoff_longitude].values)




# Create clusters - Neighborhoods

coords = np.vstack((train[[const.f_pickup_latitude, const.f_pickup_longitude]].values,
                    train[[const.f_dropoff_latitude, const.f_dropoff_longitude]].values))
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
train.loc[:, const.f_pickup_cluster] = kmeans.predict(train[[const.f_pickup_latitude, const.f_pickup_longitude]])
train.loc[:, const.f_dropoff_cluster] = kmeans.predict(train[[const.f_dropoff_latitude, const.f_dropoff_longitude]])
test.loc[:, const.f_pickup_cluster] = kmeans.predict(test[[const.f_pickup_latitude, const.f_pickup_longitude]])
test.loc[:, const.f_dropoff_cluster] = kmeans.predict(test[[const.f_dropoff_latitude, const.f_dropoff_longitude]])


#Show previously created clusters
# fig, ax = plt.subplots(ncols=1, nrows=1)
# ax.scatter(train.pickup_longitude.values[:500000], train.pickup_latitude.values[:500000], s=10, lw=0,
#            c=train.pickup_cluster[:500000].values, cmap='autumn', alpha=0.2)
# ax.set_xlim(city_long_border)
# ax.set_ylim(city_lat_border)
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')


# Date extraction
#Extracting Month
train[const.f_Month] = train[const.f_pickup_datetime].dt.month
test[const.f_Month] = test[const.f_pickup_datetime].dt.month
print(train.groupby(const.f_Month).size())
print(test.groupby(const.f_Month).size())

#Extracting Day
train[const.f_DayofMonth] = train[const.f_pickup_datetime].dt.day
test[const.f_DayofMonth] = test[const.f_pickup_datetime].dt.day
print(len(train.groupby(const.f_DayofMonth).size()))
print(len(test.groupby(const.f_DayofMonth).size()))

#Extracting Hour
train[const.f_Hour] = train[const.f_pickup_datetime].dt.hour
test[const.f_Hour] = test[const.f_pickup_datetime].dt.hour
print(len(train.groupby(const.f_Hour).size()))
print(len(test.groupby(const.f_Hour).size()))

#Extracting Day of Week
train[const.f_dayofweek] = train[const.f_pickup_datetime].dt.dayofweek
test[const.f_dayofweek] = test[const.f_pickup_datetime].dt.dayofweek
print(len(train.groupby(const.f_dayofweek).size()))
print(len(test.groupby(const.f_dayofweek).size()))


# Plot average spped depending on
# hour - 9-18 (working hours) - avg speed is low
# day of week (working days) - avg speed is low
# month of year - avg speed slightly drops from winter to summer (more traffic in summer days)

# train.loc[:, const.f_avg_speed_h] = 1000 * train[const.f_distance_haversine] / train[const.f_trip_duration]
# train.loc[:, const.f_avg_speed_m] = 1000 * train[const.f_distance_dummy_manhattan] / train[const.f_trip_duration]
# fig, ax = plt.subplots(ncols=3, sharey=True)
# ax[0].plot(train.groupby(const.f_Hour).mean()[const.f_avg_speed_h], 'bo-', lw=2, alpha=0.7)
# ax[1].plot(train.groupby(const.f_dayofweek).mean()[const.f_avg_speed_h], 'go-', lw=2, alpha=0.7)
# ax[2].plot(train.groupby(const.f_Month).mean()[const.f_avg_speed_h], 'ro-', lw=2, alpha=0.7)
# ax[0].set_xlabel('Hour of Day')
# ax[1].set_xlabel('Day of Week')
# ax[2].set_xlabel('Month of Year')
# ax[0].set_ylabel('Average Speed')
# fig.suptitle('Average Traffic Speed by Date-part')


# Avg speed depending on pick-up locations
train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 3)
train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 3)
# Average speed for regions
gby_cols = ['pickup_lat_bin', 'pickup_long_bin']
coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()
coord_count = train.groupby(gby_cols).count()[['id']].reset_index()
coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)
coord_stats = coord_stats[coord_stats['id'] > 100]
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(train.pickup_longitude.values[:500000], train.pickup_latitude.values[:500000], color='black', s=1, alpha=0.5)
ax.scatter(coord_stats.pickup_long_bin.values, coord_stats.pickup_lat_bin.values, c=coord_stats.avg_speed_h.values,
           cmap='RdYlGn', s=20, alpha=0.5, vmin=1, vmax=8)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.title('Average speed')


plt.show()
