from pymavlog import MavTLog, MavLog
import pandas as pd
import matplotlib.pyplot as plt
from GoodnessOfFitTest import *
from processGPSSignal import *
from lineOfBestFit import *
from KalmanFilterGPSArspd import *
from accelerometerMeasurement import *
from RandomProcessStatistics import *


# prove weather data model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# weather = pd.read_csv('Weather Station Data.csv')
# wind_speed = weather.loc[:, 'Wind Speed'].to_numpy()
# correlation = []
# for i in range(-5, 5):
#     correlation.append(auto_correlation(wind_speed, i))
#
# plt.plot(correlation)
# plt.show()


# get flight data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# data = MavTLog("2024-03-16 13-20-12.tlog")
data = MavLog("2024-03-16 08-21-58.bin")
data.parse()


# ~~~~~~~~~~~~~~~~~~~~~~~tlog~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gps_imu_integration = data.get("GLOBAL_POSITION_INT")
# speed = data.get("VFR_HUD")
#
# airspd_data = {"timestamp": speed["timestamp"],
#                "airspeed": speed["airspeed"]}
# airspd = pd.DataFrame(airspd_data)
# print(airspd)
#
# grdspd_data = {"timestamp": gps_imu_integration["timestamp"],
#                "grdspd_x": gps_imu_integration["vx"],
#                "grdspd_y": gps_imu_integration["vy"],
#                "grdspd_z": gps_imu_integration["vz"]}
# grdspd = pd.DataFrame(grdspd_data)
# print(grdspd)
# gps_raw = data.get("GPS_RAW_INT")
# gps_data = {"timestamp": gps_raw["timestamp"],
#             "lat": gps_raw["lat"],
#             "lon": gps_raw["lon"]}
# gps_df = pd.DataFrame(gps_data)
# plt.scatter(gps_df['lon'], gps_df['lat'])
# plt.legend()


# ~~~~~~~~~~~~~~~~~~~~~~~bin~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
gps = data.get("GPS")
gpa = data.get("GPA")
air_speed = data.get("ARSP")
imu = data.get("IMU")
gps_data = {"timestamp": gps['timestamp'],
            "Spd": gps['Spd'],
            "Lat": gps['Lat'],
            "Lng": gps['Lng'],
            "GCrs": gps['GCrs'],
            "HAcc": gpa['HAcc'],
            "VAcc": gpa['VAcc']}
arsp_data = {"timestamp": air_speed['timestamp'],
             "Airspeed": air_speed['Airspeed'],
             "DiffPress": air_speed['DiffPress']}
imu_data = {"timestamp": imu['timestamp'],
            # "time_us": imu['TimeUS'],
            "AccX": imu['AccX'],
            "AccY": imu['AccY'],
            "AccZ": imu['AccZ']}
grdsp_df = pd.DataFrame(gps_data)
arsp_df = pd.DataFrame(arsp_data)
imu_df = pd.DataFrame(imu_data)

grdsp_df['GCrsRad'] = grdsp_df['GCrs'] * math.pi / 180
grdsp_df.drop(columns='GCrs', inplace=True)

# plt.plot(arsp_df['timestamp'], arsp_df['Airspeed'], label="airspeed")
# plt.plot(arsp_df['timestamp'], arsp_df['DiffPress'], label="airspeed")
# plt.plot(grdsp_df['timestamp'], grdsp_df['Spd'], label="ground speed")
# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(grdsp_df['Lat'], grdsp_df['Lng'], grdsp_df['timestamp'], c=grdsp_df['Spd'])
# plt.legend()
# plt.show()


# get the statc airspeed data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# static_arsp_df = arsp_df[(arsp_df['timestamp'] > 1710598932) & (arsp_df['timestamp'] < 1710598984)]
# group_static_arsp_df = group_data(static_arsp_df, 1)
# goodness_of_fit(group_static_arsp_df['Airspeed'])
# press_diff = group_static_arsp_df['DiffPress'].std()

# best_fit_line = line_of_best_fit(static_arsp_df['DiffPress'])
# line_y = []
# for i in range(0, static_arsp_df.shape[0]+10):
#     line_y.append(best_fit_line[0] * i + best_fit_line[1])
# print("best_fit_line parameter:")
# print(f"slope: {best_fit_line[0]}")
# print(f"intercept: {best_fit_line[1]}")
# print(f"slope std: {best_fit_line[2]}")
# print(f"intercept std: {best_fit_line[3]}")
# significance_of_parameters(best_fit_line, static_arsp_df.shape[0])

# plt.plot(arsp_df['timestamp'], arsp_df['Airspeed'], label="airspeed")
# plt.plot(arsp_df['timestamp'], arsp_df['DiffPress'], label="pressure diff")

# plt.scatter(range(1, static_arsp_df.shape[0]+1), static_arsp_df['DiffPress'])
# plt.plot(line_y, c='y', label='best fit of line')
# plt.title("air speed error and its best-fit line at static")

# plt.hist(static_arsp_df['Airspeed'])
# plt.legend()
# plt.show()


# get the static ground speed data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# static_grdsp_df = grdsp_df[(grdsp_df['timestamp'] > 1710599026) & (grdsp_df['timestamp'] < 1710599050)] # outdoor works for 1s
# group_static_grdsp_df = group_data2(static_grdsp_df, 1)
# goodness_of_fit(group_static_grdsp_df['gps_speed'])


# get the static IMU data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# static_imu_df = imu_df[(imu_df['timestamp'] > 1710599027) & (imu_df['timestamp'] < 1710599042)]
# static_imu_df.drop_duplicates(subset='timestamp', inplace=True)
# static_imu_df['acc'] = static_imu_df.apply(lambda row: math.sqrt(row.AccX ** 2 + row.AccY ** 2 + row.AccZ ** 2 ), axis=1)
# mean_acc = static_imu_df['acc'].mean()
# static_imu_df['acc_error'] = static_imu_df.apply(lambda row: row.acc - mean_acc, axis=1)

# test_imu_df = imu_df[(imu_df['time_us'] >= 1395358679) & (imu_df['time_us'] <= 1457798692)]
# test_imu_df = imu_df[(imu_df['timestamp'] > 1710601109) & (imu_df['timestamp'] < 1710601267)]
# test_imu_df['acc'] = test_imu_df.apply(lambda row: math.sqrt(row.AccX ** 2 + row.AccY ** 2 + row.AccZ ** 2 ), axis=1)

# best_fit_line = line_of_best_fit(static_imu_df['acc_error'])
# line_y = []
# for i in range(0, static_imu_df.shape[0]+10):
#     line_y.append(best_fit_line[0] * i + best_fit_line[1])
# print("best_fit_line parameter:")
# print(f"slope: {best_fit_line[0]}")
# print(f"intercept: {best_fit_line[1]}")
# print(f"slope std: {best_fit_line[2]}")
# print(f"intercept std: {best_fit_line[3]}")
# significance_of_parameters(best_fit_line, static_imu_df.shape[0])

# plt.plot(static_imu_df['timestamp'], static_imu_df['acc'])
# plt.scatter(range(1, static_imu_df.shape[0]+1), static_imu_df['acc_error'])
# plt.plot(line_y, c='y', label='best fit of line')
# plt.title("accelerometer error and its best-fit line at static")
# plt.legend()
# plt.show()

# g = find_gravity(static_imu_df)
# print(g)
# g = [0, 0, -10.0301]
# acc_measurement = AcceleratorMeasurementSet(test_imu_df)
# acc_measurement.set_gravity(g)
# acc_measurement.remove_gravity()
# static_imu_df_no_g_df = acc_measurement.get_no_gravity_acceleration()
#
# plot_list = []
# for i in range(0, len(static_imu_df_no_g_df)):
#     plot_list.append(-static_imu_df_no_g_df[i][2])
# plt.plot(plot_list)
# plt.show()


# get the cruising airspeed and GPS data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# cruise_arsp_df = arsp_df[(arsp_df['timestamp'] > 1710603520) & (arsp_df['timestamp'] < 1710603540)]
# cruise_arsp_df = arsp_df[(arsp_df['timestamp'] > 1710603440) & (arsp_df['timestamp'] < 1710603600)]
# group_cruise_arsp_df = group_data2(cruise_arsp_df, 1)
# group_cruise_arsp_df = group_data(cruise_arsp_df, 1)

# cruise_grdsp_df = grdsp_df[(grdsp_df['timestamp'] > 1710603440) & (grdsp_df['timestamp'] < 1710603600)]
# group_cruise_grdsp_df = group_data2(cruise_grdsp_df, 1)
# group_cruise_grdsp_df = group_data(cruise_grdsp_df, 1)
# cal_heading_acc(group_cruise_grdsp_df, inplace=True)                                      # calculate the heading accuracy

# plt.plot(group_cruise_arsp_df['timestamp'], group_cruise_arsp_df['Airspeed'], label="airspeed")
# plt.plot(group_cruise_grdsp_df['timestamp'], group_cruise_grdsp_df['Spd'], label="ground speed")
# plt.legend()
# plt.show()


# Kalman filter ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# kf = new_KalmanArspGPS_bias(group_cruise_grdsp_df, group_cruise_arsp_df)
# kf.kalman_filter_process()
# airspd = kf.get_air_speed()
# windspd = kf.get_wind_speed()
# wdspd_heading = kf.get_wind_speed_heading()
#
# fig, ax = plt.subplots(2)
# ax[0].plot(group_cruise_arsp_df['timestamp'], group_cruise_arsp_df['Airspeed'], label="air speed")
# ax[0].plot(group_cruise_grdsp_df['timestamp'], group_cruise_grdsp_df['Spd'], label='ground speed')
# ax[0].plot(airspd['timestamp'], airspd['air_speed'], label="EKF airspeed")
# ax[0].plot(windspd['timestamp'], windspd['wind_speed'], label="EKF windspeed")
# ax[1].plot(wdspd_heading['timestamp'], wdspd_heading['wind_speed_heading'], label="EKF wind speed heading")
# ax[1].plot(group_cruise_grdsp_df['timestamp'], group_cruise_grdsp_df['GCrsRad'], label="Ground heading")
# ax[0].legend()
# ax[1].legend()
# plt.show()


# plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plt.plot(gps_data['timestamp'], gps_data['gps_speed'], label="gps ground speed")
# plt.plot(arsp_df['timestamp'], arsp_df['air_speed'], label="air speed")

# plt.plot(group_cruise_arsp_df['timestamp'], group_cruise_arsp_df['air_speed'], label="air speed")
# plt.plot(group_cruise_grdsp_df['timestamp'], group_cruise_grdsp_df['gps_speed'], label="ground speed")

# plt.hist(group_static_arsp_df['air_speed'], bins=14)
# plt.hist(group_static_grdsp_df['gps_speed'], bins=12)
# plt.title("Pitot tube at-static measurement distribution")

# plt.legend()
# plt.show()

