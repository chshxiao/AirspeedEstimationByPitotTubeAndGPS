import numpy as np
import math
import pandas as pd


class KalmanArspGPS:
    def __init__(self, gps_df, arsp_df):
        self.gps_data = gps_df.copy()       # gps data
        self.arsp_data = arsp_df.copy()     # airspeed data
        self.arsp_result = []               # air speed result
        self.wdsp_result = []               # wind speed result
        self.wdhd_result = []               # wind speed heading result
        self.x_prior = np.zeros([5, 1])     # prediction state
        self.x_post = np.zeros([5, 1])      # updated state
        self.x_list = []                    # best estimated states list
        self.z = np.zeros([3, 1])           # measurements
        self.residual = np.zeros([3, 1])    # residual
        self.residual_list = []             # residual list
        self.cv = np.zeros([3, 3])          # residual covariance
        self.p_prior = np.identity(5)       # predicted state covariance
        self.p_post = np.identity(5)        # updated weight covariance
        self.transition = np.identity(5)    # state transition matrix
        self.q = np.identity(5)             # dynamic model covariance matrix
        self.h = np.zeros([3, 1])             # measurement model
        self.h_design = np.zeros([3, 5])      # measurement model design matrix
        self.r = np.zeros([3, 3])             # measurement covariance matrix
        self.__set_initial_state__()


    def __set_initial_state__(self):
        self.x_post[0] = 1                                  # wind speed
        self.x_post[1] = 0                         # wind speed heading
        self.x_post[2] = 0.6                                   # scale factor
        self.x_post[3] = self.gps_data.iloc[0].at['Spd']                    # ground speed
        self.x_post[4] = self.gps_data.iloc[0].at['GCrsRad']          # ground speed heading


    def __cal_measurement_model__(self):
        # first measurement - pressure difference on pitot tube
        self.h[0] = self.x_prior[2] * \
                    (self.x_prior[0] ** 2 + self.x_prior[3] ** 2 -
                     2 * self.x_prior[0] * self.x_prior[3] * math.cos(self.x_prior[1] - self.x_prior[4]))

        # second measurement - ground speed
        self.h[1] = self.x_prior[3]

        # third measurement - ground speed heading
        self.h[2] = self.x_prior[4]


    def __cal_measurement_model_design_matrix__(self):
        # derivative of pressure to wind speed
        self.h_design[0, 0] = 2 * self.x_prior[2] * \
                              (self.x_prior[0] - self.x_prior[3] *
                               math.cos(self.x_prior[1] - self.x_prior[4]))

        # derivative of pressure to wind speed heading
        self.h_design[0, 1] = 2 * self.x_prior[2] * \
                              self.x_prior[3] * self.x_prior[0] * \
                              math.sin(self.x_prior[1] - self.x_prior[4])

        # derivative of pressure to scale factor
        self.h_design[0, 2] = self.x_prior[0] ** 2 + self.x_prior[3] ** 2 - \
                              2 * self.x_prior[0] * self.x_prior[3] * \
                              math.cos(self.x_prior[1] - self.x_prior[4])

        # derivative of pressure to ground speed
        self.h_design[0, 3] = 2 * self.x_prior[2] * \
                              (self.x_prior[3] - self.x_prior[0] *
                               math.cos(self.x_prior[1] - self.x_prior[4]))

        # derivative of pressure to ground speed heading
        self.h_design[0, 4] = - 2 * self.x_prior[2] * \
                              self.x_prior[3] * self.x_prior[0] * \
                              math.sin(self.x_prior[1] - self.x_prior[4])

        # derivative of ground speed
        self.h_design[1, 0] = 0
        self.h_design[1, 1] = 0
        self.h_design[1, 2] = 0
        self.h_design[1, 3] = 1
        self.h_design[1, 4] = 0

        # derivative of ground speed heading
        self.h_design[2, 0] = 0
        self.h_design[2, 1] = 0
        self.h_design[2, 2] = 0
        self.h_design[2, 3] = 0
        self.h_design[2, 4] = 1


    def kalman_filter_process(self):
        for i in range(0, self.gps_data.shape[0]):
            # prediction
            self.x_prior = self.transition @ self.x_post
            self.p_prior = self.transition @ self.p_post @ self.transition.transpose() + self.q

            # set up mesurement covariance matrix
            self.r[0, 0] = 0.01
            self.r[1, 1] = self.gps_data.iloc[i].at['VAcc'] ** 2
            self.r[2, 2] = self.gps_data.iloc[i].at['GCrsAcc'] ** 2

            # set up measurements
            self.z[0] = self.arsp_data.iloc[i].at['DiffPress']
            self.z[1] = self.gps_data.iloc[i].at['Spd']
            self.z[2] = self.gps_data.iloc[i].at['GCrsRad']

            # set up measurements model
            self.__cal_measurement_model__()
            self.__cal_measurement_model_design_matrix__()

            # kalman gain
            k = self.p_prior @ self.h_design.transpose() @ \
                np.linalg.inv(self.h_design @ self.p_prior @ self.h_design.transpose() + self.r)

            # update
            self.residual = self.z - self.h
            self.x_post = self.x_prior + k @ self.residual
            self.p_post = (np.identity(5) - k @ self.h_design) @ self.p_prior
            self.cv = self.r + self.h_design @ self.p_prior @ self.h_design.transpose()

            # output result
            self.x_list.append(self.x_post)


    def get_air_speed(self):
        air_speed = []
        for i in range(0, len(self.x_list)):
            airspd_square = self.x_list[i][3] ** 2 + self.x_list[i][0] ** 2 - \
                            2 * self.x_list[i][3] * self.x_list[i][0] * math.cos(self.x_list[i][1] - self.x_list[i][4])
            air_speed.append(math.sqrt(airspd_square))

        timestamp = self.gps_data.loc[:, 'timestamp'].to_list()
        res = pd.DataFrame({"timestamp": timestamp,
                            "air_speed": air_speed})
        return res


    def get_wind_speed(self):
        wind_speed = []
        for i in range(0, len(self.x_list)):
            wind_speed.append(self.x_list[i][0])

        timestamp = self.gps_data.loc[:, 'timestamp'].to_list()
        res = pd.DataFrame({"timestamp": timestamp,
                            "wind_speed": wind_speed})
        return res


class new_KalmanArspGPS:
    def __init__(self, gps_df, arsp_df):
        self.gps_data = gps_df.copy()       # gps data
        self.arsp_data = arsp_df.copy()     # airspeed data
        self.arsp_result = []               # air speed result
        self.wdsp_result = []               # wind speed result
        self.wdhd_result = []               # wind speed heading result
        self.x_prior = np.zeros([3, 1])     # prediction state
        self.x_post = np.zeros([3, 1])      # updated state
        self.x_list = []                    # best estimated states list
        self.z = np.zeros([1, 1])           # measurements
        self.residual = np.zeros([1, 1])    # residual
        self.residual_list = []             # residual list
        self.cv = np.zeros([1, 1])          # residual covariance
        self.p_prior = np.identity(3)       # predicted state covariance
        self.p_post = np.identity(3)        # updated weight covariance
        self.transition = np.identity(3)    # state transition matrix
        self.q = 0.5 * np.identity(3)             # dynamic model covariance matrix
        self.h = np.zeros([1, 1])             # measurement model
        self.h_design = np.zeros([1, 3])      # measurement model design matrix
        self.r = np.zeros([1, 1])             # measurement covariance matrix
        self.ground_data = np.zeros([2, 1])        # ground speed and heading from GPS
        self.__set_initial_state__()


    def __set_initial_state__(self):
        self.x_post[0] = 1                                  # wind speed
        self.x_post[1] = 0                         # wind speed heading
        self.x_post[2] = 0.6                                   # scale factor


    def __cal_measurement_model__(self):
        # first measurement - pressure difference on pitot tube
        self.h[0] = self.x_prior[2] * \
                    (self.x_prior[0] ** 2 + self.ground_data[0] ** 2 -
                     2 * self.x_prior[0] * self.ground_data[0] * math.cos(self.x_prior[1] - self.ground_data[1]))


    def __cal_measurement_model_design_matrix__(self):
        # derivative of pressure to wind speed
        self.h_design[0, 0] = 2 * self.x_prior[2] * \
                              (self.x_prior[0] - self.ground_data[0] *
                               math.cos(self.x_prior[1] - self.ground_data[1]))

        # derivative of pressure to wind speed heading
        self.h_design[0, 1] = 2 * self.x_prior[2] * \
                              self.ground_data[0] * self.x_prior[0] * \
                              math.sin(self.x_prior[1] - self.ground_data[1])

        # derivative of pressure to scale factor
        self.h_design[0, 2] = self.x_prior[0] ** 2 + self.ground_data[0] ** 2 - \
                              2 * self.x_prior[0] * self.ground_data[0] * \
                              math.cos(self.x_prior[1] - self.ground_data[1])


    def kalman_filter_process(self):
        for i in range(0, self.gps_data.shape[0]):
            # get ground spee and heading from GPS reading
            self.ground_data[0] = self.gps_data.iloc[i].at['Spd']
            self.ground_data[1] = self.gps_data.iloc[i].at['GCrsRad']

            # prediction
            self.x_prior = self.transition @ self.x_post
            self.p_prior = self.transition @ self.p_post @ self.transition.transpose() + self.q

            # set up mesurement covariance matrix
            self.r[0, 0] = 1

            # set up measurements
            self.z[0] = self.arsp_data.iloc[i].at['DiffPress']

            # set up measurements model
            self.__cal_measurement_model__()
            self.__cal_measurement_model_design_matrix__()

            # kalman gain
            k = self.p_prior @ self.h_design.transpose() @ \
                np.linalg.inv(self.h_design @ self.p_prior @ self.h_design.transpose() + self.r)

            # update
            self.residual = self.z - self.h
            self.residual_list.append(self.residual)
            self.x_post = self.x_prior + k @ self.residual
            self.p_post = (np.identity(3) - k @ self.h_design) @ self.p_prior
            self.cv = self.r + self.h_design @ self.p_prior @ self.h_design.transpose()

            # output result
            self.x_list.append(self.x_post)


    def get_air_speed(self):
        air_speed = []
        for i in range(0, len(self.x_list)):
            airspd_square = self.gps_data.iloc[i].at['Spd'] ** 2 + self.x_list[i][0] ** 2 - \
                            2 * self.gps_data.iloc[i].at['Spd'] * self.x_list[i][0] * \
                            math.cos(self.x_list[i][1] - self.gps_data.iloc[i].at['GCrsRad'])
            air_speed.append(math.sqrt(airspd_square))

        timestamp = self.gps_data.loc[:, 'timestamp'].to_list()
        res = pd.DataFrame({"timestamp": timestamp,
                            "air_speed": air_speed})
        return res


    def get_wind_speed(self):
        wind_speed = []
        for i in range(0, len(self.x_list)):
            wind_speed.append(self.x_list[i][0][0])

        timestamp = self.gps_data.loc[:, 'timestamp'].to_list()
        res = pd.DataFrame({"timestamp": timestamp,
                            "wind_speed": wind_speed})
        return res


    def get_wind_speed_heading(self):
        wdsp_heading = []
        for i in range(0, len(self.x_list)):
            wdsp_heading.append(self.x_list[i][1][0])

        timestamp = self.gps_data.loc[:, 'timestamp'].to_list()
        res = pd.DataFrame({"timestamp": timestamp,
                            "wind_speed_heading": wdsp_heading})
        return res


    def get_scale_factor(self):
        sf = []
        for i in range(0, len(self.x_list)):
            sf.append(self.x_list[i][2][0])

        timestamp = self.gps_data.loc[:, 'timestamp'].to_list()
        res = pd.DataFrame({"timestamp": timestamp,
                            "scale_factor": sf})
        return res


    def get_adj_pressure_diff(self):
        pressure_diff = []
        for i in range(0, self.arsp_data.shape[0]):
            pressure_diff.append(self.arsp_data.iloc[i].at['DiffPress'] + self.residual_list[0][0])

        timestamp = self.gps_data.loc[:, 'timestamp'].to_list()
        res = pd.DataFrame({"timestamp": timestamp,
                            "adj_pressure_diff": pressure_diff})
        return res


class new_KalmanArspGPS_bias:
    def __init__(self, gps_df, arsp_df):
        self.gps_data = gps_df.copy()       # gps data
        self.arsp_data = arsp_df.copy()     # airspeed data
        self.arsp_result = []               # air speed result
        self.wdsp_result = []               # wind speed result
        self.wdhd_result = []               # wind speed heading result
        self.x_prior = np.zeros([4, 1])     # prediction state
        self.x_post = np.zeros([4, 1])      # updated state
        self.x_list = []                    # best estimated states list
        self.z = np.zeros([1, 1])           # measurements
        self.residual = np.zeros([1, 1])    # residual
        self.residual_list = []             # residual list
        self.cv = np.zeros([1, 1])          # residual covariance
        self.p_prior = np.identity(4)       # predicted state covariance
        self.p_post = np.identity(4)        # updated weight covariance
        self.transition = np.identity(4)    # state transition matrix
        self.q = np.identity(4)             # dynamic model covariance matrix
        self.h = np.zeros([1, 1])             # measurement model
        self.h_design = np.zeros([1, 4])      # measurement model design matrix
        self.r = np.zeros([1, 1])             # measurement covariance matrix
        self.ground_data = np.zeros([2, 1])        # ground speed and heading from GPS
        self.q[3, 3] = 0                        # change the covarance matrix of bias in model to be zero
        self.__set_initial_state__()


    def __set_initial_state__(self):
        self.x_post[0] = 1                                  # wind speed
        self.x_post[1] = 0                         # wind speed heading
        self.x_post[2] = 0.6                                   # scale factor
        self.x_post[3] = 5                          # bias


    def __cal_measurement_model__(self):
        # first measurement - pressure difference on pitot tube
        self.h[0] = self.x_prior[2] * \
                    (self.x_prior[0] ** 2 + self.ground_data[0] ** 2 -
                     2 * self.x_prior[0] * self.ground_data[0] * math.cos(self.x_prior[1] - self.ground_data[1])) + \
                    self.x_prior[3]


    def __cal_measurement_model_design_matrix__(self):
        # derivative of pressure to wind speed
        self.h_design[0, 0] = 2 * self.x_prior[2] * \
                              (self.x_prior[0] - self.ground_data[0] *
                               math.cos(self.x_prior[1] - self.ground_data[1]))

        # derivative of pressure to wind speed heading
        self.h_design[0, 1] = 2 * self.x_prior[2] * \
                              self.ground_data[0] * self.x_prior[0] * \
                              math.sin(self.x_prior[1] - self.ground_data[1])

        # derivative of pressure to scale factor
        self.h_design[0, 2] = self.x_prior[0] ** 2 + self.ground_data[0] ** 2 - \
                              2 * self.x_prior[0] * self.ground_data[0] * \
                              math.cos(self.x_prior[1] - self.ground_data[1])

        # derivative of pressure to bias
        self.h_design[0, 3] = 1


    def kalman_filter_process(self):
        for i in range(0, self.gps_data.shape[0]):
            # get ground spee and heading from GPS reading
            self.ground_data[0] = self.gps_data.iloc[i].at['Spd']
            self.ground_data[1] = self.gps_data.iloc[i].at['GCrsRad']

            # prediction
            self.x_prior = self.transition @ self.x_post
            self.p_prior = self.transition @ self.p_post @ self.transition.transpose() + self.q

            # set up mesurement covariance matrix
            self.r[0, 0] = 1

            # set up measurements
            self.z[0] = self.arsp_data.iloc[i].at['DiffPress']

            # set up measurements model
            self.__cal_measurement_model__()
            self.__cal_measurement_model_design_matrix__()

            # kalman gain
            k = self.p_prior @ self.h_design.transpose() @ \
                np.linalg.inv(self.h_design @ self.p_prior @ self.h_design.transpose() + self.r)

            # update
            self.residual = self.z - self.h
            self.x_post = self.x_prior + k @ self.residual
            self.p_post = (np.identity(4) - k @ self.h_design) @ self.p_prior
            self.cv = self.r + self.h_design @ self.p_prior @ self.h_design.transpose()

            # output result
            self.x_list.append(self.x_post)


    def get_air_speed(self):
        air_speed = []
        for i in range(0, len(self.x_list)):
            airspd_square = self.gps_data.iloc[i].at['Spd'] ** 2 + self.x_list[i][0] ** 2 - \
                            2 * self.gps_data.iloc[i].at['Spd'] * self.x_list[i][0] * \
                            math.cos(self.x_list[i][1] - self.gps_data.iloc[i].at['GCrsRad'])
            air_speed.append(math.sqrt(airspd_square))

        timestamp = self.gps_data.loc[:, 'timestamp'].to_list()
        res = pd.DataFrame({"timestamp": timestamp,
                            "air_speed": air_speed})
        return res


    def get_wind_speed(self):
        wind_speed = []
        for i in range(0, len(self.x_list)):
            wind_speed.append(self.x_list[i][0])

        timestamp = self.gps_data.loc[:, 'timestamp'].to_list()
        res = pd.DataFrame({"timestamp": timestamp,
                            "wind_speed": wind_speed})
        return res


    def get_wind_speed_heading(self):
        wdsp_heading = []
        for i in range(0, len(self.x_list)):
            wdsp_heading.append(self.x_list[i][1])

        timestamp = self.gps_data.loc[:, 'timestamp'].to_list()
        res = pd.DataFrame({"timestamp": timestamp,
                            "wind_speed_heading": wdsp_heading})
        return res