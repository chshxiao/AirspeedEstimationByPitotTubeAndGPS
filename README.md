# AirspeedEstimationByPitotTubeAndGPS

This program applied Extended Kalman Filter (EKF) to integrate airspeed from pitot tube and ground speed and ground speed heading measurement from GPS measurements. The program predicts the wind speed and wind speed heading with residual on airspeed, ground speed, and ground speed heading.

**Background**
Windspeed, ground speed, and airspeed have the following relationship:

Extended Kalman Filter (EKF) is an algorithm to use multiple measurements to estimate the unknown parameters based on the accuracy of each measurement. The requirements of EKF include:
1. The measurement must fits in the Gaussian or a similar distribution (like high degree of freedom student t-distribution).
2. The state transition matrix of windspeed is random walk, which means the windspeed has a initial value and change randomly. The change follows the Gaussian distribution at a certain covariance.

# Details
The required Python packages are listed in requirements.txt. To install the packages or to check if they are available in the environment, text `pip install -r requirements.txt` in the console.

The program contains 5 .py files:

***readlog***
This is the main function of the program.


***processGPSSignal***
This is the package to process GPS signal. It contains three functions:
1. latlng2utm - converts latitude and longitude to utm x- and y- coordinate in meters. It's currently only applicable for UTM zone 11N from 114W to 120W in WGS84.
2. cal_heading_acc - calculate the accuracy of ground speed heading. The equation is `sigma_heading = sigma_groundspeed / groundspeed`
3. cal_ground_speed - calculate the ground speed using the utm x- and y- coordinates in meters.


***lineOfBestFit***
This is the package to get the line of best fit for detecting potential bias and scale factor in the measurements. It contains two functions:
1. line_of_best_fit - calculates the line of best fit y = mx + b of a 2D data (x, y). It returns the parameter in the format of [m, b, std_m, std_b]
2. significance_of_parameters - tests if the parameters of the line of best fit are statistically zero


***GoodnessofFitText***
This is the package to determine if the corrected measurements fit in the normal distribution. It should be applied after the line of best fit is performed and the bias and scale factor are removed. Normal distribution is the prerequisite of Kalman filter. It contains 4 functions:
1. goodness-of-fit - checks whether a list of values fits in a distribution model. The default distribution used here is the normal distribution
2. group_data - averages the data based on the timestamp interval specified.
3. group_data2 - averages the data based on the timestamp interval specified. This one groups data around each timestamp
4. check_ergodic - check if the data is ergodic (not changing as time goes)


***KalmanFilterGPSArspd***
The package to perform Kalman filter on GPS and Airspeed data. It contains three classes for different scenarios:
1. KalmanArspGPS - this is a five-state kalman filter. The state vector is: [windspeed, windspeed heading, scale factor, ground speed, ground speed heading]
2. new_KalmanArspGPS - this is a three-state kalman filter. The state vector is: [windspeed, windspeed heading, scale factor]
3. new_KalmanArspGPS_bias - this is a four-state kalman filter. The state vector is: [windspeed, windspeed heading, scale factor, airspeed bias]
The dynamic wind speed model is a random walk model. The initial wind speed can be determined by the ground wind speed and heading. Scale factor is the factor from pressure difference to airspeed.
