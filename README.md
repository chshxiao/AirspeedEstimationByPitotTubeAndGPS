# AirspeedEstimationByPitotTubeAndGPS

This program applied Extended Kalman Filter (EKF) to integrate airspeed from pitot tube and ground speed and ground speed heading measurement from GPS measurements. The program predicts the wind speed and wind speed heading with residual on airspeed, ground speed, and ground speed heading.

**Background**
Windspeed, ground speed, and airspeed have the following relationship:

Extended Kalman Filter (EKF) is an algorithm to use multiple measurements to estimate the unknown parameters based on the accuracy of each measurement. The requirements of EKF include:
1. The measurement must fits in the Gaussian or a similar distribution (like high degree of freedom student t-distribution).
2. The state transition matrix of windspeed is random walk, which means the windspeed has a initial value and change randomly. The change follows the Gaussian distribution at a certain covariance.

# Getting Started
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
