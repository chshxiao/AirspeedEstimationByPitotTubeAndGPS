import scipy as sp
import numpy as np
import math
from matplotlib import pyplot as plt


def random_process_mean(x):
    """
    This function returns the mean as a function of t for the random process
    """

    siz = x.shape[0]
    m = np.zeros([siz])
    # m = []

    for i in range(0, siz):
        m[i] = np.mean(x[i:])

    return m


def auto_correlation(x, tao):
    """
    This function calculates the auto-correlation of a random process.
    Rxx(t1, tao) = 1/N * sum( x(t1)) * x(t1+tao) )
    t1 is the starting time and tao is the time lag
    :param x: the random process values (numpy array)
    :param tao: time lag(step) between two targeted measurements
    """
    mean = np.mean(x)
    siz = x.shape[0]

    rxx = 0
    rxx_0 = 0

    if tao >= 0:
        for i in range(0, siz):
            rxx_0 = rxx_0 + (x[i] - mean)**2
            if (i + tao) < siz:
                rxx = rxx + (x[i] - mean) * (x[i + tao] - mean)
    else:
        for i in range(0, siz):
            rxx_0 = rxx_0 + (x[i] - mean) ** 2
            if (i + tao) >= 0:
                rxx = rxx + (x[i] - mean) * (x[i + tao] - mean)

    return rxx / rxx_0


def cross_correlation(x, y, tao):
    """
    This function calculates the cross-correlation of two random processes.
    Rxy(t1, tao) = 1/N * sum( x(t1)) * y(t1+tao) )
    Two random processes have to be the same size
    t1 is the starting time and tao is the time lag
    :param x: the first random process (numpy array)
    :param y: the second random process (numpy array)
    :param tao: time lag(step) y-x
    """

    siz = x.shape[0]

    rxy = np.zeros([siz])

    # for each t1
    if tao >= 0:
        for i in range(0, siz - tao):
            rxy_sum = 0

            for j in range(i, siz - tao):
                rxy_sum = rxy_sum + x[j] * y[j + tao]       # sum( x(ti) * x(ti+tao) )

            rxy[i] = rxy_sum / (siz - tao - i)              # 1/N * sum( x(ti) * x(ti+tao) )

    elif tao < 0:
        for i in range(-tao, siz):
            rxy_sum = 0

            for j in range(i, siz):
                rxy_sum = rxy_sum + x[j] * y[j + tao]

            rxy[i] = rxy_sum / (siz - i)

    return rxy


def auto_correlation_stationary(x, tao):
    """
    This function calculates the auto-correlation of a stationary random process.
    Rxx(t1, tao) = 1/N * sum( x(t1)) * x(t1+tao) )
    t1 is the starting time and tao is the time lag
    :param x: the random process values (numpy array)
    :param tao: time lag(step) between two targeted measurements
    """
    siz = x.shape[0]

    rxx = 0

    if tao >= 0:
        for i in range(0, siz - tao):
            rxx = rxx + x[i] * x[i + tao]  # sum( x(ti) * x(ti+tao) )
        rxx = rxx / (siz - tao)  # 1/N * sum( x(ti) * x(ti+tao) )

    elif tao < 0:
        for i in range(-tao, siz):
            rxx = rxx + x[i] * x[i + tao]
        rxx = rxx / (siz + tao)

    return rxx


def cross_correlation_stationary(x, y, tao):
    """
    This function calculates the cross-correlation of a stationary random process.
    Rxx(t1, tao) = 1/N * sum( x(t1)) * y(t1+tao) )
    t1 is the starting time and tao is the time lag
    :param x: the first stationary random process (numpy array)
    :param y: the second stationary random process
    :param tao: time lag(step) between two targeted measurements
    """
    siz = x.shape[0]

    rxx = 0

    if tao >= 0:
        for i in range(0, siz - tao):
            rxx = rxx + x[i] * y[i + tao]  # sum( x(ti) * x(ti+tao) )
        rxx = rxx / (siz - tao)  # 1/N * sum( x(ti) * x(ti+tao) )

    elif tao < 0:
        for i in range(-tao, siz):
            rxx = rxx + x[i] * y[i + tao]
        rxx = rxx / (siz + tao)

    return rxx


def auto_covariance_stationary(x, tao):
    """
    This function calculates the auto-covariance of a stationary random process.
    Cxx(tao) = Rxx(tao) - mean^2
    :param x: the random process values (numpy array)
    :param tao: time lag(step) between two targeted measurements
    """

    return auto_correlation_stationary(x, tao) - np.mean(x)**2


def cross_covariance_stationary(x, y, tao):
    """
    This function calculates the cross-covariance of a stationary random process.
    Cxx(tao) = Rxy(tao) - mean(x) * mean(y)
    :param x: the first random process values (numpy array)
    :param y: the second random process values (numpy array)
    :param tao: time lag(step) between two targeted measurements
    """

    return cross_correlation_stationary(x, y, tao) - np.mean(x) * np.mean(y)


def correlation_coefficient(x, y, tao):
    """
    This function returns the correlation coefficient between random processes
    x and y. The time difference is tao
    """

    cxy = cross_covariance_stationary(x, y, tao)
    std_x = math.sqrt(auto_covariance_stationary(x, 0))
    std_y = math.sqrt(auto_covariance_stationary(y, 0))

    return cxy / (std_x * std_y)

