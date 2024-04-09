import pandas as pd
import scipy as sp
import numpy as np
import math


def goodness_of_fit(x, confidence_level=0.95):
    """
    This function checks whether a list of values fit in a distribution model
    The default distribution used here is the normal distribution
    :param x:
    :return:
    """

    siz_x = x.shape[0]
    mean_x = np.mean(x)
    std_x = np.std(x)

    standardized_res_x = (x - mean_x)/std_x
    test_sort = np.sort(standardized_res_x)

    # get the intervals
    min_res = np.min(standardized_res_x)
    max_res = np.max(standardized_res_x)

    low_bound = math.ceil(min_res/0.4)
    up_bound = math.ceil(max_res/0.4)

    interval = []
    for i in range(low_bound, up_bound+1):
        interval.append(i * 0.4)

    # get the actual frequencies in each interval
    actual_freq = []
    for i in range(0, len(interval)):
        actual_freq.append((standardized_res_x < interval[i]).sum() - (standardized_res_x < (interval[i]-0.4)).sum())

    # get the expected frequencies in each interval
    expected_prob = []
    for i in range(0, len(interval)):
        if i == 0:
            expected_prob.append(sp.stats.norm.cdf(interval[i]))
        elif i == len(interval)-1:
            expected_prob.append(1-sp.stats.norm.cdf(interval[i-1]))
        else:
            expected_prob.append(sp.stats.norm.cdf(interval[i]) - sp.stats.norm.cdf(interval[i-1]))
    expected_freq = [i * siz_x for i in expected_prob]

    result_tb = pd.DataFrame({'lowerbound': interval, 'expected prob': expected_prob,
                              'expected frequency': expected_freq, 'actual frequency': actual_freq})
    print(result_tb)

    # calculate the X^2
    x2 = 0
    for i in range(0, len(interval)):
        x2 = x2 + (actual_freq[i] - expected_freq[i])**2 / expected_freq[i]

    print('X2 = ', x2)
    # compare X^2 and chi-square (95% confidence level)
    dof = len(interval) - 3                     # normal distribution 2 unknowns (mean, std)

    chi_sq = sp.stats.chi2.ppf(confidence_level, dof)
    print('chisquare = ', chi_sq)

    if x2 < chi_sq:
        print("good-of-fitness test pass")
    else:
        print("good-of-fitness test fail")


def group_data(x, interval, start_time=-1):
    """
    This function groups the data baed on the timestamp interval
    The result should fit in Gaussian distribution
    :param x: 2D dataframe with the first columns to be timestamp
    :return: 2D dataframe after the grouping
    """
    siz = x.shape[0]
    num_col_data = x.shape[1] - 1               # number of columns

    timestamp_list = []
    data_list = []

    if start_time == -1:
        begin_t = x.iat[0, 0]  # the beginning of this interval
        start_index = 0
    else:
        begin_t = start_time
        for start_index in range(0, siz):
            if x.iat[0, 0] >= begin_t:
                break

    end_t = begin_t + interval              # the expected end of this interval
    data_count = 1                          # the number of data in this interval

    data_sum = [None] * num_col_data
    for j in range(0, num_col_data):
        data_sum[j] = x.iat[start_index, j+1]

    start_index = start_index + 1           # start the iteration from the next element

    for i in range(start_index, siz):
        # the end of the interval
        if x.iat[i, 0] > end_t:
            timestamp_list.append((end_t + begin_t)/2)        # the average of timestamp of this interval
            temp = data_sum.copy()
            for j in range(0, num_col_data):
                temp[j] = temp[j] / data_count
            temp.insert(0, (end_t + begin_t)/2)
            data_list.append(temp)

            begin_t = end_t           # set up the next interval
            end_t = begin_t + interval

            data_count = 1
            for j in range(0, num_col_data):
                data_sum[j] = x.iat[i, j + 1]

        # in the interval
        else:
            data_count = data_count + 1
            for j in range(0, num_col_data):
                data_sum[j] = data_sum[j] + x.iat[i, j+1]

    res = pd.DataFrame(data_list, columns=x.columns.values)

    return res


def group_data2(x, interval):
    siz = x.shape[0]
    last_timestamp = x.iloc[:, 0].max()

    timestamp_list = []
    data_list = []

    for i in range(0, siz):
        # set up the interval
        begin_t = x.iat[i, 0]                       # the beginning of this interval
        end_t = begin_t + interval                  # the expected end of this interval

        # jump out of the loop if the interval pass the last timestamp
        if end_t > last_timestamp:
            break

        data_count = 1                              # the number of data in this interval
        data_sum = x.iat[i, 1]                      # the sum of data in this interval

        for j in range(i + 1, siz):
            # the end of the interval
            if x.iat[j, 0] > end_t:
                data_list.append(data_sum / data_count)  # the average of data in this interval
                timestamp_list.append((x.iat[j, 0] + begin_t) / 2)  # the average of timestamp of this interval
                break

            # in the interval
            else:
                data_count = data_count + 1
                data_sum = data_sum + x.iat[j, 1]

    res = pd.DataFrame({'timestamp': timestamp_list,
                        x.columns.values[1]: data_list})

    return res


def check_ergodic(x):
    """
    Check if the data is ergodic (not changing as time goes)
    :param x: a Series of data
    :return:
    """
    siz = x.shape[0]

    mean_list = []

    for i in range(0, siz):
        mean_list.append(x[i:].mean())

    return mean_list


