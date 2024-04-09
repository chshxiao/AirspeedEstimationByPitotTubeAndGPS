import pandas as pd
import numpy as np
from pyproj import CRS, Transformer
import math


def latlng2utm(data, inplace=False):
    """
    Convert Latitude and Longitude to UTM
    :param: data: DataFrame storing lat lon, must have columns named lat and lon
    :param: inplace: whether to inplace the input dataframe
    """
    if 'lat' not in data.columns or 'lon' not in data.columns:
        print("please name the latitude and longitude columns lat and lon")
        return

    # get the datum of WGS1984
    crs_latlng = CRS.from_epsg(4326)            # lat lon coordinate
    crs_utm = CRS.from_epsg(32611)              # WGS84 / UTM Zone 11N

    converter = Transformer.from_crs(crs_from=crs_latlng,
                                     crs_to=crs_utm)

    # get latitude and longitude
    lat = data.loc[:, 'lat'].to_numpy()
    lng = data.loc[:, 'lon'].to_numpy()

    # convert
    [easting, northing] = converter.transform(lat, lng)

    if inplace:
        data['easting'] = easting
        data['northing'] = northing
        return
    else:
        res = data.copy()
        res['easting'] = easting
        res['northing'] = northing
        return res


def cal_heading_acc(data, inplace=False):
    """
    Calculate heading accuracy based on velocity accuracy
    :param data:
    :param inplace:
    """
    heading_acc = []

    for i in range(0, data.shape[0]):
        heading_acc.append(data.iloc[i].at['VAcc'] / data.iloc[i].at['Spd'])

    if inplace:
        data['GCrsAcc'] = heading_acc
        return
    else:
        res = data.copy()
        res['GCrsAcc'] = heading_acc
        return res


def cal_ground_speed(data, inplace=False):
    """
    Calculate the ground speed based on UTM coordinate
    :param: data: Dataframe with UTM columns easting and northing
    """
    if 'easting' not in data.columns or 'northing' not in data.columns:
        print("please name the utm columns easting and northing")
        return

    ground_speed = [None]
    heading = [None]
    easting_diff = 0
    northing_diff = 0
    delta_t = 0
    for i in range(1, data.shape[0] - 1):
        easting_diff = data.at[i+1, 'easting'] - data.at[i-1, 'easting']
        northing_diff = data.at[i+1, 'northing'] - data.at[i-1, 'northing']
        delta_t = data.at[i+1, 'timestamp'] - data.at[i-1, 'timestamp']
        ground_speed.append(math.sqrt(easting_diff ** 2 + northing_diff ** 2) / delta_t)
        heading.append(math.atan2(easting_diff, northing_diff) * 180 / math.pi)

    ground_speed.append(None)
    heading.append(None)

    if inplace:
        data['ground_speed'] = ground_speed
        data['heading'] = heading
        return
    else:
        res = data.copy()
        res['ground_speed'] = ground_speed
        res['heading'] = heading
        return res