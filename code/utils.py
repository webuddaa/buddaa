from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, FloatType, IntegerType
import datetime, time
from operator import itemgetter

spark = SparkSession.builder.enableHiveSupport().getOrCreate()

def merge_columns(*my_list):
    return list(my_list)
merge_columns_udf = udf(lambda *x: merge_columns(*x), ArrayType(StringType()))

def my_sort(my_list):
    res = []
    for ele in my_list:
        res.append(tuple(ele))
    res = set(res)
    res = sorted(res, key=itemgetter(0))
    return list(res)

def time_to_timestamp(start_time):
    '''
    start_time示例：'2018-11-12 18:23:34'
    '''
    start_time = time.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    return int(time.mktime(start_time))

def timestamp_to_time(timestamp):
    '''
    输出示例：2018-11-12 18:23:34
    '''
    d = datetime.datetime.fromtimestamp(timestamp)
    return d.strftime('%Y-%m-%d %H:%M:%S')

def binarySearch(target, sortedList):
    left = 0
    right = len(sortedList) - 1
    while left <= right:
        mid = (left + right) // 2
        if target == sortedList[mid][0]:
            return mid
        elif target < sortedList[mid][0]:
            right = mid - 1
        else:
            left = mid + 1
    return -1

def fantai_time(start_time, sorted_time_info_list):
    '''
    输出的结果是以小时为单位的
    '''
    index = binarySearch(start_time, sorted_time_info_list)
    if index == len(sorted_time_info_list) - 1:
        return 240.0
    else:
        next_open_time = time.strptime(sorted_time_info_list[index+1][0], '%Y-%m-%d %H:%M:%S')
        this_stop_time = time.strptime(sorted_time_info_list[index][1], '%Y-%m-%d %H:%M:%S')
        d1 = int(time.mktime(next_open_time))
        d2 = int(time.mktime(this_stop_time))
        return (d1-d2) / 3600.0

def fantai_time(start_time, sorted_time_info_list):
    '''
    start_time: 2015-04-19 12:20:00
    sorted_time_info_list: [['2015-04-19 12:20:00', '2015-04-19 12:40:00'], 
                            ['2015-04-20 12:20:00', '2015-04-20 12:30:00'],
                            ['2015-04-23 12:17:30', '2015-04-23 12:56:00'],
                            ['2015-05-19 12:20:00', '2015-05-19 12:40:00']]
    输出的结果是以小时为单位的
    '''
    index = binarySearch(start_time, sorted_time_info_list)
    if index == -1:
        return
    if index == len(sorted_time_info_list) - 1:
        return 240.0
    else:
        next_open_time = datetime.strptime(sorted_time_info_list[index+1][0], '%Y-%m-%d %H:%M:%S')
        this_stop_time = datetime.strptime(sorted_time_info_list[index][1], '%Y-%m-%d %H:%M:%S')
        d1 = next_open_time.timestamp()
        d2 = this_stop_time.timestamp()
        return round((d1 - d2) / 3600, 2) 


import mzgeohash
def geohash_encode(lng, lat):
    try:
        point = (float(lng), float(lat))
        result = mzgeohash.encode(point)
        return result[:7]
    except Exception:
        return


from math import sqrt, asin, sin, cos, radians
def haversine(lonlat1, lonlat2):
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000


import numpy as np
def get_Quartile(l, percent=50):
    '''percent=25, 第一四分位数;
       percent=50, 中位数; 
       percent=75, 第三四分位数;
    '''
    l = np.array(l)
    return float(np.percentile(l, percent))


from shapely.geometry import Polygon, Point
def pointInPolygon(lng_lat, boundings):
    '''
    lng_lat: 'POINT(125.317855 43.873781)'
    boundings: 'POLYGON((115.874513 40.275324,115.901979 40.244933,115.922578 40.276372,115.874513 40.275324))'
    '''
    lng, lat = lng_lat[6:-1].split()
    position_list = boundings[9:-2].split(',')
    temp = []
    for i in position_list:
        temp.append((float(i.split()[0]), float(i.split()[1])))
    polygon = Polygon(temp)
    point = Point(float(lng), float(lat))
    return point.within(polygon)


def compute_how_long(t1,t2):
    try:
        t1 = datetime.strptime(t1.split('.')[0], '%Y-%m-%d %H:%M:%S')
        t2 = datetime.strptime(t2.split('.')[0], '%Y-%m-%d %H:%M:%S')
        return abs(t1.timestamp() - t2.timestamp()) / 3600    # 单位：小时
    except:
        raise ValueError('时间字段的字符串错误！！！')
        return
compute_how_long_udf = udf(lambda x,y: compute_how_long(x,y), FloatType())

