import argparse
from datetime import datetime
from math import sqrt, asin, sin, cos, radians
# from pyspark.sql.functions import udf
# from pyspark.sql.types import StringType, ArrayType, IntegerType, FloatType


parser = argparse.ArgumentParser(description='XXXXX')
parser.add_argument('--low', type=int, default=30)
parser.add_argument('--high', type=int, default=60)
parser.add_argument('--needDays', type=int, default=1)
args = parser.parse_args()

def voltage_to_percent(voltage):
    voltage = int(voltage)
    perd_percent = 0
    if voltage < 322:
        perd_percent = 5
    elif 322 <= voltage <= 403:
        perd_percent = round(1.14 * voltage - 362)
    else:
        perd_percent = 99
    return perd_percent
voltage_to_percent_udf = udf(lambda x: voltage_to_percent(x), IntegerType())

def merge_columns(*my_list):
    return list(my_list)
merge_columns_udf = udf(lambda *x: merge_columns(*x), ArrayType(StringType()))

my_sort_udf = udf(lambda x: sorted(x, key=lambda t: t[0]), ArrayType(ArrayType(StringType())))

def compute_how_long(t1,t2):
    try:
        t1 = datetime.strptime(t1.split('.')[0], '%Y-%m-%d %H:%M:%S')
        t2 = datetime.strptime(t2.split('.')[0], '%Y-%m-%d %H:%M:%S')
        return abs(t1.timestamp() - t2.timestamp()) / 60    # 单位：分钟
    except:
        return 0.
def find_battery_percent(time, sort_list):
    for i in range(len(sort_list)-1):
        if sort_list[i][0] < time < sort_list[i+1][0]:
            t1 = compute_how_long(time, sort_list[i][0])
            t2 = compute_how_long(time, sort_list[i+1][0])
            return int(sort_list[i][1]) if t1 < t2 else int(sort_list[i+1][1])
    t1 = compute_how_long(time, sort_list[0][0])
    t2 = compute_how_long(time, sort_list[-1][0])
    return int(sort_list[0][1]) if t1 < t2 else int(sort_list[-1][1])
find_battery_percent_udf = udf(lambda x,y: find_battery_percent(x,y), IntegerType())

def get_lowPower(l):
    res = []
    for _ in range(24*3):
        res.append(0)
    for time, battery_percent in l:
        index = computeIndexByTime(time)
        if int(battery_percent) < args.low:
            res[index] += 1
    if args.needDays > 1:
        return list(map(lambda x: round(x / args.needDays), res))
    else:
        return res
get_lowPower_udf = udf(lambda x: get_lowPower(x), ArrayType(IntegerType()))

def get_middlePower(l):
    res = []
    for _ in range(24*3):
        res.append(0)
    for time, battery_percent in l:
        index = computeIndexByTime(time)
        if args.low <= int(battery_percent) < args.high:
            res[index] += 1
    if args.needDays > 1:
        return list(map(lambda x: round(x / args.needDays), res))
    else:
        return res
get_middlePower_udf = udf(lambda x: get_middlePower(x), ArrayType(IntegerType()))

def get_highPower(l):
    res = []
    for _ in range(24*3):
        res.append(0)
    for time, battery_percent in l:
        index = computeIndexByTime(time)
        if int(battery_percent) >= args.high:
            res[index] += 1
    if args.needDays > 1:
        return list(map(lambda x: round(x / args.needDays), res))
    else:
        return res
get_highPower_udf = udf(lambda x: get_highPower(x), ArrayType(IntegerType()))


def tag_label(next_time, parsed_log_time):
    t = compute_how_long(next_time, parsed_log_time)
    if t >= 5 or not next_time:
        return 1   # 代表缺电影响订单
    else:
        return 0
tag_label_udf = udf(lambda x,y: tag_label(x,y), IntegerType())

def indexToTime(index):
    a = index // 3
    b = index % 3
    t = datetime(2019, 6, 1, a, 0) + timedelta(minutes=20*b+10)
    return t.strftime('%H:%M')
def computeIndexByTime(t):
    '''
    t: '2019-07-15 03:21:23.145'
    '''
    hour, minute = map(int, t.split()[1][:5].split(':'))
    index = 0
    if minute < 20:
        index = hour * 3
    elif minute < 40:
        index = hour * 3 + 1
    else:
        index = hour * 3 + 2
    return index
computeIndexByTime_udf = udf(lambda x: computeIndexByTime(x), IntegerType())

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000     # 单位：米
def find_nearest_site(lng,lat,new_list):
    try:
        res = []
        for site_guid,parking_location_lng,parking_location_lat in new_list:
            dis = haversine(lng,lat,float(parking_location_lng), float(parking_location_lat))
            if dis < 100:
                res.append([site_guid, dis])
        return sorted(res, key=lambda x: x[1])[0][0]
    except:
        return
find_nearest_site_udf = udf(lambda x1,x2,x3: find_nearest_site(x1,x2,x3), StringType())

def get_count(new_list):
    res = []
    for _ in range(24*3):
        res.append(0)
    for i, j in new_list:
        res[int(i)] += int(j)
    if args.needDays > 1:
        return list(map(lambda x: round(x / args.needDays), res))
    else:
        return res
get_count_udf = udf(lambda x: get_count(x), ArrayType(IntegerType()))


