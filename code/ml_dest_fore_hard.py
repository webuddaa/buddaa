#coding: utf-8

import pandas as pd
import numpy as np
import mzgeohash as geo
from compiler.ast import flatten

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import json


# base数据
## 用户历史订单的dataframe
user_lock_geohash_posi_df_base = pd.read_pickle(open('./data/nj/user_lock_geohash_posi_base.pd', 'rb'))
user_lock_geohash_posi_df_base.set_index('user_guid', inplace = True)

## 该区域的停车点的dataframe
lock_geohash_posi_base_df = pd.read_pickle(open('./data/nj/lock_geohash_posi_base.pd', 'rb'))
lock_geohash_posi_base_df.set_index('lock_geohash_posi', inplace = True)

## 用户的开锁习惯
user_lock_custom_base_df = pd.read_pickle(open('./data/nj/user_lock_custom_base.pd', 'rb'))
user_lock_custom_base_df.set_index('user_guid', inplace = True)

def string2dict(s):
    my_dict = dict(json.loads(s))
    return my_dict

def column_convert(arrLike):
    s = arrLike['lock_custom_string']
    my_dict = string2dict(s)
    return my_dict


user_lock_custom_base_df['lock_custom'] = user_lock_custom_base_df.apply(column_convert, axis=1)
user_lock_custom_base_df.drop(['lock_custom_string'], axis=1, inplace=True)


## 用户的平均骑行距离
#user_ride_info_base_df = pd.read_pickle('./data/user_ride_info_base.pd', 'rb')


# 训练集数据
train_record_df = pd.read_csv(open('./data/nj/record_train.csv'), names = ['user_guid', 'open_time', 'open_lat', 'open_lng', 'update_lat', 'update_lng'])


# 测试集数据
test_record_df = pd.read_csv(open('./data/nj/record_test.csv'), names = ['user_guid', 'open_time', 'open_lat', 'open_lng', 'update_lat', 'update_lng'])


# 时间划分 3小时
hour_s_list = range(2, 25, 3)

def hour_discrete(hour):
    i = len(hour_s_list) - 1
    if hour == 1:
        return i

    for ele in reversed(hour_s_list):
        if hour >= ele:
            return i
        i = i - 1
    return i


# 配置参数
goal_grid_size = 25


# 生成目标区域网格
def generate_geohash_posi_grid(center_geohash_posi, n):
    l = 2 * n + 1
    gs = [[center_geohash_posi for i in range(l)] for i in range(l)]
    gs_map = {}

    gs[n][n] = center_geohash_posi
    gs_map[center_geohash_posi] = [n, n]
    for i in range(n):
        gs[n - i - 1][n - i - 1] = geo.neighbors(gs[n - i][n - i])['nw']
        gs_map[geo.neighbors(gs[n - i][n - i])['nw']] = [n - i - 1, n - i - 1]

        gs[n - i - 1][n + i + 1] = geo.neighbors(gs[n - i][n + i])['ne']
        gs_map[geo.neighbors(gs[n - i][n + i])['ne']] = [n - i - 1, n + i + 1]

        gs[n + i + 1][n + i + 1] = geo.neighbors(gs[n + i][n + i])['se']
        gs_map[geo.neighbors(gs[n + i][n + i])['se']] = [n + i + 1, n + i + 1]

        gs[n + i + 1][n - i - 1] = geo.neighbors(gs[n + i][n - i])['sw']
        gs_map[geo.neighbors(gs[n + i][n - i])['sw']] = [n + i + 1, n - i - 1]

        for j in range(i + 1):
            gs[n - i - 1][n - j] = geo.adjacent(gs[n - i][n - j], 'n')
            gs_map[geo.adjacent(gs[n - i][n - j], 'n')] = [n - i - 1, n - j]

            gs[n - i - 1][n + j] = geo.adjacent(gs[n - i][n + j], 'n')
            gs_map[geo.adjacent(gs[n - i][n + j], 'n')] = [n - i - 1, n + j]

            gs[n + i + 1][n - j] = geo.adjacent(gs[n + i][n - j], 's')
            gs_map[geo.adjacent(gs[n + i][n - j], 's')] = [n + i + 1, n - j]

            gs[n + i + 1][n + j] = geo.adjacent(gs[n + i][n + j], 's')
            gs_map[geo.adjacent(gs[n + i][n + j], 's')] = [n + i + 1, n + j]

            gs[n - j][n - i - 1] = geo.adjacent(gs[n - j][n - i], 'w')
            gs_map[geo.adjacent(gs[n - j][n - i], 'w')] = [n - j, n - i - 1]

            gs[n + j][n - i - 1] = geo.adjacent(gs[n + j][n - i], 'w')
            gs_map[geo.adjacent(gs[n + j][n - i], 'w')] = [n + j, n - i - 1]

            gs[n - j][n + i + 1] = geo.adjacent(gs[n - j][n + i], 'e')
            gs_map[geo.adjacent(gs[n - j][n + i], 'e')] = [n - j, n + i + 1]

            gs[n + j][n + i + 1] = geo.adjacent(gs[n + j][n + i], 'e')
            gs_map[geo.adjacent(gs[n + j][n + i], 'e')] = [n + j, n + i + 1]

    return gs, gs_map


# 七位的geohash编码
def geohash_encode_7(user_lng, user_lat):
    lng_f = float(user_lng)
    lat_f = float(user_lat)
    geohash_str = geo.encode((lng_f, lat_f))
    return geohash_str[:7]

# 提取特征 - 关锁地点热度
def feature_lock_geohash_posi(goal_geohash_posi_grid):
    row_num = len(goal_geohash_posi_grid)
    res = np.zeros((row_num, row_num), dtype = np.int32)

    for i in range(row_num):
        for j in range(row_num):
            if goal_geohash_posi_grid[i][j] in lock_geohash_posi_base_df.index:
                res[i][j] = lock_geohash_posi_base_df.ix[goal_geohash_posi_grid[i][j]]['count']

    return res

# 提取特征 - 用户的关锁热度
def feature_user_lock_geohash_posi(user_guid, goal_geohash_posi_grid):
    row_num = len(goal_geohash_posi_grid)
    res = np.zeros((row_num, row_num), dtype = np.int32)


    if user_guid in user_lock_geohash_posi_df_base.index:
        user_lock_geohash_posi_dict = dict(user_lock_geohash_posi_df_base.ix[user_guid]['lock_geohash_posi_map'])
    else:
        return res

    for i in range(row_num):
        for j in range(row_num):
            if user_lock_geohash_posi_dict.has_key(goal_geohash_posi_grid[i][j]):
                res[i][j] = user_lock_geohash_posi_dict[goal_geohash_posi_grid[i][j]]

    return res


# 提取特征 - 用户的习惯信息
def feature_user_open_lock_custom(user_guid, open_geohash_posi, goal_geohash_posi_grid, goal_geohash_posi_grid_map):
    row_num = len(goal_geohash_posi_grid)
    res = np.zeros((row_num, row_num), dtype=np.int32)

    if user_guid in user_lock_custom_base_df.index:
        user_open_lock_custom_dict = dict(user_lock_custom_base_df.ix[user_guid]['lock_custom'])
    else:
        return res

    potential_area, _ = generate_geohash_posi_grid(open_geohash_posi, 1)
    potential_area_list = flatten(potential_area)

    for i in range(row_num):
        for j in range(row_num):
            for ele in potential_area_list:
                if user_open_lock_custom_dict.has_key(ele):
                    lock_posi_counter = user_open_lock_custom_dict[ele]
                    for potential_area in lock_posi_counter:
                        if goal_geohash_posi_grid_map.has_key(potential_area):
                            m, n = goal_geohash_posi_grid_map[potential_area]
                            res[m][n] = lock_posi_counter[potential_area]

    return res




# 提取特征 - 用户的骑行信息
def feature_user_ride_info(user_guid):
    user_ride_distance_mean = user_ride_info_base_df.ix[user_guid]['mean']
    user_ride_distance_var = user_ride_info_base_df.ix[user_guid]['var']

    return np.array([user_ride_distance_mean, user_ride_distance_var])

# 提取特征 - 开锁时间
def feature_open_time(open_time):
    res = []
    date = datetime.strptime(open_time, '%Y-%m-%d %H:%M:%S')

    hour_period_index = hour_discrete(date.hour)
    hour_period_array = np.zeros(len(hour_s_list) + 1, dtype = np.int32)
    hour_period_array[hour_period_index] = 1

    weekday = date.weekday() + 1
    if weekday > 5:
        hour_period_array[-1] = 1
    else:
        hour_period_array[-1] = 0

    return hour_period_array

# 定位目标区域的posi
def lock_geohash_posi_location(real_lock_geohash_posi, goal_geohash_posi_grid):
    row_num = len(goal_geohash_posi_grid)

    for i in range(row_num):
        for j in range(row_num):
            if goal_geohash_posi_grid[i][j] == real_lock_geohash_posi:
                return [i, j]
    return []

def generate_label_array(point, num):
    res = np.zeros(num * num, dtype = np.int32)
    index = point[0] * num + point[1]
    res[index] = 1
    return res


def feature_label_generate(record_df):

    feature_list = []
    label_list = []

    total_num = 0
    outline_num = 0

    for _, record in record_df.iterrows():

        total_num = total_num + 1

        user_guid = record['user_guid']

        open_lng = record['open_lng']
        open_lat = record['open_lat']
        open_geohash_posi = geohash_encode_7(open_lng, open_lat)

        open_time = record['open_time']

        real_geohash_posi = geohash_encode_7(record['update_lng'], record['update_lat'])

        goal_geohash_posi_grid, goal_geohash_posi_grid_map = generate_geohash_posi_grid(open_geohash_posi, goal_grid_size)

        # 特征提取
        lock_geohash_posi_feature = feature_lock_geohash_posi(goal_geohash_posi_grid)
        user_lock_geohash_posi_feature = feature_user_lock_geohash_posi(user_guid, goal_geohash_posi_grid)
        user_open_lock_custom_feature = feature_user_open_lock_custom(user_guid, open_geohash_posi, goal_geohash_posi_grid, goal_geohash_posi_grid_map)
        #user_ride_info_feature = feature_user_ride_info(user_guid)
        open_time_feature = feature_open_time(open_time)

        feature = np.concatenate((lock_geohash_posi_feature.flatten(), user_lock_geohash_posi_feature.flatten(), user_open_lock_custom_feature.flatten(), open_time_feature.flatten()))

        label_point = lock_geohash_posi_location(real_geohash_posi, goal_geohash_posi_grid)

        if len(label_point) == 0:
            outline_num = outline_num + 1
            continue

        label = generate_label_array(label_point, 2 * goal_grid_size + 1)

        feature_list.append(feature)
        label_list.append(label)


    print('total_num: {}, outline_num: {}'.format(total_num, outline_num))

    return [feature_list, label_list]


def train_main(train_record_df):

    feature_list, label_list = feature_label_generate(train_record_df)

    X_train, X_test, y_train, y_test = train_test_split(feature_list, label_list, test_size=0.2, random_state=0)

    model = LinearRegression()

    ## 训练
    model.fit(X_train, y_train)

    ## validation
    print("validation result: ")
    print(model.score(X_test, y_test))

    return model


def test_main(test_record_df, model):

    feature_list, label_list = feature_label_generate(test_record_df)

    print("test result: ")
    print(model.score(feature_list, label_list))


if __name__ == '__main__':

    model = train_main(train_record_df)

    test_main(test_record_df, model)


