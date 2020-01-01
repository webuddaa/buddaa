'''
从kafka中读取数据，解析数据，转化成dataframe
'''
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import json

spark = SparkSession.builder.enableHiveSupport().getOrCreate()

def data_extract(x):
    line = x.split(',', 3)
    action_time = line[1].split('.')[0]
    content = json.loads(line[3])
    try:
        bikeNo = content['condition']['bikeNo']
        option = content['condition']['option']
        alertType = content['condition']['alertType']
        return [action_time, bikeNo, option, alertType]
    except Exception as e:
        return [action_time, None, None, None]   

# 正则匹配
alert_useful_rdd = spark.sparkContext.sequenceFile('/dataspace/kafka/alert-useful/2018-{09-2[5-8], 10-{0*,1*,2[0-8]}}/*')
alert_useful_rdd = alert_useful_rdd.values().map(lambda x: x.decode('utf-8'))
alert_useful_df = alert_useful_rdd.map(data_extract).toDF(['action_time', 'bike_no', 'option', 'alertType'])








