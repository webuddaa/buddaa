for hour in 19 20 21 22 23
do
    PYSPARK_PYTHON=/usr/bin/python3.7 spark-submit --master yarn --deploy-mode client --queue DataBusinessDept_Turing_Scheduling --name "xiangfeng"  --num-executors 20 --driver-memory 4g --executor-memory 4g --executor-cores 4 /home/master/hadoop_project/scripts/suanfa/route_v3_beforeToday.py --day 19 --hour $hour
done