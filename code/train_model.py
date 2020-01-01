import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('data.csv')

data.drop(['user_guid', 'start_period', 'new_weather', 'new_high_temperature', 'new_low_temperature', 'out_in_counts'], axis=1, inplace=True)

y = data.pop("fantai_time")
X = data.as_matrix()
y = y.as_matrix()

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 设置参数
clf = xgb.XGBRegressor(learning_rate=0.1, n_estimators=100)
# 训练
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy : %.2f" % metrics.mean_squared_error(y_test, y_pred))
