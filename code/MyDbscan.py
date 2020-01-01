from sklearn.cluster import DBSCAN
from datetime import datetime
import numpy as np

class MyDbscan:
    '''
    根据时间序列来聚类
    '''
    def __init__(self, data, eps=4):
        self.data = data
        self.eps = eps
    def _similarity(self, x, y):
        i, j = int(x[0]), int(y[0])
        t1 = datetime.strptime(self.data[i], '%Y-%m-%d %H:%M:%S')
        t2 = datetime.strptime(self.data[j], '%Y-%m-%d %H:%M:%S')
        return abs(t1.timestamp() - t2.timestamp()) / 60    # 单位：minute
    def myDbscan(self):
        X = np.arange(len(self.data)).reshape(-1,1)
        res = DBSCAN(eps=self.eps, min_samples=1, metric=self._similarity).fit(X)
        class_num = np.max(res.labels_) + 1
        groups = [[] for _ in range(class_num)]
        for label, content in zip(res.labels_, self.data):
            groups[label].append(content)
        return groups

if __name__ == '__main__':
    timeList = [
    '2019-09-23 12:23:45','2019-09-23 12:27:45','2019-09-29 12:23:45',
    '2019-09-23 12:25:45','2019-09-24 12:23:45','2019-09-24 12:27:45'
    ]

    db = MyDbscan(timeList)
    print(db.myDbscan())




