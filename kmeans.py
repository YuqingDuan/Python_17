# kmeans算法
# 通过程序实现录取学生的聚类
import pandas as pda
import numpy as npy
import matplotlib.pylab as pyl

fname = "A:/Python_17/luqu.csv"
dataf = pda.read_csv(fname)
x = dataf.iloc[:, 1:4].as_matrix()

from sklearn.cluster import Birch
from sklearn.cluster import KMeans

kms = KMeans(n_clusters = 2)
y = kms.fit_predict(x)

# x轴代表学生序号，y轴代表学生类别
s = npy.arange(0, len(y))
pyl.plot(s, y, "o")
pyl.show()

# 通过程序实现商品的聚类
import pandas as pda
import numpy as npy
import matplotlib.pylab as pyl
import pymysql

conn = pymysql.connect(host = "127.0.0.1", user = "root", passwd = "Devilhunter9527", db  ="dangdang")
sql = "select price, comment from taob limit 3000"
dataf = pda.read_sql(sql, conn)
x = dataf.iloc[:, :].as_matrix()

from sklearn.cluster import KMeans
kms = KMeans(n_clusters = 3)
y = kms.fit_predict(x)

for i in range(0, len(y)):
    if(y[i] == 0):
        pyl.plot(dataf.iloc[i: i+1, 0: 1].as_matrix(), dataf.iloc[i: i+1, 1: 2].as_matrix(), "*r")
    if(y[i] == 1):
        pyl.plot(dataf.iloc[i: i+1, 0: 1].as_matrix(), dataf.iloc[i: i+1, 1: 2].as_matrix(), "sy")
    if(y[i] == 2):
        pyl.plot(dataf.iloc[i: i+1, 0: 1].as_matrix(), dataf.iloc[i: i+1, 1: 2].as_matrix(), "*y")
pyl.show()




