# 决策树
import pandas as pda
fname = "A:/Python_17/lesson.csv"
dataf = pda.read_csv(fname)
x = dataf.iloc[:, 1:5].as_matrix()
y = dataf.iloc[:, 5].as_matrix()

for i  in range(0, len(x)):
    for j in range(0, len(x[i])):
        thisdata = x[i][j]
        if(thisdata == "是" or thisdata =="多" or thisdata == "高"):
            x[i][j] = 1
        else:
            x[i][j] = -1

for i in range(0, len(y)):
    thisdata = y[i]
    if(thisdata == "高"):
        y[i] = 1
    else:
        y[i] = -1

xf = pda.DataFrame(x)
yf = pda.DataFrame(y)
x2 = xf.as_matrix().astype(int)
y2 = yf.as_matrix().astype(int)

from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion = "entropy")
dtc.fit(x2, y2)

# 可视化决策树
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
with open("A:/Python_17/dtc.dot", "w") as file:
    export_graphviz(dtc, feature_names = ["practice", "course_num", "on_sale", "have_code"], out_file = file)

# 直接预测销量高低
import numpy as npy
x3 = npy.array([[1, -1, -1, 1], [1, 1, 1, 1], [-1, 1, -1, 1]])
rst = dtc.predict(x3)
print(rst)












