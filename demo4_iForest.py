# 加载模型所需要的的包
import numpy   as np
import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 构造一个数据集，只包含一列数据，有些可能是错的
data=pd.read_csv("data.csv", index_col='_time')
data=data.sort_index(ascending=True)
#构建模型 ,n_estimators=100 ,构建100颗树

model = IsolationForest()#(n_estimators=100)
# 训练模型
model.fit(data[['value']])

# 预测 decision_function 可以得出 异常评分
data['scores']  = model.decision_function(data[['value']])

#  predict() 函数 可以得到模型是否异常的判断，-1为异常，1为正常
data['anomaly'] = model.predict(data[['value']])
print(data)