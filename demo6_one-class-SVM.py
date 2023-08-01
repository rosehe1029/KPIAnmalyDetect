import time
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt # 导入Matplotlib

# 数据准备
raw_data = pd.read_csv("data.csv", index_col='_time') # 读取数据
train_set,test_set=raw_data.iloc[:int(len(raw_data)*0.7)],raw_data.iloc[int(len(raw_data)*0.7):]

# 异常数据检测
model_onecalsssvm = OneClassSVM(nu=0.1, kernel="rbf") # 创建异常检测算法模型对象
model_onecalsssvm.fit(train_set) # 训练模型
pre_test_outliers = model_onecalsssvm.predict(test_set) # 异常检测
# 异常结果统计
toal_test_data = np.hstack((test_set, pre_test_outliers.reshape(test_set.shape[0], 1))) # 将测试集和检测结果合并
normal_test_data = toal_test_data[toal_test_data[:, -1] == 1] # 获得异常检测结果中集
outlier_test_data = toal_test_data[toal_test_data[:, -1] == -1] # 获得异常检测结果异常数据
n_test_outliers = outlier_test_data.shape[1] # 获得异常的结果数量
total_count_test = toal_test_data.shape[0] # 获得测试集样本量
print ('outliers: {0}/{1}'.format(n_test_outliers, total_count_test)) # 输出异常的结果数量
print ('{:*^60}'.format(' all result data (limit 5) ')) # 打印标题
print (toal_test_data[:5]) # 打印输出前5条合并后的数据集
