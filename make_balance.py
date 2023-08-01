import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

data=pd.read_csv("data_normal.csv", parse_dates=True, index_col='_time')
data=data.sort_index(ascending=True) 

X,y=data.iloc[:,:-1],data.iloc[:,-1]
# 综合采样（先过采样再欠采样）
## # combine 表示组合抽样，所以 SMOTE 与 Tomek 这两个英文单词写在了一起
from imblearn.combine import SMOTETomek
kos = SMOTETomek(random_state=0)  # 综合采样
X_kos, y_kos = kos.fit_resample(X, y)
print('综合采样后，训练集 y_kos 中的分类情况：{}'.format(Counter(y_kos)))

data_new=pd.concat([X_kos,y_kos],axis=1)
data_new.to_csv("data_balance.csv",index=None)