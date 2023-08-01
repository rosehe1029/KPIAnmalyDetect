import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import classification_report
# 设置显示中文
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

data=pd.read_csv("data_k_means.csv", parse_dates=True, index_col='_time')
data=data.sort_index(ascending=True) 
train,test=data.iloc[:int(len(data)*0.7)],data.iloc[int(len(data)*0.7):]
print(train.shape,test.shape)

train_X,train_Y=train.iloc[:,:-1],train.iloc[:,-1]
test_X,test_Y=test.iloc[:,:-1],test.iloc[:,-1]
print(train_X.shape)
#分类器实例化
svm=SVC(kernel='rbf',C=10,gamma=0.1,probability=True, class_weight={1: 1000}).fit(train_X,train_Y)

y_pred=svm.predict(test_X)
target_names=['class0','class1']
print(classification_report(test_Y,y_pred,target_names = target_names))

from sklearn.metrics import roc_curve
y_test_proba=svm.predict_proba(test_X)
print(y_test_proba.shape)
fpr, tpr, thresholds = roc_curve(test_Y, y_test_proba[:,1])
plt.plot(fpr, tpr)
plt.show()
from sklearn.metrics import auc
auc_score = auc(fpr, tpr)
print('auc',auc_score)


