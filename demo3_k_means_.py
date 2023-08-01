#  -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib as mpl
import matplotlib.pyplot as plt

def k_means__():
    data=pd.read_csv("data.csv")#, index_col='_time')
    #data=data.sort_index(ascending=True) 

    # 数据准备
    df=data['value'].values.reshape(-1, 1)
    X = np.array(df)  # 准备 sklearn.cluster.KMeans 模型数据
    print("Shape of cluster data:", X.shape)

    # KMeans 聚类分析(sklearn.cluster.KMeans)
    nCluster = 2
    kmCluster = KMeans(n_clusters=nCluster).fit(X)  # 建立模型并进行聚类，设定 K=2
    print("Cluster centers:\n", kmCluster.cluster_centers_)  # 返回每个聚类中心的坐标
    print(type(kmCluster.cluster_centers_))
    #pd.DataFrame(kmCluster.cluster_centers_).to_csv(r"聚类中心.csv")
    #print("Cluster results:\n", kmCluster.labels_)  # 返回样本集的分类结果
    '''
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import silhouette_samples
 

    y=kmCluster.predict(X)
    print('y',y)
    # 查看轮廓系数均值
    #print(silhouette_score(X,kmCluster.labels_))
    '''
 
    # 整理聚类结果
    listName = data['_time'].tolist()  # 将 dfData 的首列 '地区' 转换为 listName
    dictCluster = dict(zip(listName,kmCluster.labels_))  # 将 listName 与聚类结果关联，组成字典
    listCluster = [[] for k in range(nCluster)]
    for v in range(0, len(dictCluster)):
        k = list(dictCluster.values())[v]  # 第v个城市的分类是 k
        listCluster[k].append(list(dictCluster.keys())[v])  # 将第v个城市添加到 第k类
    print("\n聚类分析结果(分为{}类):".format(nCluster))  # 返回样本集的分类结果
    for k in range(nCluster):
        print("第 {} 类：{}".format(k, listCluster[k]))  # 显示第 k 类的结果
    
    #存储
    with open(r'k_means_result.csv', 'w') as f:
        [f.write('{0},{1}\n'.format(key, value)) for key, value in dictCluster.items()]
    
    k_means_result=pd.read_csv('k_means_result.csv',names=['_time','label'])
    k_means_result=k_means_result.iloc[:,-1]
    data=pd.concat([data,k_means_result],axis=1)
    data.to_csv('data_k_means.csv',index=None)
    ''' 
    pd.DataFrame(listCluster).T.to_csv(r"聚类结果\1.csv",index=False)
    '''
    normal_value=data[data['label']==0].iloc[:,:-1]
    outlier=data[data['label']==1].iloc[:,:-1]
    return normal_value,outlier


def Outliers_fig(fig_title, fig_num, col_name, xlabel, ylabel):
    # 画布大小
    fig = plt.figure(figsize=(14, 7), edgecolor='blue')
    # 标题
    plt.suptitle(fig_title, fontsize=20, x=0.5, y=0.970)
    # 调整子图在画布中的位置
    plt.subplots_adjust(bottom=0.145, top=0.9000, left=0.075, right=0.990)
    ax = fig.add_subplot(fig_num)
    ax.plot_date(normal_value.index, normal_value[col_name], 'bo', linewidth=1.5, label='正常值')
    ax.plot_date(outlier.index, outlier[col_name], 'ro', linewidth=1.5, label='异常值')
    ax.set_xlabel(xlabel, fontsize=18, labelpad=7)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=7)
    ax.tick_params(labelsize=18, direction='out')
    #ax.grid(linestyle='--')
    ax.legend(fontsize=15)
    # 中文显示
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    return ax

if __name__ == '__main__':
    normal_value,outlier=k_means__()
    fig_title = "k-means异常值检测"
    Displacement_ax = Outliers_fig(fig_title, 111, 'value', '_time', 'value')
    plt.show()