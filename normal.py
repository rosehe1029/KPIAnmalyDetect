import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from datetime import datetime, timedelta



def normal_plot_outliers(data, scale):
    # 计算判断异常点和极端异常点的临界值
    #outlier_ll = data.values.mean() - 2* data.values.std()
    #outlier_ul = data.values.mean() + 2* data.values.std()

    extreme_outlier_ll = data.values.mean() - scale* data.values.std()
    extreme_outlier_ul = data.values.mean() + scale* data.values.std()

    # 寻找异常点
    #data[(data.values > outlier_ul) | (data.values < outlier_ll)]
    # 寻找极端异常点
    outlier=data[(data.values > extreme_outlier_ul) | (data.values < extreme_outlier_ll)]
    normal_value=data[(data.values<=extreme_outlier_ul) & (data.values>=extreme_outlier_ll)]
    
    return outlier, normal_value, (extreme_outlier_ll,extreme_outlier_ul)


'''数据处理'''
def read_data():
    data=pd.read_csv("data.csv", parse_dates=True, index_col='_time')
    data=data.sort_index(ascending=True) 
    data_under_=data[data.values<2000]
    data_outer_=data[data.values>=2000]
    # 过滤异常值
    outlier, normal_value, value = normal_plot_outliers(data_under_['value'], 3)
    outlier = pd.DataFrame(outlier)
    outlier=pd.concat([outlier,data_outer_])
    normal_value = pd.DataFrame(normal_value)
    normal_value['label']=0
    outlier['label']=1
    data_new=pd.concat([outlier,normal_value])
    data_new=data_new.sort_index(ascending=True) 
    data_new.to_csv("data_normal.csv")
    return normal_value, outlier


'''异常值分布图'''
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

    # 绘图
    normal_value, outlier = read_data()
    fig_title = "正态分布图法异常数据检测"
    Displacement_ax = Outliers_fig(fig_title, 111, 'value', '_time', 'value')
    plt.show()

    
