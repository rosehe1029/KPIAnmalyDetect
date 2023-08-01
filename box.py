import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from datetime import datetime, timedelta



def box_plot_outliers(data_ser, box_scale):
    """
    利用箱线图去除异常值
    :param data_ser: 接收 pandas.Series 数据格式
    :param box_scale: 箱线图尺度，取3
    :return:
    """
    iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
    # 下阈值
    val_low = data_ser.quantile(0.25) - iqr#*0.5
    # 上阈值
    val_up = data_ser.quantile(0.75) + iqr#*0.5
    # 异常值
    outlier = data_ser[(data_ser < val_low) | (data_ser > val_up)]
    # 正常值
    normal_value = data_ser[(data_ser > val_low) & (data_ser < val_up)]
    return outlier, normal_value, (val_low, val_up)


'''数据处理'''
def read_data():
    data=pd.read_csv("data.csv", parse_dates=True, index_col='_time')
    data=data.sort_index(ascending=True) 
    data_under_=data[data.values<2000]
    data_outer_=data[data.values>=2000]
    # 过滤异常值
    outlier, normal_value, value = box_plot_outliers(data_under_['value'], 18)
    outlier = pd.DataFrame(outlier)
    outlier=pd.concat([outlier,data_outer_])
    normal_value = pd.DataFrame(normal_value)
    normal_value['label']=0
    outlier['label']=1
    data_new=pd.concat([outlier,normal_value])
    data_new=data_new.sort_index(ascending=True) 
    data_new.to_csv("data_box.csv")
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
    fig_title = "箱型图异常数据检测"
    Displacement_ax = Outliers_fig(fig_title, 111, 'value', '_time', 'value')
    plt.show()

    
