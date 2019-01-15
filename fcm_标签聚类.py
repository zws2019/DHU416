# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:04:43 2018

@author: zws
"""
import numpy as np
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans
import skfuzzy
import pandas as pd
import matplotlib as mpl
import seaborn as sns  # 美化图形的一个绘图包
sns.set_style("whitegrid")  # 设置图形主图

# 导入数据
w_size = 20

def loadstock():
    names = ['日期','开盘','最高','最低','收盘','成交量','成交额']
#    names = ['date','open','high','low','close','volume','price_change']
    data = pd.read_csv('000001.csv', names=names,header=1,encoding = "gbk")
#    data = pd.read_table('SH#900901.txt', names=names,header=1,encoding = "gbk")
#    predictor_names = ['close']
    predictor_names = ['收盘']
    training_features = np.asarray(data[predictor_names], dtype = "float32")#将列表转化为矩阵
    plt.figure()
    plt.plot(training_features, c = 'red') #标准化后的股票数据
    plt.show()
    X = []
    Y = []
    for i in range(len(training_features) - w_size):#  x ；前window_size，y后window_size
        d = training_features[i] - min(training_features)
        X.append(training_features[i:i + w_size]-d)
#        Y.append(training_features[i + 17])
    X ,Y = np.array(X) , np.array(Y)
    X ,Y= X.reshape(np.shape(X)[0],-1),Y.reshape(np.shape(Y)[0])
    return X, Y

tr,_ = loadstock()
train = np.array(tr)
train = train.T
fpc_all = []
#循环调参c
#for j in range(2,40):
#    center, u1, u0, d, jm, p, fpc = cmeans(train, m=1.5, c=j, error=0.005, maxiter=1000000)
#    print(j)
#    fpc_all.append(fpc)
#plt.figure()
#plt.plot(range(2,40),fpc_all)
#plt.grid(True, linestyle = "-.", color = "b", linewidth = "1")
#plt.xlabel('Clustering number') 
#plt.ylabel('The fuzzy partition coefficient (FPC)')  
#plt.show()
c_num = 3  # 设置聚类个数3/5/30
center, u1, u0, d, jm, p, fpc = cmeans(train, m=1.5, c=c_num, error=0.005, maxiter=1000000)
for i in u1:
    label_1 = np.argmax(u1, axis=0)
    
  # 相同趋势类别，可视化
for j in range(c_num):
    t1 = np.where(label_1==j)[0]
    print("类别:%d,数量：%d"%(j,len(t1)))
    plt.figure()
    for i in range(len(t1)):
        plt.plot(range(w_size),tr[t1][i])
    plt.ylim(-1000,2000)
    plt.show()
# 保存标签数据
label_1 = pd.DataFrame(data = label_1)
label_1.to_csv('label_3.csv')


# 可视化聚类中心
plt.figure()  # 3类中心可视化
label = ["Fall","Smooth","Rise"]    # 请根据实际情况调整顺序
marker = ['p','+','o','D','.']
for i in range(c_num):
    plt.plot(range(w_size),center[i],label=label[i],marker=marker[i])
legend = plt.legend(loc = 2,ncol=2)
frame = legend.get_frame() 
frame.set_alpha(0) 
frame.set_facecolor('none') # 设置图例legend背景透明 
plt.ylim([200,1700])
plt.show()

#plt.figure()  # 5类中心可视化
#label = ["Big rise","Small rise","Big fall","Small fall","Smooth"]
#marker = ['p','+','o','D','.']
#for i in range(c_num):
#    plt.plot(range(w_size),center[i],label=label[i],marker=marker[i])
#legend = plt.legend(loc = 2,ncol=2)
#frame = legend.get_frame() 
#frame.set_alpha(0) 
#frame.set_facecolor('none') # 设置图例legend背景透明 
#plt.ylim([200,1700])
#plt.show()
#
#plt.figure()  # 30类中心可视化
#for i in range(c_num):
#    plt.plot(range(w_size),center[i])
#plt.ylim(-500,2000)
#plt.show()

#print('||||||||||||||||||||||||||||  新数据  |||||||||||||||||||||||||||||')
#newdata = pd.read_csv('2017_10.csv')
#u2, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans_predict(newdata.T, center, 1.5, error=0.005, maxiter=1000)
#for i in u2:
#    label_2 = np.argmax(u2, axis=0)
#print(label_2)
#
#label_1 = label_1.tolist()
#u_words = set(label_1)
#print(u_words)
#for wo in u_words:
#    print(label_1.count(wo))



