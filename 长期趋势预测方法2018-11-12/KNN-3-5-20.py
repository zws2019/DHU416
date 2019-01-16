# -*- coding: UTF-8 -*-
import pandas as pd
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns  # 美化图形的一个绘图包
sns.set_style("whitegrid")  # 设置图形主图
#column_names = ['日期','开盘','最高','最低','收盘','成交量','成交额']
#data = pd.read_csv('000001.csv',names = column_names)
#data = data.replace(to_replace='?', value=np.nan)
#data = data.dropna(how='any')
#data_label = pd.read_csv('label_1.csv',encoding = "gbk")
#print(np.mat(data))
def KNN_pre(w_size,trend,c):  # w_size 预测所用窗口大小，trend趋势长度，c 趋势长度
    def loadstock():
        names = ['日期','开盘','最高','最低','收盘','成交量','成交额']
    #    names = ['date','open','high','low','close','volume','price_change']
        data = pd.read_csv('000001.csv', names=names,header=1,encoding = "gbk")
        if c == 3:
            data_label = pd.read_csv('label_3.csv',encoding = "gbk")
        elif c == 5:
            data_label = pd.read_csv('label_5.csv',encoding = "gbk")
        else:
            data_label = pd.read_csv('label_30_20.csv',encoding = "gbk")
#        if trend == 20:
#            data_label = pd.read_csv('label_30_20.csv',encoding = "gbk")
#        elif trend == 40:
#            data_label = pd.read_csv('label_30_40.csv',encoding = "gbk")
#        else:
#            data_label = pd.read_csv('label_30_60.csv',encoding = "gbk")
        data_label = np.array(data_label['0'], dtype = "int32")
    #    data = pd.read_table('SH#900901.txt', names=names,header=1,encoding = "gbk")
    #    predictor_names = ['close']
        predictor_names = ['收盘']
        training_features = np.asarray(data[predictor_names], dtype = "float32")  # 将列表转化为矩阵
#        plt.figure()
#        plt.plot(training_features, c = 'red') # 标准化后的股票数据
#        plt.show()
        X = []
        Y = []
        for i in range(len(training_features) - trend - w_size):#  x ；前window_size，y后window_size
#            d = training_features[i+w_size-1] - min(training_features)
#            X.append(training_features[i:i + w_size]-d)
            X.append(training_features[i:i + w_size])
            Y.append(data_label[w_size+i])
        X ,Y = np.array(X) , np.array(Y)
        X ,Y= X.reshape(np.shape(X)[0],-1),Y.reshape(np.shape(Y)[0])
        return X, Y
    X, Y= loadstock()
    datatrain,datatest, labeltrain, labeltest = train_test_split(X, Y,test_size=0.20, random_state=20)
#    datatrain = X[:int(0.8*len(X))]
#    datatest = X[int(0.8*len(X)):]
#    labeltrain = Y[:int(0.8*len(X))]
#    labeltest = Y[int(0.8*len(X)):]
#    print(len(labeltest))
#    print(len(np.where(labeltest[:-w_size]==labeltest[w_size:])[0]))
#    print(len(np.where(labeltest[:-w_size]==labeltest[w_size:])[0])/len(labeltest))
    # print datatrain,datatest, labeltrain, labeltest
    #标准化数据，保证每个维度的特征数方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    Sdatatrain = ss.fit_transform(datatrain)
    Sdatatest = ss.transform(datatest)
    
    def classify0(inX, dataSet, labels, k):
        dataSetSize = dataSet.shape[0]
        # 距离度量
        diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5
#        # 变化趋势度量
#        inX_new = inX[1:]- inX[:-1]
#        dataSet_new = dataSet[:,1:] - dataSet[:,:-1]
#        diffMat = np.tile(inX_new, (dataSetSize,1)) - dataSet_new
#        sqDiffMat = diffMat**2
#        sqDistances = sqDiffMat.sum(axis=1)
#        distances = sqDistances**0.5

#        diffMat_2 = np.tile(0.8*inX, (dataSetSize,1)) - dataSet
#        sqDiffMat_2 = diffMat_2**2
#        sqDistances_2 = sqDiffMat_2.sum(axis=1)
#        distances_2 = sqDistances_2**0.5
#        
#        diffMat_3 = np.tile(1.2*inX, (dataSetSize,1)) - dataSet
#        sqDiffMat_3 = diffMat_3**2
#        sqDistances_3 = sqDiffMat_3.sum(axis=1)
#        distances_3 = sqDistances_3**0.5
#        if min(distances)<min(distances_2) and min(distances)<min(distances_3):
#            distances = distances
#        elif min(distances_2)<min(distances_3):
#            distances = distances_2
#        else:
#            distances = distances_3

        sortedDistIndicies = distances.argsort()  # 返回从小到大排列的索引值
        classCount={}
        for i in range(k):
            voteIlabel = labels[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  # get返回指定键voteIlabel的值，指定键不存在时返回默认值default:0
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # items返回可遍历的(键, 值) 元组数组。itemgetter选择键0或者值1进行排列。reverse逆序
        return sortedClassCount[0][0]
    
    labeltest = np.array(labeltest)
    labeltrain = np.array(labeltrain)
    labelpredict = []
    for i in Sdatatest:
        labelpredict.append(classify0(i, Sdatatrain, labeltrain,1))
    
#    for i in range(len(labelpredict)):  # 输出分类结果
#        labelpredictresult=(labelpredict[i])
#        labeltestraw = (labeltest[i])
#        print("分类返回结果为%d\t真实返回结果为%d" %(labelpredictresult,labeltestraw))
    
    errorCount = 0.0
    for i in range(len(labeltest)):
        if(labelpredict[i]!= labeltest[i]):
                errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/len(labeltest) * 100))
    print("显示进度：趋势长度%d 类别%d 窗口%d"%(trend,c,w_size))  
    return 1- errorCount/len(labeltest)
if __name__ == '__main__':
#    for j in [3,5,30]:
#        KNN_pre(20,20,j)
    a = []
    knn_all_ws = []
#    for j in [20,40,60]:
    for j in [3,5,30]:
        for i in range(3,150):
#            a.append(KNN_pre(i,j,20))
            a.append(KNN_pre(i,20,j))
        knn_all_ws.append(a)
        a = []
    plt.figure(figsize=(8, 5))
#    label = ["Trend:20 day","Trend:40 day","Trend:60 day"]
    label = ["Class:3","Class:5","Class:30"]
    c = ["r","g","b"]
    for i in [0,1,2]:
        plt.plot(range(3,3+len(knn_all_ws[i])),knn_all_ws[i],label=label[i],c=c[i])
    plt.xlabel("Windows Size")
    plt.ylabel("Accuracy Prediction")
    plt.yticks(np.linspace(0,1,11))
    plt.legend(loc=4)
    plt.show()
