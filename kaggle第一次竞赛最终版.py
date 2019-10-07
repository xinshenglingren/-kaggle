# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:39:54 2019

@author: 潘保恒
"""
import numpy as np
import csv

data = np.loadtxt('HTRU_2_train.csv',delimiter=',')
testData = np.loadtxt('HTRU_2_test.csv',delimiter=',')
np.random.shuffle(data)        #数据随机打乱
traningLab = data[:,-1]      #训练集的标签集
traningData = data[:,0:-1]      #训练集的数据集
#数据预处理
def preData(data):
    for i in range(len(data[0])):
        sum = 0
        for d in data:
            sum += d[i]
        Mean  = sum / len(data)
        sum = 0
        for d in data:
            sum += (d[i] - Mean)**2
            sum = sum/ len(data)
        Standard  = sum**(0.5)
        for d in data:
            d = (d - Mean)/Standard 
    return data
traningData= preData(traningData)
# sigmoid函数
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))
#梯度下降求最优参数
def grad_descent(data, target):
    data = np.insert(data, 0, 1, axis=1)
    dataMat = np.mat(data)      #将读取的数据转换为矩阵
    targetMat = np.mat(target).transpose()       #将读取的数据转换为矩阵
    m,n = np.shape(dataMat)
    weights = np.ones((n, 1))
    alpha = 0.001      #设置梯度的阀值，该值越大梯度下降幅度越大
    maxCycle = 500      #学习次数
    weights = np.ones((n,1)) #设置初始的参数，并都赋默认值为1。注意这里权重以矩阵形式表示三个参数。
    for i in range(maxCycle):
        h = sigmoid(dataMat*weights) # sigmoid函数
        weights = weights - alpha*dataMat.transpose()*(h - targetMat) #迭代更新权重
    return weights
weights = grad_descent(traningData, traningLab)
#判断输出数据生成csv文档
def judge(data, weights, alpha):      
    data = np.insert(data, 0, 1, axis=0)
    dataMat = np.mat(data)      #将读取的数据转换为矩阵
    m,n = np.shape(dataMat)
    h = sigmoid(dataMat*weights)
    if h>alpha:
        return 1
    else:
        return 0
with open("F:\\人工智能方向基础ppt\\spyder_code\\nrew.csv","w",encoding='utf8',newline='') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow(["id","y"])
    for i in range(len(testData)):
        writer.writerow([i+1, judge(testData[i], weights, 0.5)])
        
        
        
        
        
        
        
#import numpy as np
#import csv
#
#data = np.loadtxt('HTRU_2_train.csv',delimiter=',')
#testData = np.loadtxt('HTRU_2_test.csv',delimiter=',')
#np.random.shuffle(data)#数据随机打乱
#traningData = data[:,0:-1]      #训练集的数据集
#traningLab = data[:,-1]      #训练集的标签集
#
##数据预处理
#def preData(data):
#    for i in range(len(data[0])):
#        sum = 0
#        for d in data:
#            sum += d[i]
#        Mean  = sum / len(data)    #均值
#        sum = 0
#        for d in data:
#            sum += (d[i] - Mean )**2
#            sum = sum/ len(data)
#        Standard = sum**(0.5)          #标准差
#        for d in data:
#            d = (d - Mean )/Standard
#    return data
#traningData= preData(traningData)
## sigmoid函数
#def sigmoid(z):
#    return (1 / (1 + np.exp(-z)))
##梯度下降求最优参数
#def grad_descent(data, target):
#    data = np.insert(data, 0, 1, axis=1)
#    dataMat = np.mat(data)      #将读取的数据转换为矩阵
#    targetMat = np.mat(target).transpose()      #将读取的数据转换为矩阵
#    m,n = np.shape(dataMat)
#    weights = np.ones((n, 1))
#    alpha = 0.001      #设置梯度的阀值，该值越大梯度下降幅度越大
#    maxCycle = 500      #学习次数
#    weights = np.ones((n,1)) #设置初始的参数，并都赋默认值为1。注意这里权重以矩阵形式表示三个参数。
#    for i in range(maxCycle):
#        h = sigmoid(dataMat*weights)   #sigmoid 函数
#        weights = weights - alpha*dataMat.transpose()*(h - targetMat) #迭代更新权重
#    return weights
#weights = grad_descent(traningData, traningLab)
#
##判断输出数据生成csv文档
#def judge(data, weights, alpha):     
#    data = np.insert(data, 0, 1, axis=0)
#    dataMat = np.mat(data)     #将读取的数据转换为矩阵
#    m,n = np.shape(dataMat)
#    h = sigmoid(dataMat*weights)
#    if h>alpha:
#        return 1
#    else:
#        return 0
#with open("F:\\人工智能方向基础ppt\\spyder_code\\nrew.csv","w",encoding='utf8',newline='') as csvfile:
#    writer=csv.writer(csvfile)
#    writer.writerow(["id","y"])
#    for i in range(len(testData)):
#        writer.writerow([i+1, judge(testData[i], weights, 0.5)])

#def plotBestFit(weights):  #画出最终分类的图
#    dataMat,labelMat=loadDataSet()
#    dataArr = array(dataMat)
#    n = shape(dataArr)[0]
#    xcord1 = []; ycord1 = []
#    xcord2 = []; ycord2 = []
#    for i in range(n):
#        if int(labelMat[i])== 1:
#            xcord1.append(dataArr[i,1])
#            ycord1.append(dataArr[i,2])
#        else:
#            xcord2.append(dataArr[i,1])
#            ycord2.append(dataArr[i,2])
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
#    ax.scatter(xcord2, ycord2, s=30, c='green')
#    x = arange(-3.0, 3.0, 0.1)
#    y = (-weights[0]-weights[1]*x)/weights[2]
#    ax.plot(x, y)
#    plt.xlabel('X1')
#    plt.ylabel('X2')
#    plt.show()


