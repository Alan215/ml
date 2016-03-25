# -*- coding:utf8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
import random
import pandas as pd

# k = 5
# neighbor = 100
# sigma = 100
#
#
# #读取生成的数据
# def read_data():
#     data = []
#     file1 = open("data/kmeans_data.txt")
#     lst = file1.readlines()
#     file1.close()
#     for i in range(len(lst)):
#         tmp = lst[i].split(' ')
#         tmp[0] = float(tmp[0])
#         tmp[1] = float(tmp[1][0:-1])
#         data.append(tmp)
#     return data
#
#
# #相异程度<0.001的两个列表当成相同
# def is_equal(a, b):
#     if len(a) != len(b):
#         return False
#     for i in range(len(a)):
#         diff = a[i] - b[i]
#         if diff > 0.001 or diff < -0.001:
#             return False
#     return True
#
#
# def is_similar(d, cluster_center):
#     for c in cluster_center:
#         if is_equal(c, d):
#             return True
#     return False
#
# #列表元素清0
# def zero_list(d):
#     for i in range(len(d)):
#         d[i] = 0
#
# #欧式距离
# def distance(x, y):
#     n = len(y)
#     dis = 0
#     for i in range(n):
#         dis += (x[i] - y[i]) ** 2
#     return dis
#
# #返回x最近的聚类中心
# def nearest(x, cc):
#     dis = -1
#     near = -1
#     for i in range(len(cc)):
#         d = distance(x, cc[i])
#         if dis < 0 or dis > d:
#             dis = d
#             near = i
#     return near
#
#
# def divide(d, y):
#     if y == 0:
#         zero_list(d)
#         return
#     for i in range(len(d)):
#         d[i] /= y
#
#
# def add(a, b):
#     n = len(a)
#     for i in range(n):
#         a[i] += b[i]
#
#
# def k_means(data):
#     m = len(data)
#     n = len(data[0])
#     cluster = [-1 for x in range(m)]     # 所有样本尚未聚类
#     cluster_center = [[] for x in range(k)]  # 聚类中心
#     cc = [[] for x in range(k)]          # 下一轮的聚类中心
#     c_number = [0 for x in range(k)]     # 每个簇中样本的数目
#
#     # 随机选择簇中心
#     i = 0
#     while i < k:
#         j = random.randint(0, m-1)
#         if is_similar(data[j], cluster_center):
#             continue
#         cluster_center[i] = data[j][:]
#         cc[i] = [0 for x in range(n)]
#         i += 1
#     for times in range(40):
#         for i in range(m):
#             c = nearest(data[i], cluster_center)
#             cluster[i] = c      # 第i个样本归于第c簇
#             c_number[c] += 1
#             add(cc[c], data[i])
#         for i in range(k):
#             divide(cc[i], c_number[i])
#             c_number[i] = 0
#             cluster_center[i] = cc[i][:]
#             zero_list(cc[i])
# #        print times, cluster_center
#     return cluster, cluster_center
#
#
# def show_data(data, cluster):
#     m = len(data)
#     cluster1 = [[] for x in range(k)]
#     cluster2 = [[] for x in range(k)]
#     display_mode = "rgbcmyk"
#     for i in range(m):
#         c = cluster[i]
#         cluster1[c].append(data[i][0])
#         cluster2[c].append(data[i][1])
#     for i in range(k):
#         plt.plot(cluster1[i], cluster2[i], display_mode[i]+'o', markersize=4)
#     plt.grid(True)
#     plt.show()


# data = read_data()
#
# cluster, center = k_means(data)
# show_data(data, cluster)
# #print "cluster", cluster
# for i in range(k):
#     print "center:", center[i]
#     print "amount of points", cluster.count(i)

#super parameter
def kmeans(K, data, stop_times):
    clusters = []

    num = data.shape[0]
    dim = data.shape[1]
    r = np.zeros((num,K))

    #some function
    def square_dis(x, y):
        return sum((x-y)**2)

    def max_distance(sample, arr):
        max_dis = square_dis(sample, arr[0])
        max_index = 0
        for i in range(1, len(arr)):
            if square_dis(sample, arr[i])<max_dis:
                max_dis = square_dis(sample, arr[i])
                max_index = i
        return max_index

    #initial mul
    mul = np.zeros((K, dim))
    for i in range(K):
        rand = np.random.permutation(num)
        mul[i] = np.mean(data[rand[:K]])

    times = 0
    while(times<stop_times):
        for i in range(num):
            max_index = max_distance(data[i], mul)
            r[i, max_index] = 1
            a = np.arange(K)
            r[i, a!=max_index] = 0
        mul = data.T.dot(r).T/np.stack([np.sum(r,0),np.sum(r,0)]).T
        times += 1
    for i in range(K):
        clusters.append(data[r[:,i]==1])

    return clusters
