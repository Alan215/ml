# -*- coding:utf8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
import random

#读取生成的数据
def read_data():
    data = []
    file1 = open("data/s_data.txt")
    lst = file1.readlines()
    file1.close()
    for i in range(len(lst)):
        tmp = lst[i].split(' ')
        tmp[0] = float(tmp[0])
        tmp[1] = float(tmp[1][0:-1])
        data.append(tmp)
    return data


#相异程度<0.001的两个列表当成相同
def is_equal(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        diff = a[i] - b[i]
        if diff > 0.001 or diff < -0.001:
            return False
    return True


def is_similar(d, cluster_center):
    for c in cluster_center:
        if is_equal(c, d):
            return True
    return False

#列表元素清0
def zero_list(d):
    for i in range(len(d)):
        d[i] = 0

#欧式距离
def distance(x, y):
    n = len(y)
    dis = 0
    for i in range(n):
        dis += (x[i] - y[i]) ** 2
    return dis

#返回x最近的聚类中心
def nearest(x, cc):
    dis = -1
    near = -1
    for i in range(len(cc)):
        d = distance(x, cc[i])
        if dis < 0 or dis > d:
            dis = d
            near = i
    return near


def divide(d, y):
    if y == 0:
        zero_list(d)
        return
    for i in range(len(d)):
        d[i] /= y


def add(a, b):
    n = len(a)
    for i in range(n):
        a[i] += b[i]


def k_means(data):
    m = len(data)
    n = len(data[0])
    cluster = [-1 for x in range(m)]     # 所有样本尚未聚类
    cluster_center = [[] for x in range(k)]  # 聚类中心
    cc = [[] for x in range(k)]          # 下一轮的聚类中心
    c_number = [0 for x in range(k)]     # 每个簇中样本的数目

    # 随机选择簇中心
    i = 0
    while i < k:
        j = random.randint(0, m-1)
        if is_similar(data[j], cluster_center):
            continue
        cluster_center[i] = data[j][:]
        cc[i] = [0 for x in range(n)]
        i += 1
    for times in range(40):
        for i in range(m):
            c = nearest(data[i], cluster_center)
            cluster[i] = c      # 第i个样本归于第c簇
            c_number[c] += 1
            add(cc[c], data[i])
        for i in range(k):
            divide(cc[i], c_number[i])
            c_number[i] = 0
            cluster_center[i] = cc[i][:]
            zero_list(cc[i])
        print times, cluster_center
    return cluster


def similar(data, i, j):
    n = len(data[i]) - 1
    s = 0
    for k in range(n):
        s += (data[i][k] - data[j][k]) ** 2
    return math.exp(-s / (2 * sigma**2))


def is_neighbor(x, nearest):
    n = len(nearest)
    b = False
    for i in range(n):
        if x > nearest[i]:
            nearest.insert(i, x)
            nearest.pop()
            b = True
            break
    return b


def laplace_matrix(data):
    m = len(data)
    w = [[] for x in range(m)]
    for i in range(m):
        w[i] = [0 for x in range(m)]
    nearest = [0 for x in range(neighbor)]

    for i in range(m):
        zero_list(nearest)
        for j in range(i+1, m):
            w[i][j] = similar(data, i, j)
            if not is_neighbor(w[i][j], nearest):
                w[i][j] = 0
            w[j][i] = w[i][j]   #对称
        w[i][i] = 0
    for i in range(m):
        s = 0
        for j in range(m):
            s += w[i][j]
        if s == 0:
            print "矩阵第", i, "行全为0"
            continue
        for j in range(m):
            w[i][j] /= s
            w[i][j] = -w[i][j]
        w[i][i] += 1    #单位阵主对角线为1
    return w


def spectral_cluster(data):
    lm = laplace_matrix(data)
    eg_values, eg_vectors = linalg.eig(lm)
    idx = eg_values.argsort()
    eg_vectors = eg_vectors[:, idx]

    m = len(data)
    eg_data = [[] for x in range(m)]
    for i in range(m):
        eg_data[i] = [0 for x in range(k)]
        for j in range(k):
            eg_data[i][j] = eg_vectors[i][j]
    return k_means(eg_data)


def show_data(data, cluster):
    m = len(data)
    n = len(data[0])
    cluster1 = [[] for x in range(k)]
    cluster2 = [[] for x in range(k)]
    display_mode = "rgbcmyk"
    for i in range(m):
        c = cluster[i]
        cluster1[c].append(data[i][0])
        cluster2[c].append(data[i][1])
    for i in range(k):
        plt.plot(cluster1[i], cluster2[i], display_mode[i]+'o', markersize=4)
    plt.grid(True)
    plt.show()


k = 2
accur_lst = []
data = read_data()


neighbor = 100
sigma = 200
# sigma_lst = range(1,1001,10)
# for sigma in sigma_lst:
cluster = spectral_cluster(data)
#    print cluster
#    show_data(data, cluster)
right = 0
result = [cluster[0]]*100
result.extend([1-cluster[0]]*100)
for i in range(len(cluster)):
    if cluster[i]==result[i]:
        right += 1
accur_lst.append(1.0*right/len(cluster))
show_data(data, cluster)
# plt.scatter(sigma_lst, accur_lst)
# plt.show()

#sigma = 200
#neighbor_lst = range(1,100)
#for neighbor in neighbor_lst:
#    cluster = spectral_cluster(data)
##    print cluster
##    show_data(data, cluster)
#    right = 0
#    result = [cluster[0]]*100
#    result.extend([1-cluster[0]]*100)
#    for i in range(len(cluster)):
#        if cluster[i]==result[i]:
#            right += 1
#    accur_lst.append(1.0*right/len(cluster))
#plt.scatter(neighbor_lst, accur_lst)
#plt.show()

