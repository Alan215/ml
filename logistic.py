# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def sigmoid(x):
    return 1.0/(1+np.e**(-x))

def logistic(data, theta):
    output = sigmoid(np.dot(data, theta[1:].T)+theta[0])
    return output

def sgd(data, target, initial_theta, step_size = 1, times = 100):
    theta = initial_theta
    not_conveged = True
    itea = 0
    while(True):
        i = np.random.randint(len(data))
        itea += 1
        print itea
        step_size = step_size/(1+1.0*itea/times)
        theta[0] = theta[0] + step_size*(target[i] - logistic(data[i], theta))
        theta[1:] = theta[1:] + step_size*(target[i] - logistic(data[i], theta))*data[i]
        if itea >= times:
            break
    return theta

def count_accuracy(output, Y):
    output[output>0.5] = 1
    output[output<0.5] = 0
    error = np.sum(np.abs(output - Y))
    return 1.0*(len(output)-error)/len(output)


# iris = datasets.load_iris()
# data = iris.data[:100]
# target = iris.target[:100]

file1 = open("data/q1x.dat")
lst1 = file1.read().split('\n')[:-1]
file1.close()
lst1 = [[float(a) for a in l.split()] for l in lst1]
file2 = open("data/q1y.dat")
lst2 = file2.read().split('\n')[:-1]
file2.close()
lst2 = [float(a) for a in lst2]
data = np.array(lst1)
target = np.array(lst2)


step_size = 0.76
times = 1000
initial_theta = [1]
for i in range(data.shape[1]):
    initial_theta.append(1.0 * np.random.randint(1, 100)/100)
initial_theta = np.array(initial_theta)
x = []
y = []

np.random.seed(0)
indices = np.random.permutation(len(data))
train_num = np.abs(1.0 * len(data)/10 * 9)
X_train = data[indices[:train_num]]
Y_train = target[indices[:train_num]]
X_test = data[indices[train_num:]]
Y_test = target[indices[train_num:]]

for step_size in np.arange(0.01, 1, 0.01):
    theta = sgd(X_train, Y_train, initial_theta, step_size, times)
    output = logistic(X_test, theta)
    accuracy = count_accuracy(output, Y_test)
    x.append(step_size)
    y.append(accuracy)
plt.plot(x,y)
plt.show()

# for times in np.arange(10, 2000, 10):
#     theta = sgd(X_train, Y_train, initial_theta, step_size, times)
#     output = logistic_output(X_test, theta)
#     accuracy = count_accuracy(output, Y_test)
#     x.append(times)
#     y.append(accuracy)
# plt.plot(x,y)
# plt.show()
