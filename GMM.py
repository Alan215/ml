import kmeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import *
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

#super parameter
clusters = 5
stop_times = 100

df = pd.read_csv("data/kmeans.csv", header=None)
data = df.values
cov = np.cov(data.T)

mul = kmeans.kmeans(clusters, data, 100)

num = data.shape[0]
dim = data.shape[1]
z = np.zeros((num, clusters))

cov1 = cov
cov2 = cov


times = 0
while(times<stop_times):
    p = np.zeros((400,2))
    for i in range(400):
        p[i,0] = np.exp(-1/2*(data[i]-mul1).dot(inv(cov1)).dot((data[i]-mul1).T))*det(cov1)**(-1/2)
        p[i,1] = np.exp(-1/2*(data[i]-mul2).dot(inv(cov2)).dot((data[i]-mul2).T))*det(cov2)**(-1/2)

        z[i,0] = p[i,0]/(p[i,0]+p[i,1])
        z[i,1] = p[i,1]/(p[i,0]+p[i,1])
    tmp1 = 0
    tmp2 = 0
    tmp3 = 0
    tmp4 = 0
    for i in range(400):
        tmp1 += z[i,0]*data[i]
        tmp2 += z[i,1]*data[i]
        tmp3 += np.outer(z[i,0]*(data[i]-mul1),data[i]-mul1)
        tmp4 += np.outer(z[i,1]*(data[i]-mul2),data[i]-mul2)
    mul1 = tmp1/sum(z[:,0])
    mul2 = tmp2/sum(z[:,1])
    cov1 = tmp3/sum(z[:,0])
    cov2 = tmp4/sum(z[:,1])
    times += 1
    print times, mul1,mul2


z1 = np.diag(np.exp(-1/2*(data-mul1).dot(inv(cov1)).dot((data-mul1).T))/((2*np.pi)*det(cov1)**(1/2)))
z2 = np.diag(np.exp(-1/2*(data-mul2).dot(inv(cov2)).dot((data-mul2).T))/((2*np.pi)*det(cov2)**(1/2)))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(data[:,0], data[:,1], z1, color='blue')
ax.plot_trisurf(data[:,0], data[:,1], z2, color='red')
ax.set_xlabel('X')
# ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
# ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
# ax.set_zlim(-100, 100)

plt.show()
