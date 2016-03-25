import kmeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import *
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

#super parameter
def GMM(K, data, stop_times):
    num = data.shape[0]
    dim = data.shape[1]

    clusters = kmeans.kmeans(K, data, 100)

    mul = np.zeros((K, dim))
    cov = np.zeros((K, dim, dim))
    for i in range(K):
        mul[i] = np.mean(clusters[i],axis=0)
        cov[i] = np.cov(clusters[i].T)

    #latent variable
    z = np.zeros((num, K))

    times = 0
    # print data[999]
    while(times<stop_times):
        #compute
        p = np.zeros((num, K))
        #E step
        for i in range(num):
            for j in range(K):
                p[i, j] = np.exp(-1/2*(data[i]-mul[j]).dot(inv(cov[j])).dot((data[i]-mul[j]).T))*det(cov[j])**(-1/2)
                # print p[i,j]

            for j in range(K):
                z[i,j] = p[i, j]/np.sum(p[i, :])
        #M step
        for j in range(K):
            tmp = np.zeros((1,dim))
            for i in range(num):
                tmp += z[i,j]*data[i]
            mul[j] = tmp/np.sum(z[:,j])
        for j in range(K):
            tmp = np.zeros((dim,dim))
            for i in range(num):
                tmp += z[i,j]*np.outer((data[i]-mul[j]),(data[i]-mul[j]))
            cov[j] = tmp/np.sum(z[:,j])
        times += 1
        # print times
    print mul, cov

    out = []
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(K):
        out.append(np.diag(np.exp(-1/2*(data-mul[i]).dot(inv(cov[i])).dot((data-mul[i]).T))/((2*np.pi)*det(cov[i])**(1/2))))
        ax.plot_trisurf(data[:,0], data[:,1], out[i], color=np.random.rand(50))
    ax.set_xlabel('X')
    # ax.set_xlim(-40, 40)
    ax.set_ylabel('Y')
    # ax.set_ylim(-40, 40)
    ax.set_zlabel('Z')
    # ax.set_zlim(-100, 100)

    plt.show()
