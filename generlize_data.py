import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# #kmeans
# data1 = np.random.multivariate_normal([1,-1], [[1,0],[0,1]], 200)
# data2 = np.random.multivariate_normal([5.5,-4.5], [[1,0],[0,1]], 200)
# data3 = np.random.multivariate_normal([1,4], [[1,0],[0,1]], 200)
# data4 = np.random.multivariate_normal([6,4.5], [[1,0],[0,1]], 200)
# data5 = np.random.multivariate_normal([9,0.0], [[1,0],[0,1]], 200)
#
# data = np.concatenate([data1, data2, data3, data4, data5], 0)
# df = pd.DataFrame(data)
# df.to_csv("data/kmean_data.csv",header=False,index=False)
#
#
# plt.scatter(data1[:,0], data1[:,1], color='red')
# plt.scatter(data2[:,0], data2[:,1], color='blue')
# plt.scatter(data3[:,0], data3[:,1], color='khaki')
# plt.scatter(data4[:,0], data4[:,1], color='green')
# plt.scatter(data5[:,0], data5[:,1], color='magenta')
# plt.show()

# #GMM
data1 = np.random.multivariate_normal([1,1], [[1,0],[0,1]], 200)
data2 = np.random.multivariate_normal([-1,-1], [[1,0],[0,1]], 200)

data = np.concatenate([data1, data2], 0)
print len(data)
df = pd.DataFrame(data)
df.to_csv("data/GMM_data.csv",header=False,index=False)

plt.scatter(data1[:,0], data1[:,1], c='red')
plt.scatter(data2[:,0], data2[:,1], c='blue')
plt.show()