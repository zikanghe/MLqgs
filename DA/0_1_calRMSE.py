# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 17:05:05 2022

@author: Zikang He
"""

# In[pcolor Truth Da free run ]
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt



yy = np.load('true2.npy')
yy1 = np.ones(40000*36).reshape((40000,36))
yy1[:,0:10]=yy[10001:50001,0:10]
yy1[:,10:20]=yy[10001:50001,20:30]
yy1[:,20:35]=yy[10001:50001,40:55]
yy1[:,35]=yy[10001:50001,55]

data =np.load('daPF.npz')
xxF = data['mua']


Fr =np.load('freerun.npy')
Fr=Fr[1:400001]

Rmse = np.zeros((36,2))
for i in range(36):
   mse = np.sum((xxF[:,i] - yy1[:,i]) ** 2) / len(yy1[:,i])
   Rmse[i,0] = sqrt(mse)
   mse = np.sum((Fr[:,i] - yy1[:,i]) ** 2) / len(yy1[:,i])
   Rmse[i,1] = sqrt(mse)

plt.plot(Rmse[:,0],label = 'Data Assimilation')
plt.plot(Rmse[:,1],label = 'Free Run')
plt.xlim([0,35]);
plt.axvline(x=9,linestyle='--')
plt.axvline(x=19,linestyle='--')
plt.axvline(x=27,linestyle='--')
plt.xticks([0,9,19,27,35],fontsize = 10);
plt.legend(fontsize=10,ncol=1)
plt.title(str('RMSE').title(),fontsize=10)
plt.ylim([0,0.08]);
plt.savefig("data RMSE.tiff", dpi=300)
plt.show()

