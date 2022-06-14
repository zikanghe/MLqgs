# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:48:22 2022

@author: Zikang He
"""

"""
Python script to compute a training set for a machine learning method. If the training is computed from observation, a DA algorithm is applied.
run 'python compute_trainingset.py -h' to see the usage
"""


# Load the python modules.
import os, sys
import numpy as np

# Insert the common folder in the path (to be able to load python module)
sys.path.insert(0,os.path.join(os.path.pardir,'common'))



xx_train = np.load('true_test.npy')
#xx_train = np.load('trunc_test.npy')

	# Compute traininset

	#Input of the dataset
xx_in = xx_train[:-1]

	# Estimation of the true value after dtObs MTU
xx_out = xx_train[1:]

	# Truncated value after dtObs MTU
yy = np.load('freerun_test.npy')
xx_trunc = yy[:-1]

	# Estimation of the model error
dtObs = 10.0
delta = (xx_out - xx_trunc) / dtObs

# In[pcolor Truth Da free run ]
import numpy as np
import matplotlib.pyplot as plt




[xgrd,ygrd] = np.meshgrid(range(40000),range(36))
plt.figure(figsize=(12,9))
plt.subplot(311)
plt.title('Truth',fontsize=20)
levels = np.linspace(-0.2, 0.2, 9)
plt.contourf(xgrd,ygrd,xx_train.T,levels=levels.round(5),cmap=plt.cm.rainbow)
plt.axhline(y=9,linestyle='--',color='black')
plt.axhline(y=19,linestyle='--',color='black')
plt.axhline(y=27,linestyle='--',color='black')
plt.yticks(np.arange(0,36,10),fontsize=18)
plt.ylabel('Variable Index',fontsize=18)
plt.ylim([0,35])
plt.xticks([])
plt.xlim([0,40000])
#plt.xlabel('Model Steps',fontsize=18)
plt.colorbar()

plt.subplot(312)
plt.title('Free Run',fontsize=20)
plt.contourf(xgrd,ygrd,yy.T,levels=levels.round(5),cmap=plt.cm.rainbow)
plt.axhline(y=9,linestyle='--',color='black')
plt.axhline(y=19,linestyle='--',color='black')
plt.axhline(y=27,linestyle='--',color='black')
plt.yticks(np.arange(0,36,10),fontsize=18)
plt.ylabel('Variable Index',fontsize=18)
plt.ylim([0,35])
plt.xticks([])
plt.xlim([0,40000])
#plt.xlabel('Model Steps',fontsize=18)
plt.colorbar()

plt.subplot(313)
plt.title('Truth-Free Run',fontsize=20)
plt.contourf(xgrd,ygrd,xx_train.T-yy.T,levels=levels.round(5),cmap=plt.cm.rainbow)
plt.axhline(y=9,linestyle='--',color='black')
plt.axhline(y=19,linestyle='--',color='black')
plt.axhline(y=27,linestyle='--',color='black')
plt.yticks(np.arange(0,36,10),fontsize=18)
plt.ylabel('Variable Index',fontsize=18)
plt.ylim([0,35])
plt.xticks(np.arange(0,50001,10000),fontsize=18)
plt.xlim([0,40000])
plt.xlabel('Model Steps',fontsize=18)
plt.colorbar()
plt.show()