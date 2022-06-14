# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:23:47 2022

@author: Zikang He
"""

import numpy as np
import matplotlib.pyplot as plt

xx1 = np.load('hybridM1.npz')
xx1 = xx1['xx']
xx1 =np.mean(xx1,axis=1)
xx2 = np.load('hybridM2.npz')
xx2 = xx2['xx']
xx2 =np.mean(xx2,axis=1)
[xgrd,ygrd] = np.meshgrid(range(15001),range(36))
plt.figure(figsize=(12,9))
plt.subplot(311)
plt.title('M1',fontsize=20)
levels = np.linspace(-15, 15, 9)
plt.contourf(xgrd,ygrd,xx1.T,levels=levels.round(5),cmap=plt.cm.rainbow)
plt.axhline(y=9,linestyle='--',color='black')
plt.axhline(y=19,linestyle='--',color='black')
plt.axhline(y=27,linestyle='--',color='black')
plt.yticks(np.arange(0,36,10),fontsize=18)
plt.ylabel('Variable Index',fontsize=18)
plt.ylim([0,35])
plt.xticks([])
plt.xlim([0,150])
#plt.xlabel('Model Steps',fontsize=18)
plt.colorbar()

plt.subplot(312)
plt.title('M2',fontsize=20)
plt.contourf(xgrd,ygrd,xx2.T,levels=levels.round(5),cmap=plt.cm.rainbow)
plt.axhline(y=9,linestyle='--',color='black')
plt.axhline(y=19,linestyle='--',color='black')
plt.axhline(y=27,linestyle='--',color='black')
plt.yticks(np.arange(0,36,10),fontsize=18)
plt.ylabel('Variable Index',fontsize=18)
plt.ylim([0,35])
plt.xticks([])
plt.xlim([0,150])
#plt.xlabel('Model Steps',fontsize=18)
plt.colorbar()

plt.subplot(313)
plt.title('M1-M2',fontsize=20)
levels = np.linspace(-4, 4, 9)
plt.contourf(xgrd,ygrd,xx1.T-xx2.T,levels=levels.round(5),cmap=plt.cm.rainbow)
plt.axhline(y=9,linestyle='--',color='black')
plt.axhline(y=19,linestyle='--',color='black')
plt.axhline(y=27,linestyle='--',color='black')
plt.yticks(np.arange(0,36,10),fontsize=18)
plt.ylabel('Variable Index',fontsize=18)
plt.ylim([0,35])
plt.xticks(np.arange(0,160,30),fontsize=18)
plt.xlim([0,120])
plt.xlabel('Model Steps',fontsize=18)
plt.colorbar()
plt.savefig("result.tiff", dpi=300)
plt.show()