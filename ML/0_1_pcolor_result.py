# -*- coding: utf-8 -*-
"""
Created on Mon May 30 09:13:44 2022

@author: Zikang He
"""
import numpy as np
import matplotlib.pyplot as plt
f = open('esemble_mean.dat')
fr=np.loadtxt(f)
fr = fr[10001:50001,1:37]
f = open('predict.dat')
fr2=np.loadtxt(f)
fr2 = fr2[10001:50001,1:37]

[xgrd,ygrd] = np.meshgrid(range(40000),range(36))
plt.figure(figsize=(12,9))
plt.subplot(311)
plt.title('Truth',fontsize=20)
levels = np.linspace(-0.2, 0.2, 9)
plt.contourf(xgrd,ygrd,fr.T,levels=levels.round(5),cmap=plt.cm.rainbow)
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
plt.title('Data Assimilation Filter',fontsize=20)
plt.contourf(xgrd,ygrd,fr2.T,levels=levels.round(5),cmap=plt.cm.rainbow)
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
plt.title('Free Run',fontsize=20)
plt.contourf(xgrd,ygrd,fr.T-fr2.T,levels=levels.round(5),cmap=plt.cm.rainbow)
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
#plt.savefig("Compare truth_DA.tiff", dpi=300)
plt.show()