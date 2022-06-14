# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 10:04:43 2022

@author: Zikang He
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 30 09:13:44 2022

@author: Zikang He
"""
import numpy as np
import matplotlib.pyplot as plt


yy = np.load('true2.npy')
yy1 = np.ones(40000*36).reshape((40000,36))
yy1[:,0:10]=yy[10001:50001,0:10]
yy1[:,10:20]=yy[10001:50001,20:30]
yy1[:,20:35]=yy[10001:50001,40:55]
yy1[:,35]=yy[10001:50001,55]


fh=np.load('hybrid.npy')
fh = np.mean(fh,1)
fh = fh[0:40000]

ft=np.load('trunc.npy')
ft = np.mean(ft,1)
ft = ft[0:40000]

[xgrd,ygrd] = np.meshgrid(range(40000),range(36))
plt.figure(figsize=(12,9))
plt.subplot(311)
plt.title('Truth',fontsize=20)
levels = np.linspace(-0.2, 0.2, 9)
plt.contourf(xgrd,ygrd,yy1.T,levels=levels.round(5),cmap=plt.cm.rainbow)
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
plt.contourf(xgrd,ygrd,ft.T,levels=levels.round(5),cmap=plt.cm.rainbow)
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
plt.title('Hybrid',fontsize=20)
plt.contourf(xgrd,ygrd,fh.T,levels=levels.round(5),cmap=plt.cm.rainbow)
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