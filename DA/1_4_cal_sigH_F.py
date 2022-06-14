# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:28:37 2022

@author: Zikang He
"""



# this code is used to filter ocean variable and re-calculate sigma



import numpy as np

sig = np.load('sigma_hf_h.npy')
sig_o = 0.1*np.copy(sig)

a_nx = 2
a_ny = 4
o_nx = 2
o_ny = 4
Na = a_ny*(2*a_nx+1)
No = o_ny*o_nx
Nx = 2*Na+2*No
yy = np.load('obs_h.npy')
yy1 = np.copy(yy)
from utils import mean_fileter as filt
Ns = 51
for i in range(Nx):
    if i>=2*Na:
         yy[:,i] = filt(yy[:,i],Ns)
yy2 = np.copy(yy)

xx = np.load('true2.npy')
xx = xx[10000:60000,:]
yy = yy[10000:60000,:]
yy1 = yy1[10000:60001,:]
yy2 = yy2[10000:60001,:]
d1 = xx -yy1
d2 = xx -yy2
sig1 =np.var(d1,axis=0)
sig2 =np.var(d2,axis=0)
np.save('sigma_hf_H1.npy', sig1*10)
np.save('sigma_hf_H2.npy', sig2*10)

# In[plot RMSE]
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


#plt.plot(sig1,label = 'obs noise',linestyle='--')
#plt.plot(sig2,label = 'obs noise filt',linestyle='--')
#plt.plot(RMSEb,label = 'DA2')
plt.plot(sig1,label = 'obs noise',linestyle='--')
plt.plot(sig2,label = 'obs noise filt',linestyle='--')
plt.xlim([19,35]);
plt.axvline(x=9,linestyle='--')
plt.axvline(x=19,linestyle='--')
plt.axvline(x=27,linestyle='--')
plt.xticks([27,35],fontsize = 10);
plt.legend(fontsize=10,ncol=3)
plt.title(str('observe Noise').title(),fontsize=10)
plt.ylim([0,0.0016]);
plt.yticks([0,0.0002,0.0004,0.0006,0.0008,0.0010]);
#plt.savefig("RMSE3.tiff", dpi=300)
plt.show()
# In[plot RMSE]
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


#plt.plot(sig1,label = 'obs noise',linestyle='--')
#plt.plot(sig2,label = 'obs noise filt',linestyle='--')
#plt.plot(RMSEb,label = 'DA2')
plt.plot(d,label = 'obs noise - obs noise filt')
#plt.plot(sig2,label = 'obs noise filt',linestyle='--')
plt.xlim([19,35]);
plt.axvline(x=9,linestyle='--')
plt.axvline(x=19,linestyle='--')
plt.axvline(x=27,linestyle='--')
plt.xticks([27,35],fontsize = 10);
plt.legend(fontsize=10,ncol=3)
plt.title(str('observe Noise').title(),fontsize=10)
#plt.ylim([-2.5e-6,0.5e-6]);
#plt.yticks([0,0.0002,0.0004,0.0006,0.0008,0.0010]);
#plt.savefig("RMSE3.tiff", dpi=300)
plt.show()