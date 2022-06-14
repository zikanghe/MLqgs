# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:33:00 2022

@author: Zikang He
"""


    #History
import matplotlib.pyplot as plt
import numpy as np


history = np.load('L2SM2.npz')
fig, ax = plt.subplots()
ax.plot(history['loss'], label='loss', color='gray')
ax.plot(history['val_loss'], label='val_loss', color='black')
ax.set_xlabel('epochs')
ax.set_ylabel('L2 loss')
ax.set_yscale('log')
ax.legend()
plt.savefig("loss2.tiff", dpi=300)

# In[]
 # Plot the results

    #Get min/max to format the axis
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Insert the common folder in the path (to be able to load python module)
sys.path.insert(0,os.path.join(os.path.pardir,'common'))

data = np.load('train.npz')
xx = data['x'][0:]
yy = data['y'][0:]
    # set train/val limits
ival = -4999
itrain = 35000
xx_train, yy_train = xx[:itrain], yy[:itrain]
xx_val, yy_val = xx[ival:], yy[ival:]
    #Training is taken at the begininng of the time series, Vaidation at the end
icomp = 15

print('Size of the training set:',xx_train.shape)
print('Size of the validation set:',xx_val.shape)
    # Scatter plot of the output to be learnt (pointwise)
fig, ax = plt.subplots(figsize=(10,5))
sns.regplot(x=xx_train[:,icomp],y=yy_train[:,icomp], ax=ax, label='train')
sns.regplot(x=xx_val[:,icomp],y=yy_val[:,icomp], ax=ax, label='validation')
ax.set_xlabel('$x_{'+str(icomp)+'}$')
ax.set_ylabel('$\epsilon^m$')
ax.legend();