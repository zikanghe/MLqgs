# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:08:40 2022

@author: Zikang He
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:22:43 2022

@author: Zikang He
"""



# Load the python modules.
import os, sys
# Insert the common folder in the path (to be able to load python module)
sys.path.insert(0,'../ML/common')
import argparse
import numpy as np
import random as rn
from multiprocessing import freeze_support, get_start_method

from sklearn.model_selection import ParameterGrid
import tensorflow.keras.backend as K
import tensorflow as tf
# Parse the commande line

if __name__ == "__main__":

	if get_start_method() == "spawn":
		freeze_support()


# Load the config files




#File of the
	burn = 0




	# Load the dataset

	data = np.load('train_QGS.npz')
	xx = data['x'][burn:]
	yy = data['y'][burn:]

	itrain = 10000
	ival = -1000
	# Training is taken at the begininng of the time series, Vaidation at the end
	xx_train, yy_train = xx[:itrain], yy[:itrain]
	xx_val, yy_val = xx[ival:], yy[ival:]

	# Define the NN model
	K.clear_session()

	# Inialize random generation numbers
	np.random.seed(2020)
	rn.seed(2020)
	os.environ['PYTHONHASHSEED']=str(2020)
	tf.random.set_seed(2020)

	from test_utils import buildmodel as buildmodel
	model = buildmodel([[43, 5, 'tanh'], [28, 1, 'tanh']],reg=0.072,batchlayer=1)
	model.summary()
	model.compile(loss='mse', optimizer='RMSprop')
	verbose = 1


	#Train the NN
	history = model.fit(xx_train, yy_train,
		epochs=100,batch_size=33,validation_data=(xx_val, yy_val)
		)
	# SAVE HISTORY
	np.savez('history_QGSM1.npz',
		loss=history.history['loss'],
		val_loss=history.history['val_loss'])
	# SAVE WEIGHTS
	model.save_weights('weight_QGSM1.h5')

# In[]
    #History
	import matplotlib.pyplot as plt
	import numpy as np


	history = np.load('history_QGSM1.npz')
	fig, ax = plt.subplots()
	ax.plot(history['loss'], label='loss', color='gray')
	ax.plot(history['val_loss'], label='val_loss', color='black')
	ax.set_xlabel('epochs')
	ax.set_ylabel('L2 loss')
	ax.set_yscale('log')
	ax.legend()