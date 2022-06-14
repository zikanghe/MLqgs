# Plot L96d
import numpy as np


# Default value of paramter (overwrited using the configuration files)




# trunc HMM (no param)


import numpy as np

#TODO: allow list of index instead of maxind

from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Layer, Dense
from tensorflow.keras import regularizers
from tensorflow.keras import Model



def buildmodel2(m=36):
	inputs = Input(shape=36)
	x = BatchNormalization()(inputs)
	x = Dense(100,activation="relu")(x)
	x = Dense(50,activation="relu")(x)
	output = Dense(36,activity_regularizer=regularizers.L2(0.0001))(x)
	return Model(inputs,output)


