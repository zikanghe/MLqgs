# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:02:19 2022

@author: Zikang He
"""

# this code used to calculate sigma hf of the sigma hf

import numpy as np
from multiprocessing import freeze_support, get_start_method
from dapper.mods.Qgs.utils import mean_fileter as filt


if __name__ == "__main__":

    if get_start_method() == "spawn":
        freeze_support()

    xx = np.load('true2.npy')
    xx_l = np.copy(xx)
    Ns = 3
    for i in range(56):
        xx_l[:,i] = filt(xx[:,i],Ns)

    xx = xx[10000:60001,:]
    xx_l = xx_l[10000:60001,:]
    dx = xx - xx_l
    sigma_hf = np.std(dx, axis=0)


    np.save('sigma_hf_h.npy',sigma_hf)

