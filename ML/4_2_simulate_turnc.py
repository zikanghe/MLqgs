# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 21:50:35 2022

@author: Zikang He
"""



# this code used to generate the truth of the paper
#In this part, the data which is about to reach the stable state is selected as the initial field
import numpy as np
from dapper.admin import HiddenMarkovModel
#sys.path.extend([os.path.abspath('../Qgs/qgs')])
# Importing the model's modules

# Start on a random initial condition

from dapper.mods.Qgs.qgs.params.params import QgParams
from dapper.mods.Qgs.qgs.functions.tendencies import create_tendencies
from dapper.mods.Qgs.qgs.integrators.integrator import RungeKuttaIntegrator
from multiprocessing import freeze_support, get_start_method


# In[1] parameter setting
if __name__ == "__main__":
	if get_start_method() == "spawn":
		freeze_support()

	a_nx = 2
	a_ny = 2
	o_nx = 2
	o_ny = 4
	Na = a_ny*(2*a_nx+1)
	No = o_ny*o_nx
	Nx = 2*Na+2*No


#start Time
    ################
	model_parameters = QgParams()
	model_parameters.set_atmospheric_channel_fourier_modes(a_nx, a_ny)
	model_parameters.set_oceanic_basin_fourier_modes(o_nx, o_ny)

	model_parameters.set_params({'kd': 0.0290, 'kdp': 0.0290, 'n': 1.5, 'r': 1.e-7,
	                                 'h': 136.5, 'd': 1.1e-7})
	model_parameters.atemperature_params.set_params({'eps': 0.7, 'T0': 289.3, 'hlambda': 15.06, })
	model_parameters.gotemperature_params.set_params({'gamma': 5.6e8, 'T0': 301.46})
	model_parameters.atemperature_params.set_insolation(103.3333, 0)
	model_parameters.gotemperature_params.set_insolation(310., 0)
	model_parameters.print_params()

	f, Df = create_tendencies(model_parameters)
	integrator = RungeKuttaIntegrator()
	integrator.set_func(f)


# In[2] model setting



    
    # Saving the model state n steps
	def step(x0, t0, dt):
		y = x0
		integrator.integrate(t0, t0+dt, 0.01, ic=y, write_steps=0)
		t0, y0 = integrator.get_trajectories()
		return y0

#parameter setting

    #1:Dyn
   
	Dyn = {
		'M': Nx,
		'model': step,
		'linear': Df,
		'noise': 0,
		}



    #2:Time(need to modify)
    # transient time to attractor
	from dapper import Chronology
	#integration_time = 5.e5
	integration_time = 500
	to_time = integration_time
	dt = 10.0
	dto = 10.0
	t = Chronology(dt, dtObs=dto, T=to_time,  BurnIn=0)



	from dapper.tools.randvars import RV
	from utils import sampling
	X0 = RV(36, func=sampling)
   #4:Obs
	from dapper.tools.math import partial_direct_Obs
	jj = np.arange(Nx)  # obs_inds
	Obs = partial_direct_Obs(Nx, jj)
    #modelling.GaussRV(C=0.333 * np.eye(Nx))
	Obs['noise'] = 0.0033

	HMM = HiddenMarkovModel(Dyn, Obs, t, X0)

# In[integrate]
	from tqdm import tqdm

	Dyn, chrono = HMM.Dyn, HMM.t
	N=20
	xx = np.zeros((HMM.t.K+1, N, Dyn.M))
	X0 = HMM.X0.sample(N)
	#Initial time step
	xx[0] = X0


	for k, kObs, t, dt in tqdm(chrono.ticker):
		xx[k] = Dyn(xx[k - 1], t - dt, dt)
	np.save('trunc.npz',xx)
