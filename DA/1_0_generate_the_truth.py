# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:34:54 2022

@author: Zikang He
"""


# this code used to generate the truth of the paper
import numpy as np
from dapper.admin import HiddenMarkovModel

import time
from multiprocessing import freeze_support, get_start_method


from dapper.mods.Qgs.qgs.params.params import QgParams
from dapper.mods.Qgs.qgs.functions.tendencies import create_tendencies
from dapper.mods.Qgs.qgs.integrators.integrator import RungeKuttaIntegrator


if __name__ == "__main__":

    if get_start_method() == "spawn":
        freeze_support()
        
    a_nx = 2 
    a_ny = 4
    o_nx = 2 
    o_ny = 4 
    Na = a_ny*(2*a_nx+1) 
    No = o_ny*o_nx


    Nx = 2*Na+2*No
    #start Time
    T1 =time.perf_counter()
    ################
    # General parameters
    # Setting some model parameters
    # Model parameters instantiation with default specs
    model_parameters = QgParams()
    # Mode truncation at the wavenumber 2 in both x and y spatial coordinate
    model_parameters.set_atmospheric_channel_fourier_modes(a_nx, a_ny)
    # Mode truncation at the wavenumber 2 in the x and at the
    # wavenumber 4 in the y spatial coordinates for the ocean
    model_parameters.set_oceanic_basin_fourier_modes(o_nx, o_ny)

# Setting MAOOAM parameters according to the publication linked above
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
    
    # Saving the model state n steps
    def step(x0, t0, dt):
       y = x0
       integrator.integrate(t0, t0+dt, 0.01, ic=y, write_steps=0)
       t0, y0 = integrator.get_trajectories()
       #t,y = integrate_runge_kutta(f, t0, t0+dt, 0.1, ic=y, forward=True,write_steps=0, b=None, c=None, a=None)
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
    transient_time = 1.e8
    # integration time on the attractor
    integration_time = 5.e5    
    to_time = transient_time + integration_time
    dt = 1.e5
    from dapper import Chronology
    t = Chronology(dt=dt, dtObs=dt, T=to_time,  BurnIn=transient_time)

    np.random.seed(21217) 
    ic = np.random.rand(model_parameters.ndim)*0.01

    from dapper.tools.randvars import GaussRV
    x0=ic
    X0 = GaussRV(mu=x0, C=0.0)
   # X0 = modelling.RV(M=Nx,C=0.0, mu=x0)
   #4:Obs
    jj = np.arange(Nx)  # obs_inds

    from dapper.tools.math import partial_direct_Obs
    Obs = partial_direct_Obs(Nx, jj)
    #modelling.GaussRV(C=0.333 * np.eye(Nx))
    Obs['noise'] = 0.0033

    
    HMM = HiddenMarkovModel(Dyn, Obs, t, X0)
    from dapper import *
    from dapper.tools.convenience import simulate
    xx, yy = simulate(HMM)
    T2 =time.perf_counter()
    print(str(T2-T1)+' seconds')   
    np.save('trueall.npy',xx)
    np.save('obsall_half.npy',yy)
