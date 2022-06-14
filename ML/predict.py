# -*- coding: utf-8 -*-
"""
Created on Sun May 29 15:51:59 2022

@author: Zikang He
"""

# ## Modules import
import numpy as np
import time
from multiprocessing import freeze_support, get_start_method
#sys.path.extend([os.path.abspath('../Qgs/qgs')])
# Importing the model's modules

# Start on a random initial condition

from dapper.mods.Qgs.qgs.params.params import QgParams
from dapper.mods.Qgs.qgs.functions.tendencies import create_tendencies
from dapper.mods.Qgs.qgs.integrators.integrator import RungeKuttaIntegrator

# Load the python modules.
import os, sys
# Insert the common folder in the path (to be able to load python module)
sys.path.insert(0,os.path.join(os.path.pardir,'common'))



T1 =time.perf_counter()
# Initializing the random number generator (for reproducibility). -- Disable if needed.
N = 1
data = np.load('true2_low2.npy')
data =data[0:50001]
N0 = data.shape[0]
save_state = np.random.get_state()
np.random.seed(1)
idx = np.random.choice(N0, N, replace=True)
np.random.set_state(save_state)
E = data[idx]

if __name__ == "__main__":

    if get_start_method() == "spawn":
        freeze_support()

    print_parameters = True


    def print_progress(p):
        sys.stdout.write('Progress {:.2%} \r'.format(p))
        sys.stdout.flush()


    class Bcolors:
        """to color the instructions in the console"""
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'


    print("\n" + Bcolors.HEADER + Bcolors.BOLD + "Model qgs v0.2.5 (Atmosphere + ocean (MAOOAM) configuration)" + Bcolors.ENDC)
    print(Bcolors.HEADER + "============================================================" + Bcolors.ENDC + "\n")
    print(Bcolors.OKBLUE + "Initialization ..." + Bcolors.ENDC)
    # ## Systems definition

    # General parameters

    # Time parameters
    dt = 0.01
    # Saving the model state n steps
    write_steps = 100
    # transient time to attractor
    transient_time = 3.e6
    # integration time on the attractor
    integration_time = 5.e5
    # file where to write the output
    filename = "predict.dat"
    T = time.process_time()

    # Setting some model parameters
    # Model parameters instantiation with default specs
    model_parameters = QgParams()
    # Mode truncation at the wavenumber 2 in both x and y spatial coordinate
    model_parameters.set_atmospheric_channel_fourier_modes(2, 2)
    # Mode truncation at the wavenumber 2 in the x and at the
    # wavenumber 4 in the y spatial coordinates for the ocean
    model_parameters.set_oceanic_basin_fourier_modes(2, 4)

    # Setting MAOOAM parameters according to the publication linked above
    model_parameters.set_params({'kd': 0.0290, 'kdp': 0.0290, 'n': 1.5, 'r': 1.e-7,
                                 'h': 136.5, 'd': 1.1e-7})
    model_parameters.atemperature_params.set_params({'eps': 0.7, 'T0': 289.3, 'hlambda': 15.06, })
    model_parameters.gotemperature_params.set_params({'gamma': 5.6e8, 'T0': 301.46})

    model_parameters.atemperature_params.set_insolation(103.3333, 0)
    model_parameters.gotemperature_params.set_insolation(310., 0)

    if print_parameters:
        print("")
        # Printing the model's parameters
        model_parameters.print_params()

    # Creating the tendencies functions
    f, Df = create_tendencies(model_parameters)

    # ## Time integration
    # Defining an integrator
    integrator = RungeKuttaIntegrator()
    integrator.set_func(f)

    # Start on a random initial condition
    ic = E
    #ic = ic.T
    # Integrate over a transient time to obtain an initial condition on the attractors
    print(Bcolors.OKBLUE + "Starting a transient time integration..." + Bcolors.ENDC)
    y = ic
    ym = np.mean(y,axis=0)
    # Now integrate to obtain a trajectory on the attractor
    total_time = 0.
    traj = np.insert(ym, 0, total_time)
    traj = traj[np.newaxis, ...]
    t_up = 10.0 / integration_time * 100

    from test2_utils import buildmodel2 as buildmodel

    model = buildmodel()
    model.summary()
    model.load_weights('weight_QGS.h5')



    def step(x0, t0, dt):
       y = np.copy(x0)
       integrator.integrate(t0, t0+dt, 0.01, ic=y, write_steps=0)
       ta, ya = integrator.get_trajectories()
       output = ya
       return output

    def MLstep(x0, t0, dt):
       y = np.copy(x0)
       ml_step = model.predict
       output2 = dt*ml_step(y[..., np.newaxis]).squeeze()
       return output2


    total_time = 0.
    integration_time = 5.e5
    print(Bcolors.OKBLUE + "Starting the time evolution ..." + Bcolors.ENDC)
    while total_time < integration_time:
        y1 = step(y, total_time, 10.0)
        y2 = MLstep(y, total_time, 10.0)
        y=y1+y2
        ym = np.mean(y2,axis=0)
        total_time += 10.0
        ty = np.insert(ym, 0, total_time)
        traj = np.concatenate((traj, ty[np.newaxis, ...]))
        if total_time/integration_time*100 % 0.1 < t_up:
            print_progress(total_time/integration_time)

    print(Bcolors.OKGREEN + "Evolution finished, writing to file " + filename + Bcolors.ENDC)

    np.savetxt(filename, traj)

    print(Bcolors.OKGREEN + "Time clock :" + Bcolors.ENDC)
    print(str(time.process_time()-T)+' seconds')
