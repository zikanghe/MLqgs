# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:41:17 2022

@author: Zikang He
"""
# In[1]
# Load the python modules.
import os, sys
#Insert the common folder in the path (to be able to load python module)
sys.path.insert(0,os.path.join(os.path.pardir,'common'))
import numpy as np
from dapper import Chronology

from toolbox import load_config, path, get_filenames, load_data, rmse_norm
from l2s_utils import plot_L96_2D, default_param, buildmodel

#To control plot layouts
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper")

from tqdm.notebook import tqdm


# In[2]
#Source of the data (comment/uncomment the lines given your choice)
source = 'local'
#source = 'ftp'

# In[3]

# Directory where to save the figure (if figdir = None, no figure are saved)
figdir = '.'
#figdir = None



# In[4]
# Configuration files
config_path = 'config/paths.yml'
config_ftp = 'config/ftp.yml' #optional if data are generated locally
config_ref = 'config/ref_test.yml'
config_sens = 'config/sens_test.yml'
config_sens_po = 'config/sens_test_po.yml'

# In[5]

# Load the paths (yml file)
paths = load_config(config_path)

# Load the reference
ref = load_config(config_ref)

# Rootdir for the L2S data
rootdir = path(paths['rootdir'])

# Template names
template_ref = ref['template']

#Check if figdir exist and set the figdir bool accordingly
savefig = bool(figdir)

#Directory containing the simulations
refdir = path(os.path.join(rootdir,ref['savedir']))

#Print:
print("Directory containing the simulations for the reference experiment:\n->",refdir)

# In[6]
#Load the ftp configuration file if needed
if source == 'ftp':
    ftp = load_config(config_ftp)
    ftpurl, ftpdir = ftp['url'], ftp['test']
    print('\033[93m'+'Warning! The source is set to ftp. \nIt will override existing local simulation in the simulation directory'+'\033[0m')
else:
    ftp = None
    ftpurl, ftpdir = None, None
    

# In[7]

# Define the parameters
used_parameter = { 'p', 'std_o', 'dtObs', 'std_m' ,'N','T','seed', 'Nfil_train'}

#List of values for the used paramters
dparam = {k:ref.get(k,[default_param[k]])[0] for k in used_parameter}



used_parameters = { 'p', 'std_o', 'dtObs', 'std_m' ,'N','T','seed'}
print('Parameters of the reference experiment:\n',dparam)

# In[8]

# Load the data of the true run
fname_truth = template_ref['truth'].format(**dparam)
print('Load truth simulation:',fname_truth)
data_truth = load_data(refdir,fname_truth,ftpurl=ftpurl,ftpdir=ftpdir)

# Load the data of the hybrid run
fname_hybrid = template_ref['hybrid'].format(**dparam)
print('Load hybrid simulation:', fname_hybrid)
data_hybrid = load_data(refdir,fname_hybrid,ftpurl=ftpurl,ftpdir=ftpdir)

# In[9]
# Format the data (the time step of the hybrid model is different from the time step of the real model)

#hybrid data
xx_hybrid = data_hybrid['xx']

#chronology
chrono_hybrid = Chronology(dkObs=1,T=data_hybrid['T'],dt=float(data_hybrid['dt']))
chrono_truth = Chronology(dkObs=1,T=data_truth['T'],dt=float(data_truth['dt']))

#Compute the time step subsample:
dk = int(chrono_hybrid.dt/chrono_truth.dt)

#Compute the size of the hybrid model stat:
nU = xx_hybrid.shape[-1]

#Subsampe the true model in space and time to fit the hybrid model space
xx_truth = data_truth['xx'][::dk,:,:nU]

#Define the time axis
tt = chrono_hybrid.tt

#Display the size of the model simulations
print('Size of the time axis:',tt.shape)
print('Size of the truth simulation (time step, members, space):',xx_truth.shape)
print('Size of the hybrid simulation (time step, members, space):',xx_hybrid.shape)

# Plot the figure 1

#Ensemble member to plot:
iens=6

#Number of time steps to plot
limT = 600

#Plot
fig, ax = plot_L96_2D(xx_truth[:limT,iens],xx_hybrid[:limT,iens],tt[:limT],['Truth','NN hyb.'])
ax[2].set_xlabel('Time [MTU]')
#Save (if needed)
if figdir:
    fig.savefig(os.path.join(figdir,'philtra-fig1.png'),
            dpi=200, bbox_inches='tight', pad_inches=0)
