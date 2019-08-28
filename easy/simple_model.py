#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Demonstrate usage of LFPy.Network with network of ball-and-stick type
morphologies with active HH channels inserted in the somas and passive-leak
channels distributed throughout the apical dendrite. The corresponding
morphology and template specifications are in the files BallAndStick.hoc and
BallAndStickTemplate.hoc.

Execution (w. MPI):

    mpirun -np 2 python example_network.py

Copyright (C) 2017 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""
# import modules:
import os
import sys
if sys.version < '3':
    from urllib2 import urlopen
else:    
    from urllib.request import urlopen
import ssl
import zipfile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.signal as ss
import scipy.stats as st
from mpi4py import MPI
import neuron
from LFPy import NetworkCell, Network, Synapse, RecExtElectrode, FourSphereVolumeConductor
# import csv
import pickle
import time
from matplotlib.collections import PolyCollection
from os.path import join
import random as random

# set up MPI variables:
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# avoid same sequence of random numbers from numpy and neuron on each RANK,
# e.g., in order to draw unique cell and synapse locations and random synapse
# activation times
GLOBALSEED = 1234
np.random.seed(GLOBALSEED + RANK)

################################################################################
# Set up shared and population-specific parameters
################################################################################
# from param import *
from param import spike_t, spike_std, distr_t, weighttrain,cellParameters, OUTPUTPATH, populationParameters, networkParameters, electrodeParameters, networkSimulationArguments, num_cells, population_names,population_sizes, connectionProbability, synapseModel,synapseParameters, weightArguments, weightFunction, minweight, delayFunction, delayArguments, mindelay, multapseFunction, multapseArguments, synapsePositionArguments

# Save the cell params for population plot
fi=open("example_network_output/cells.pkl","wb")
pickle.dump(cellParameters,fi)
fi.close()

if __name__ == '__main__':
    start = time.time()

    ############################################################################
    # Main simulation
    ############################################################################
    # create directory for output:
    if not os.path.isdir(OUTPUTPATH):
        if RANK == 0:
            os.mkdir(OUTPUTPATH)
    COMM.Barrier()

    # instantiate Network:
    network = Network(**networkParameters)
    
    # create E and I populations:
    for ii, (name, size) in enumerate(zip(population_names, population_sizes)):
        if RANK == 0:
            print(ii, name, size)
        network.create_population(name=name, POP_SIZE=size,
                                  **populationParameters[ii])

        # initial spike train
        if name =='E':
            for j,cell in enumerate(network.populations[name].cells):
                if j%4==0:
                    idx = cell.get_rand_idx_area_norm(section='dend', nidx=1)
                    for i in idx:#if more than one synapse
                        syn = Synapse(cell=cell, idx=i, syntype='Exp2Syn',
                                      weight=weighttrain,
                                      **dict(synapseParameters[0][0]))
                        syn.set_spike_times(np.array([distr_t[j]]))
  
    # create connectivity matrices and connect populations:
    for i, pre in enumerate(population_names):
        for j, post in enumerate(population_names):
            # boolean connectivity matrix between pre- and post-synaptic neurons
            # in each population (postsynaptic on this RANK)
            connectivity = network.get_connectivity_rand(
                pre=pre, post=post,
                connprob=connectionProbability[i][j]
                )
            print(np.shape(connectivity))

            # connect network:
            (conncount, syncount) = network.connect(
                pre=pre, post=post,
                connectivity=connectivity,
                syntype=synapseModel,
                synparams=synapseParameters[i][j],
                weightfun=np.random.normal,
                weightargs=weightArguments[i][j],
                minweight=minweight,
                delayfun=delayFunction,
                delayargs=delayArguments[i][j],
                mindelay=mindelay,
                multapsefun=multapseFunction,
                multapseargs=multapseArguments[i][j],
                syn_pos_args=synapsePositionArguments[i][j],
                save_connections = True, # Creates synapse_positions.h5
                )
    # set up extracellular recording device:
    electrode = RecExtElectrode(**electrodeParameters)
    EEG_electrode_params = dict(
        x=0,
        y=0,
        z=90000.,
        method="soma_as_point"
        )
    EEG_electrode = RecExtElectrode(**EEG_electrode_params)
    # run simulation:
    SPIKES, OUTPUT, DIPOLEMOMENT = network.simulate(
        electrode=electrode,
        # electrode = EEG_electrode,
        **networkSimulationArguments,
    )

    # collect somatic potentials across all RANKs to RANK 0:
    if RANK == 0:
        somavs = []
        for i, name in enumerate(population_names):
            somavs.append([])
            somavs[i] += [cell.somav
                          for cell in network.populations[name].cells]
            for j in range(1, SIZE):
                somavs[i] += COMM.recv(source=j, tag=15)
    else:
        somavs = None
        for name in population_names:
            COMM.send([cell.somav for cell in network.populations[name].cells],
                      dest=0, tag=15)

    ############################################################################
    # Save data for plots
    ############################################################################
    fi=open("example_network_output/somavs.pkl","wb")
    pickle.dump(somavs,fi)
    fi.close()

    fi=open("example_network_output/spikes.pkl","wb")
    pickle.dump(SPIKES,fi)
    fi.close()

    fi=open("example_network_output/dipoles.pkl","wb")
    pickle.dump(DIPOLEMOMENT,fi)
    fi.close()

    fi=open("example_network_output/pop_names.pkl","wb")
    pickle.dump(population_names,fi)
    fi.close()

    fi=open("example_network_output/network_dt.pkl","wb")
    pickle.dump(network.dt,fi)
    fi.close()


############################################################################
# customary cleanup of object references - the psection() function may not
# write correct information if NEURON still has object references in memory,
# even if Python references has been deleted. It will also allow the script
# to be run in successive fashion.
############################################################################
#network.pc.gid_clear() # allows assigning new gids to threads
electrode = None
syn = None
synapseModel = None
# for population in network.populations.values():
#     for cell in population.cells:
#         cell = None
#     population.cells = None
    # population = None
pop = None
network = None
neuron.h('forall delete_section()')
total_time = time.time() - start
if RANK == 0:
    print("total runtime:", total_time,"s.")
