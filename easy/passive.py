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
import h5py
# from collections import OrderedDict

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
# Function declarations
################################################################################




def decimate(x, q=10, n=4, k=0.8, filterfun=ss.cheby1):
    """
    scipy.signal.decimate like downsampling using filtfilt instead of lfilter,
    and filter coeffs from butterworth or chebyshev type 1.

    Parameters
    ----------
    x : ndarray
        Array to be downsampled along last axis.
    q : int
        Downsampling factor.
    n : int
        Filter order.
    k : float
        Aliasing filter critical frequency Wn will be set as Wn=k/q.
    filterfun : function
        `scipy.signal.filter_design.cheby1` or
        `scipy.signal.filter_design.butter` function

    Returns
    -------
    ndarray
        Downsampled signal.

    """
    if not isinstance(q, int):
        raise TypeError("q must be an integer")

    if n is None:
        n = 1

    if filterfun == ss.butter:
        b, a = filterfun(n, k / q)
    elif filterfun == ss.cheby1:
        b, a = filterfun(n, 0.05, k / q)
    else:
        raise Exception('only ss.butter or ss.cheby1 supported')

    try:
        y = ss.filtfilt(b, a, x)
    except: # Multidim array can only be processed at once for scipy >= 0.9.0
        y = []
        for data in x:
            y.append(ss.filtfilt(b, a, data))
        y = np.array(y)

    try:
        return y[:, ::q]
    except:
        return y[::q]

def remove_axis_junk(ax, lines=['right', 'top']):
    """remove chosen lines from plotting axis"""
    for loc, spine in ax.spines.items():
        if loc in lines:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def draw_lineplot(
        ax, data, dt=0.1,
        T=(0, 200),
        scaling_factor=1.,
        vlimround=None,
        label='local',
        scalebar=True,
        unit='mV',
        ylabels=True,
        color='r',
        ztransform=True,
        filter_data=False,
        filterargs=dict(N=2, Wn=0.02, btype='lowpass')):
    """helper function to draw line plots"""
    tvec = np.arange(data.shape[1])*dt

    tinds = (tvec >= T[0]) & (tvec <= T[1])

    # apply temporal filter
    if filter_data:
        b, a = ss.butter(**filterargs)
        data = ss.filtfilt(b, a, data, axis=-1)

    #subtract mean in each channel
    if ztransform:
        dataT = data.T - data.mean(axis=1)
        data = dataT.T

    zvec = -np.arange(data.shape[0])
    vlim = abs(data[:, tinds]).max()

    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim)) / scaling_factor
    else:
        pass

    yticklabels = []
    yticks = []

    for i, z in enumerate(zvec):
        if i == 0:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z, lw=1,
                    rasterized=False, label=label, clip_on=False,
                    color=color)
        else:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z, lw=1,
                    rasterized=False, clip_on=False,
                    color=color)
        yticklabels.append('ch. %i' % (i+1))
        yticks.append(z)

    if scalebar:
        ax.plot([tvec[-1], tvec[-1]],
                [-1, -2], lw=2, color='k', clip_on=False)
        ax.text(tvec[-1]+np.diff(T)*0.02, -1.5,
                '$2^{' + '{}'.format(np.log2(vlimround)
                                    ) + '}$ ' + '{0}'.format(unit),
                color='k', rotation='vertical',
                va='center')

    ax.axis(ax.axis('tight'))
    ax.yaxis.set_ticks(yticks)
    if ylabels:
        ax.yaxis.set_ticklabels(yticklabels)
        ax.set_ylabel('channel', labelpad=0.1)
    else:
        ax.yaxis.set_ticklabels([])
    remove_axis_junk(ax, lines=['right', 'top'])
    ax.set_xlabel(r't (ms)', labelpad=0.1)

    return vlimround

##################################
# LOAD DATA FROM FILES
##################################

### CONNECTIONS
filename = 'example_network_output/synapse_positions.h5'
f = h5py.File(filename,'r')

with open('example_network_output/spikes.pkl','rb') as handle:
  SPIKES = pickle.loads(handle.read())

################################################################################
# Set up shared and population-specific parameters
################################################################################

from param import spike_t,spike_std,distr_t,weighttrain, cellParameters, OUTPUTPATH, populationParameters, networkParameters, electrodeParameters, networkSimulationArguments, num_cells, population_names,population_sizes, connectionProbability, synapseModel,synapseParameters, weightArguments, weightFunction, minweight, delayFunction, delayArguments, mindelay, multapseFunction, multapseArguments, synapsePositionArguments
# from param import *

# L5 Excitatory
cellParameters[0]['morphology'] = 'morphologies/ball_and_2_sticks_pas.hoc'
cellParameters[0]['templatefile'] = 'morphologies/ball_and_2_sticks_pas_Template.hoc'
cellParameters[0]['templatename'] = 'ball_and_2_sticks_pas_Template'

#L4 inhibitory
cellParameters[1]['morphology'] = 'morphologies/Stellate_pas.hoc'
cellParameters[1]['templatefile'] = 'morphologies/Stellate_pas_Template.hoc'
cellParameters[1]['templatename'] = 'Stellate_pas_Template'

#dont overwrtie files
networkSimulationArguments['rec_current_dipole_moment'] = False
networkSimulationArguments['to_file'] = False

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
    # print("Aaaaaaaaaaaa", f.items())
    syns = []
    for x in f.items():

        nm = str(x[0])
        a = f[nm][()]
        syns.append(a)
    # print(syns)
    # print(SPIKES['times'])
    # create E and I populations:
    for ii, (name, size) in enumerate(zip(population_names, population_sizes)):
        if RANK == 0:
            print(ii, name, size)
        network.create_population(name=name, POP_SIZE=size,
                                  **populationParameters[ii])
        #initial train and all spikes
        if name =='E': #Receiving cell is excitatory
            for j,cell in enumerate(network.populations[name].cells):
                if j%4==0:

                    weighttrain = np.random.normal(0.05,0.02)
                    idx = cell.get_rand_idx_area_norm(section='dend', nidx=1)
                    for i in idx:
                        syn = Synapse(cell=cell, idx=i, syntype='Exp2Syn',
                                      weight=weighttrain,
                                      **dict(synapseParameters[0][0]))
                        syn.set_spike_times(np.array([distr_t[j]]))

                for i in range(len(syns[0])): #E:E og ei
                    if syns[0][i][0] == j: #ii første?
                        pre_gid = syns[0][i][1]
                        ind = SPIKES['gids'][0].index(pre_gid) #i her?
                        tim = SPIKES['times'][0][ind]
                        if tim.size>0:
                            # print("EE", tim)
                            x = syns[0][i][2]
                            y = syns[0][i][3]
                            z = syns[0][i][4]
                            idx = cell.get_closest_idx(x,y,z)
                            syn = Synapse(cell=cell, idx=idx, syntype='Exp2Syn',
                                         weight = weightArguments[0][0]['loc'],
                                         **dict(synapseParameters[0][0]))
                            syn.set_spike_times(np.array([tim+delayArguments[0][0]['loc']]))
                for i in range(len(syns[2])): #I:E
                    if syns[2][i][0] == j:
                        pre_gid = syns[2][i][1]
                        ind = SPIKES['gids'][1].index(pre_gid)
                        tim = SPIKES['times'][1][ind]
                        if tim.size>0:
                            # print("IE",tim)
                            x = syns[2][i][2]
                            y = syns[2][i][3]
                            z = syns[2][i][4]
                            idx = cell.get_closest_idx(x,y,z)
                            syn = Synapse(cell=cell, idx=idx, syntype='Exp2Syn',
                                          weight= weightArguments[1][0]['loc'],
                                          **dict(synapseParameters[1][0]))
                            syn.set_spike_times(np.array([tim+delayArguments[1][0]['loc']]))
                                    
        if name =='I': #Receiving cell is inhibitory
            for j,cell in enumerate(network.populations[name].cells):
                j = j+population_sizes[0]
                for i in range(len(syns[1])): #E:I
                    if syns[1][i][0] == j:
                        pre_gid = syns[1][i][1]
                        ind = SPIKES['gids'][0].index(pre_gid)
                        tim = SPIKES['times'][0][ind]
                        if tim.size>0:
                            # print("EI",tim)
                            x = syns[1][i][2]
                            y = syns[1][i][3]
                            z = syns[1][i][4]
                            idx = cell.get_closest_idx(x,y,z)
                            syn = Synapse(cell=cell, idx=idx, syntype='Exp2Syn',
                                          weight=weightArguments[0][1]['loc'],
                                          **dict(synapseParameters[0][1]))
                            syn.set_spike_times(np.array([tim+delayArguments[0][1]['loc']]))

                for i in range(len(syns[3])): #I:I
                    if syns[3][i][0] == j:
                        pre_gid = syns[3][i][1] 
                        ind = SPIKES['gids'][1].index(pre_gid)
                        tim = SPIKES['times'][1][ind]
                        if tim.size>0:
                            # print("II",tim)
                            x = syns[3][i][2]
                            y = syns[3][i][3]
                            z = syns[3][i][4]
                            idx = cell.get_closest_idx(x,y,z)
                            syn = Synapse(cell=cell, idx=idx, syntype='Exp2Syn',
                                          weight=weightArguments[1][1]['loc'],
                                          **dict(synapseParameters[1][1]))
                            syn.set_spike_times(np.array([tim+delayArguments[1][1]['loc']]))
   
    for i, pre in enumerate(population_names):
        for j, post in enumerate(population_names):
            # boolean connectivity matrix between pre- and post-synaptic neurons
            # in each population (postsynaptic on this RANK)
            connectivity = network.get_connectivity_rand(
                pre=pre, post=post,
                connprob=connectionProbability[i][j]
                )

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
                save_connections = False, # Creates synapse_positions.h5
                )
    electrode = RecExtElectrode(**electrodeParameters)
    EEG_electrode_params = dict(
        x=0,
        y=0,
        z=90000.,
        method="soma_as_point"
        )
    EEG_electrode = RecExtElectrode(**EEG_electrode_params)

    # run simulation:
    SPIKES2, OUTPUT2, DIPOLEMOMENT2 = network.simulate(
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

f.close()

with open('example_network_output/network_dt.pkl','rb') as handle:
  network_dt = pickle.loads(handle.read())


###RASTER###
fig, ax = plt.subplots(1, 1)
for name, spts, gids in zip(population_names, SPIKES2['times'], SPIKES2['gids']):
    t = []
    g = []
    for spt, gid in zip(spts, gids):
        t = np.r_[t, spt]
        g = np.r_[g, np.zeros(spt.size)+gid]
    ax.plot(t[t >= 0], g[t >= 0], '.', label=name)
ax.legend(loc=0)
remove_axis_junk(ax, lines=['right', 'top'])
ax.set_xlabel('t (ms)')
ax.set_ylabel('gid')
ax.set_title('spike raster')
fig.savefig('plots/spike_raster_pas.pdf')
plt.close(fig)

##extracellular pots
fig, axes = plt.subplots(1, 3, figsize=(6.4, 4.8))
fig.suptitle('extracellular potentials')
for i, (ax, name, label) in enumerate(zip(axes, ['E', 'I', 'imem'],
                                          ['E', 'I', 'sum'])):
    draw_lineplot(ax, decimate(OUTPUT2[0][name], q=16), dt=network_dt*16,
                  T=(0, 25),
                  scaling_factor=1.,
                  vlimround=None,
                  label=label,
                  scalebar=True,
                  unit='mV',
                  ylabels=True if i == 0 else False,
                  color='C{}'.format(i),
                  ztransform=True
                 )
    ax.set_title(label)
fig.savefig('plots/extracellular_potential_passive.pdf')
plt.close(fig)
f.close()


#plot of top LFP 
filename = 'example_network_output/OUTPUT.h5'
f = h5py.File(filename,'r')
fig, axes = plt.subplots(1, 1, figsize=(6.4, 4.8))
fig.suptitle('Ch1')

timses = np.linspace(0,25,801)
plt.plot(timses,OUTPUT2[0]['E'][0])
difr = f["OUTPUT[0]"]['E'][0][0]-OUTPUT2[0]["E"][0][0]

plt.plot(timses,f["OUTPUT[0]"]['E'][0]-difr,'r')
fig.savefig('plots/ch1.pdf')
plt.close(fig)
f.close()


###########################################
# somatic potentials
###########################################
fig = plt.figure()
gs = GridSpec(5, 1)
ax = fig.add_subplot(gs[:4])
draw_lineplot(ax, decimate(np.array(somavs[0]), q=8), dt=network_dt*16,
              T=(0, 25),
              scaling_factor=1.,
              vlimround=16,
              label='E',
              scalebar=True,
              unit='mV',
              ylabels=False,
              color='C0',
              ztransform=True
             )
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_ylabel('E')
ax.set_title('somatic potentials')
ax.set_xlabel('')
ax = fig.add_subplot(gs[4])
draw_lineplot(ax, decimate(np.array(somavs[1]), q=8), dt=network_dt*16,
              T=(0, 25),
              scaling_factor=1.,
              vlimround=16,
              label='I',
              scalebar=True,
              unit='mV',
              ylabels=False,
              color='C1',
              ztransform=True
             )
ax.set_yticks([])
ax.set_ylabel('I')

fig.savefig('plots/soma_potentials_passive.pdf')
plt.close(fig)



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
