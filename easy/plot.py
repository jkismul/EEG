# import modules:
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.signal as ss
import scipy.stats as st
# from mpi4py import MPI
import neuron
# from LFPy import NetworkCell, Network, Synapse, RecExtElectrode, FourSphereVolumeConductor
import LFPy
import csv
import pickle

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

#########################################################################
#########################################################################
#########################################################################

##################################
# LOAD DATA FROM FILES
##################################

with open('example_network_output/network_dt.pkl','rb') as handle:
  network_dt = pickle.loads(handle.read())

# with open('example_network_output/eeg_top.pkl','rb') as handle:
#   eeg_top = pickle.loads(handle.read())

with open('example_network_output/dipoles.pkl','rb') as handle:
  DIPOLEMOMENT = pickle.loads(handle.read())

with open('example_network_output/spikes.pkl','rb') as handle:
  SPIKES = pickle.loads(handle.read())

with open('example_network_output/pop_names.pkl','rb') as handle:
  population_names = pickle.loads(handle.read())

with open('example_network_output/somavs.pkl','rb') as handle:
  somavs = pickle.loads(handle.read())

###########################################################################
###########################################################################
###########################################################################

# CREATE SINGLE PASSIVE CELL FOR PLOT
cell_parameters = {
    'morphology' : 'morphologies/ball_and_2_sticks.hoc', # from Mainen & Sejnowski, J Comput Neurosci, 1996
    'cm' : 1.0,         # membrane capacitance
    'Ra' : 150.,        # axial resistance
    #'v_init' : -65.,    # initial crossmembrane potential
    'passive' : True,   # turn on NEURONs passive mechanism for all sections
    'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -70},
    'nsegs_method' : 'lambda_f', # spatial discretization method
    'lambda_f' : 100.,           # frequency where length constants are computed
    'dt' : 2.**-5,      # simulation time step size
    'tstart' : -200.,      # start time of simulation, recorders start at t=0
    'tstop' : 25.,     # stop simulation at 100 ms.
}

# Create cell
cell = LFPy.Cell(**cell_parameters)
# Align cell
cell.set_rotation(x=4.99, y=-4.33, z=3.14)

# Define synapse parameters
synapse_parameters = {
    # 'idx' : 0,
    'idx' :cell.get_closest_idx(x=0., y=0., z=0.),
    'e' : 0.,                   # reversal potential
    'syntype' : 'Exp2Syn',       # synapse type
    'tau1' : 1.,                 # synaptic time constant
    'tau2' : 3.,
    'weight' : .000001*5000,            # synaptic weight
    'record_current' : True,    # record synapse current
}

# Create synapse and set time of synaptic input
synapse = LFPy.Synapse(cell, **synapse_parameters)
synapse.set_spike_times(np.array([2.]))

cell.simulate(rec_current_dipole_moment=True)
hum = cell.current_dipole_moment
# print(np.shape(cell.current_dipole_moment))
# print(np.shape(hum[:,2]))
# print(np.shape(cell.tvec))
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(cell.tvec,-1*cell.current_dipole_moment[:,2])
ax2 = fig.add_subplot(212)
ax2.plot(cell.tvec,cell.somav)

fig.savefig('plots/single_cell.pdf')
plt.close(fig)
###########################################################################
###########################################################################
###########################################################################


# PLOT DATA

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

fig.savefig('plots/soma_potentials.pdf')
plt.close(fig)

################################################################
# EEG
# current-dipole moments E, z dir
################################################################
fig = plt.figure()
fig.suptitle('current-dipole moment')
plt.xlabel('t [ms]')
plt.ylabel(r'$\mathbf{p}\cdot\mathbf{e}_z$ (nA$\mu$m)')
t = np.arange(DIPOLEMOMENT.shape[0])*network_dt
inds = (t >= 0) & (t <= 25)
plt.plot(t[inds][::16], decimate(DIPOLEMOMENT['E'][inds, 2] +
 DIPOLEMOMENT['I'][inds, 2], q=16))
# plt.plot(t[inds],eeg_top[0,:],'r')
fig.savefig('plots/current_dipole_moment.pdf')
plt.close(fig)

##############################################
# spike raster
##############################################

fig, ax = plt.subplots(1, 1)
for name, spts, gids in zip(population_names, SPIKES['times'], SPIKES['gids']):
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
fig.savefig('plots/spike_raster.pdf')
plt.close(fig)



################################################################
# extracellular potentials, E and I contributions, sum
###############################################################
fig, axes = plt.subplots(1, 3, figsize=(6.4, 4.8))
fig.suptitle('extracellular potentials')
import h5py
filename = 'example_network_output/OUTPUT.h5'
f = h5py.File(filename,'r')
for i, (ax, name, label) in enumerate(zip(axes, ['E', 'I', 'imem'],
                                          ['E', 'I', 'sum'])):
    draw_lineplot(ax, decimate(f["OUTPUT[0]"][name], q=16), dt=network_dt*16,
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
fig.savefig('plots/extracellular_potential.pdf')
plt.close(fig)
f.close()

###############################################
# population illustration (per RANK)
###############################################
fig = plt.figure(figsize=(6.4, 4.8*2))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=5)
import h5py
filename = 'example_network_output/cell_positions_and_rotations.h5'
f = h5py.File(filename,'r')

with open('example_network_output/cells.pkl','rb') as handle:
  CellParams = pickle.loads(handle.read())


cells = [LFPy.Cell(morphology = CellParams[0]['morphology']),LFPy.Cell(morphology = CellParams[1]['morphology']) ]

#NOTE: X and Y rotations are hard set to 0 here.
for i, (name, pop) in enumerate(f.items()):
    pop = len(f[name])
    cell = cells[i]
    for j in range(pop):
      cell.set_pos(f[name][(j)][(1)],f[name][(j)][(2)],f[name][(j)][(3)])
      cell.set_rotation(0,0,f[name][(j)][(6)])
      c = 'C0' if name == 'E' else 'C1'
      for idx in range(cell.totnsegs):
            if idx == 0: #soma
                ax.scatter(cell.xmid[idx],cell.ymid[idx], cell.zmid[idx],c = c)
            else: #dendrites
                ax.plot([cell.xstart[idx],cell.xend[idx]], [cell.ystart[idx],cell.yend[idx]],
                        [cell.zstart[idx],cell.zend[idx]],c)
f.close()
filename = 'example_network_output/synapse_positions.h5'
f = h5py.File(filename, 'r')

for i in range(len(f["E:E"][()])):
    ax.scatter(f["E:E"][i][2],f["E:E"][i][3],f["E:E"][i][4],s=4,c='y',marker = "*")
for i in range(len(f["E:I"][()])):
    ax.scatter(f["E:I"][i][2],f["E:I"][i][3],f["E:I"][i][4],s=4,c='g',marker = "*")
for i in range(len(f["I:E"][()])):
    ax.scatter(f["I:E"][i][2],f["I:E"][i][3],f["I:E"][i][4],s=4,c='r',marker = "*")
for i in range(len(f["I:I"][()])):
    ax.scatter(f["I:I"][i][2],f["I:I"][i][3],f["I:I"][i][4],s=4,c='k',marker = "*")


ax.set_xlabel('$x$ ($\mu$m)')
ax.set_ylabel('$y$ ($\mu$m)')
ax.set_zlabel('$z$ ($\mu$m)')
ax.set_title('network populations')
fig.savefig('plots/population.pdf')
plt.close(fig)
f.close()
###############################################################
###############################################################
