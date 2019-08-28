from LFPy import NetworkCell, Network, Synapse, RecExtElectrode, FourSphereVolumeConductor
import numpy as np
import neuron
import scipy.stats as st

# relative path for simulation output:
OUTPUTPATH = 'example_network_output'



# class NetworkCell parameters:
cellParameters = [dict() for x in range(2)] # NOTE: 2 is the number of different cell/pop types

# L5 Excitatory
cellParameters[0] = {                                                         # set params as usual, just include template stuff
    'morphology':'morphologies/ball_and_2_sticks.hoc',                        # load cell morphology
    'templatefile':'morphologies/ball_and_2_sticks_Template.hoc',             # load template of cell, to create network
    'templatename':'ball_and_2_sticks_Template',                              # template file can hold several templates

    'cm' : 1.0,                                                 # membrane capacitance
    'Ra' : 150,                                                 # axial resistance
    'tstart': -200.,                                            # start time
    'passive' : True,                                           # switch on passive mechs
    'nsegs_method' : 'lambda_f',                                # method for setting number of segments,
    'lambda_f' : 100,                                           # segments are isopotential at this frequency
    'passive_parameters' : {'g_pas' : 0.0002, 'e_pas' : -65.}, # passive params
    'tstop': 25,                                                # stop time
    'templateargs':None,                                        # JFK: no idea. Parameters provided to template-definition
    'delete_sections':False,                                    # JFK: no idea. delete pre-existing section-references.
    }

#L4 inhibitory
cellParameters[1] = {                                           # set params as usual, just include template stuff    
    'morphology':'morphologies/Stellate.hoc',                   # load cell morphology
    'templatefile':'morphologies/Stellate_Template.hoc',        # load template of cell, to create network
    'templatename':'Stellate_Template',                         # template file can hold several templates
    
    'cm' : 1.0,                                                 # membrane capacitance
    'Ra' : 150,                                                 # axial resistance
    'tstart': -200.,                                            # start time
    'passive' : True,                                           # switch on passive mechs
    'nsegs_method' : 'lambda_f',                                # method for setting number of segments,
    'lambda_f' : 100,                                           # segments are isopotential at this frequency
    'passive_parameters' : {'g_pas' : 0.0002, 'e_pas' : -65.}, # passive params
    'tstop': 25,                                                # stop time
    'templateargs':None,                                        # JFK: no idea. Parameters provided to template-definition
    'delete_sections':False,                                    # JFK: no idea. delete pre-existing section-references.
    }

# class Population parameters
populationParameters = [dict() for x in range(2)]

# L5 excitatory pop
populationParameters[0] = (
    dict(
    Cell= NetworkCell,                                    # to create network
    cell_args=cellParameters[0],                          # load params from above
    pop_args=dict(                                        # parameters for population
        radius=100.,                                      # place within this radius, micrometer
        loc=77500.,                                           # with this avg
        scale=20.),                                       # and this std
    rotation_args=dict(x=0., y=0.),                       # Cells are randomly rotated around z-axis using the Cell.set_rotation method.
    )
)

# L4 inhibitory pop
populationParameters[1] = (
    dict(
    Cell= NetworkCell,                                    # to create network
    cell_args=cellParameters[1],                         # load params from above
    pop_args=dict(                                        # parameters for population
        radius=100.,                                      # place within this radius, micrometer
        loc=77850.,      #350 == (525\2+190\2) from layer thickness microcircuit                                     # with this avg
        scale=20.),                                       # and this std
    rotation_args=dict(x=0., y=0.),                       # Cells are randomly rotated around z-axis using the Cell.set_rotation method.
    )
)

# class Network parameters:
networkParameters = dict(
    dt=2**-5,
    tstart=-200.,
    tstop=25.,
    v_init=-64.,
    celsius=37.0,                    # JFK: Mainen was done at 23C, using a Qfactor to upscale to 37.
    OUTPUTPATH=OUTPUTPATH
)

# class RecExtElectrode parameters:
electrodeParameters = dict(                        # creates 13 electrodes at x,y origin, every 100micrometer
    x=np.zeros(13),
    y=np.zeros(13),
    z=np.linspace(78500., 77300., 13),
    N=np.array([[0., 1., 0.] for _ in range(13)]), # Normal vectors [x, y, z] of each circular electrode contact surface, default None
    r=5.,                                          # radius of each contact surface, default None
    n=50,                                          # if N is not None and r > 0, the number of discrete points used to compute the n-point average potential on each circular contact point.
    sigma=0.3,                                     # extracellular conductivity in units of (S/m).
    method="soma_as_point"                         # switch between the assumption of ‘linesource’, ‘pointsource’, ‘soma_as_point’ to represent each compartment when computing extracellular potentials
)

# method Network.simulate() parameters:
networkSimulationArguments = dict(
    rec_current_dipole_moment=True,                 # If True, compute and record current-dipole moment from transmembrane currents
    rec_pop_contributions=True,                     # If True, compute and return single-population contributions to the extracellular potential during simulation time
    to_memory=True,
    to_file=True, # Creates OUTPUT.h5
    rec_vmem =True
)

# population names, sizes and connection probability:
num_cells=100
population_names = ['E', 'I']



population_sizes = [int(0.80*num_cells), int(0.20*num_cells)]
connectionProbability = [[0.05, 0.05], [0.05, 0.05]]

# synapse model. All corresponding parameters for weights,
# connection delays, multapses and layerwise positions are
# set up as shape (2, 2) nested lists for each possible
# connection on the form:
# [["E:E", "E:I"],
#  ["I:E", "I:I"]].
weighttrain = 0.007
synapseModel = neuron.h.Exp2Syn
# synapse parameters
synapseParameters = [[dict(tau1=1.0, tau2=3.0, e=0.),
                      dict(tau1=1.0, tau2=3.0, e=0.)],
                     [dict(tau1=1.0, tau2=12.0, e=-80.),
                      dict(tau1=1.0, tau2=12.0, e=-80.)]]
# synapse max. conductance (function, mean, st.dev., min.):
weightFunction = np.random.normal
weightArguments = [[dict(loc=0.002, scale=0.0002),
                    dict(loc=0.002, scale=0.0002)],
                   [dict(loc=0.01, scale=0.001),
                    dict(loc=0.01, scale=0.001)]]
minweight = 0.
# conduction delay (function, mean, st.dev., min.):
delayFunction = np.random.normal
delayArguments = [[dict(loc=1.5, scale=0.),
                   dict(loc=1.5, scale=0.)],
                  [dict(loc=1.5, scale=0.),
                   dict(loc=1.5, scale=0.)]]
mindelay = 0.3
multapseFunction = np.random.normal
multapseArguments = [[dict(loc=2., scale=.5), dict(loc=2., scale=.5)],
                     [dict(loc=5., scale=1.), dict(loc=5., scale=1.)]]
# method NetworkCell.get_rand_idx_area_and_distribution_norm
# parameters for layerwise synapse positions:
synapsePositionArguments = [[dict(section=['soma', 'apic'],
                                  fun=[st.norm, st.norm], ### soma greia lager vel bare krøll her, skal ikke koble E til soma
                                  funargs=[dict(loc=78000.,scale=100.),#loc=77500., scale=0.0001),#750ish, 
                                           dict(loc=78000., scale=100.)],
                                  funweights=[0.5, 1.]
                                 ) for _ in range(2)],
                            [dict(section=['soma', 'dend'],
                            # [dict(section=['soma', 'soma'],
                                  fun=[st.norm, st.norm],
                                  funargs=[dict(loc=77850., scale=100.0001),
                                           dict(loc=77850., scale=100.0001)],
                                  funweights=[1., 0.5]
                                 ) for _ in range(2)]]

spike_t = 2.0
spike_std = 0.0
distr_t = np.random.normal(spike_t,spike_std,size=population_sizes[0]+population_sizes[1])
