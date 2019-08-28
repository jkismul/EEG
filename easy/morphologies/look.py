#!/usr/bin/env python

import LFPy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colorbar
import neuron as nrn
import scipy.stats as st
import random


cell_parameters = {'morphology': 'ball_and_2_sticks.hoc'}
cell = LFPy.Cell(**cell_parameters)
cell.set_pos(x=0,y=0,z=0)
cell.set_rotation(x=0,y=0,z=0)




plt.close('all')
fig = plt.figure()

for idx in range(cell.totnsegs):
    print(idx)
    if idx == 0:
        plt.plot(cell.xmid[idx], cell.zmid[idx], 'ro')
    if idx != 0:
        plt.plot([cell.xstart[idx], cell.xend[idx]],
                [cell.zstart[idx], cell.zend[idx]], c='k')


plt.show()