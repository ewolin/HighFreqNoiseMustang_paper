#!/usr/bin/env python

from noiseplot import setupPSDPlot

import numpy as np


for mod in ['High_T-vs-dB.txt', 'Low_Perm_T-vs-dB.txt', 'Low_Port_T-vs-dB.txt']:
    fig, ax = setupPSDPlot()
    t, db = np.loadtxt(mod, unpack=True)
    ax.plot(t, db, ls='-', color='red', marker='o')
    ax.set_xlim(1e-2, 1)
    fig.show()
    fig.savefig('{0}.png'.format(mod))
    
