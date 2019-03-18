import os

import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import pandas as pd

def plotBrune(ax):
    '''Plot Brune corner frequencies for mag/dist ranges of interest
       (see esupp for details on calculation)'''
    # Read file containing list of corner frequencies
    dirname = os.path.dirname(__file__)
    brunecsv = os.path.join(dirname,'brune-all.csv')
    df = pd.read_csv(brunecsv, names=['delta', 'mag', 'T', 'dB'], 
                     skiprows=1, sep=',')
    df = df[df['mag'] <= 4]

    # Plot grid of corner frequencies:
    # Draw lines connecting all magnitudes at a given distance,
    # then all distances at a given magnitude 
    gridcolor='#1f77b4' # darker blue
    linestyle={'color' : gridcolor,
               'mfc' : gridcolor,
               'linewidth' : 1,
               'linestyle' : '--'}
    deltas = pd.unique(df['delta'])
    for delta in deltas:
        df_subset = df[df['delta'] == delta]
        if delta == 1:
            label=r'Brune f$_c$'
        else:
            label=''
        ax.plot(df_subset['T'], df_subset['dB'], marker='.', 
                **linestyle, label=label)
    mags = pd.unique(df['mag'])
    for mag in mags:
        df_subset = df[df['mag'] == mag]
        ax.plot(df_subset['T'], df_subset['dB'], marker='None', 
                **linestyle, label='')

    # Plot labels on grid for magnitudes and distances
    df_text = df[df['delta'] == 0.01]
    mags = np.arange(1,5)
    for mag in mags:
        df_plot = df_text[df_text['mag'] == mag]
        ax.text(df_plot['T'], df_plot['dB']+4, 
                "M{0}".format(mag), va='center', ha='center', 
                color=gridcolor)
    df_text = df[df['mag'] == 4]
    for delta in deltas:
        df_plot = df_text[df_text['delta'] == delta]
        ax.text(df_plot['T']+0.02, df_plot['dB']+2.5, 
                "{0:0d} km".format(int(delta*100)), va='center', 
                rotation=30, color=gridcolor)
        

def setupPSDPlot():
    '''Set up a plot with Peterson noise model for plotting PSD curves.
       x axis = period (s)
       y axis = decibels '''
    # Set paths to various noise models
    codedir = os.path.dirname(__file__)
    piecewise = os.path.join(codedir,'PiecewiseModels')
    nhnm = np.loadtxt(codedir+'/peterson_HNM.mod', unpack=True)
    nlnm = np.loadtxt(codedir+'/peterson_LNM.mod', unpack=True)
    nhnb = np.loadtxt(piecewise+'/High_T-vs-DB.txt', unpack=True)
    nlportb = np.loadtxt(piecewise+'/Low_Port_T-vs-dB.txt', unpack=True)
    nlpermb = np.loadtxt(piecewise+'/Low_Perm_T-vs-dB.txt', unpack=True)
    #nlportb = np.loadtxt(piecewise+'/stitch.txt', unpack=True)

    # Set up axes
    width=5
    fig = plt.figure(figsize=(width,width/1.14))
    gs_plots = gridspec.GridSpec(1,2, width_ratios=[1, 0.05])
    ax = fig.add_subplot(gs_plots[0,0])
    ax_cb = fig.add_subplot(gs_plots[0,1])

#    colorlist = ['gold', '#ff7f0e', '#d62728']
#    colorlist = ['green', 'gold', '#ff7f0e', '#d62728']
    colorlist = ['gold', '#ff7f0e', '#d62728', 'green']
    ax.set_prop_cycle(cycler('color', colorlist))

    ax.semilogx()
    ax.set_xlim(0.05, 200)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel(r'Power (dB[m$^2$/s$^4$/Hz])')
    ax.set_ylim(-200, -50)

    # Plot Peterson noise models 
    ax.plot(nhnm[0], nhnm[1], linewidth=2, color='black', label='NHNM/NLNM')
    ax.plot(nlnm[0], nlnm[1], linewidth=2, color='black')

#    icheck, = np.where((nlnm[0] >= 1./0.4)&(nlnm[0]<=1./0.2))
    f_min = 0.2 
    f_max = 0.4
    icheck, = np.where((nlnm[0] >= 1./f_max)&(nlnm[0]<=1./f_min))
    ax.plot(nlnm[0][icheck], nlnm[1][icheck]-5, linewidth=1, color='grey', linestyle='--')
    #print(nlnm[0][icheck])

    # Plot high-frequency extensions 
    #ax.plot(nlpermb[0], nlpermb[1], color='grey', ls='-.', lw=3, 
    #        label='Low Permanent Baseline')
    ax.plot(nlportb[0], nlportb[1], linewidth=3, ls='--', color='grey', 
            label='Low Portable Baseline')
    ax.plot(nhnb[0], nhnb[1], color='grey', ls=(0,(1,1)), lw=3, 
            label='High Baseline')

    # Plot Brune corner frequency grid
    plotBrune(ax)
# dummy point w/no labels in case we don't plot HF noise models
    #ax.plot(np.zeros(1), np.zeros([1]), color='w', alpha=0, label=' ')

    return fig, ax
