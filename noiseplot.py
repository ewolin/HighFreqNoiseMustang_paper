import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import pandas as pd

from cycler import cycler


def plotBrune(ax):

    # Brune spectra:
#    ax.plot(0.0109, -130.427, marker='*') # M0.5 at 1 km
#    ax.plot(0.109, -136.005, marker='*') # M2.5 at 100 km
#    ax.plot(0.109, -110.726, marker='*') # M2.5 at 10 km
#    ax.plot(0.109, -90.198, marker='*') # M0.5 at 1 km

    gridcolor='#d62728' #new mpl style: red
    gridcolor='#9467bd' #new mpl style: purple
    gridcolor='#36b1bf' # light blue
    gridcolor='limegreen' 
    gridcolor='#1f77b4' # darker blue
#    gridcolor='#bcbd22' # darker blue

    brunecsv='/Users/ewolin/research/NewNewNoise/brune-all.csv'
    linestyle={'color' : gridcolor,
               'mfc' : gridcolor,
               'linewidth' : 1,
               'linestyle' : '--'}
#               'label' : ''}

    df = pd.read_csv(brunecsv, names=['delta', 'mag', 'T', 'dB'], skiprows=1, sep=',')
    df = df[df['mag'] <= 4]

    deltas = pd.unique(df['delta'])
    for delta in deltas:
        df_subset = df[df['delta'] == delta]
        if delta == 1:
            label=r'Brune f$_c$'
#            label=r'$P$-wave f$_c$'
        else:
            label=''
        ax.plot(df_subset['T'], df_subset['dB'], marker='.', **linestyle, label=label)
    mags = pd.unique(df['mag'])
    label=''
    for mag in mags:
        df_subset = df[df['mag'] == mag]
        ax.plot(df_subset['T'], df_subset['dB'], marker='None', **linestyle, label='')

# plot text labels: ax.text requires only 1 value for x and y
# do magnitude ones first
    df_text = df[df['delta'] == 0.01]
    mags = np.arange(1,5)
    for mag in mags:
        df_plot = df_text[df_text['mag'] == mag]
        ax.text(df_plot['T'], df_plot['dB']+4, "M{0}".format(mag), va='center', ha='center', color=gridcolor)
    df_text = df[df['mag'] == 4]
    for delta in deltas:
        df_plot = df_text[df_text['delta'] == delta]
        ax.text(df_plot['T']+0.02, df_plot['dB']+2.5, "{0:0d} km".format(int(delta*100)), va='center', rotation=30, color=gridcolor)
        
#        ax.text(1, -160, 'hello')#df_subset['T'], df_subset['dB']+5, 'hello')
#        ax.text(df_subset['T'], df_subset['dB'], 'hello')


    #ax.plot(df['T'], df['dB'], marker='*')
    

def setupPSDPlot():
    '''Set up a plot with Peterson noise model for plotting PSD curves.
       x axis = period (s)
       y axis = decibels '''
    codedir = '/Users/ewolin/code/HighFreqNoiseMustang'
    choosedir = '/Users/ewolin/research/NewNewNoise/ChosenNoise'
    nhnm = np.loadtxt(codedir+'/peterson_HNM.mod', unpack=True)
    nlnm = np.loadtxt(codedir+'/peterson_LNM.mod', unpack=True)
    nhnb = np.loadtxt(choosedir+'/GS_90_T-vs-DB.txt', unpack=True)
    nlportb = np.loadtxt(choosedir+'/Gt200_1_T-vs-dB.txt', unpack=True)
    nlpermb = np.loadtxt(choosedir+'/Perm_1_T-vs-dB.txt', unpack=True)

#    fig = plt.figure(figsize=(7,6.14))
#    fig = plt.figure(figsize=(4,3.5))
#    fig = plt.figure(figsize=(4.5,4))
    width=5
    fig = plt.figure(figsize=(width,width/1.14))
    gs_plots = gridspec.GridSpec(1,2, width_ratios=[1, 0.05])
    ax = fig.add_subplot(gs_plots[0,0])
    ax_cb = fig.add_subplot(gs_plots[0,1])

#    colorlist = ['gold','#ff7f0e','#d62678']
#    colorlist = [(0.97254902,  0.90588235,  0.52156863),
#                 (0.92941176,  0.62352941,  0.62352941),
#                 (0.43921569,  0.58039216,  0.90588235)]
#    colorlist = ['#f23c50', '#ffcb05', '#4ad9d9']
# 0.16470588,0.6,0.21176471 green
# 0.1372549 ,  0.40784314,  0.60784314 velocity blue
# 0.86666667,  0.13333333,  0.18039216 race red
# 0.99215686,  0.54509804,  0.22352941 orange fury
#    colorlist = [(0.16470588,0.6,0.21176471),
#                 (0.99215686,  0.54509804,  0.22352941),
#                 (0.86666667,  0.13333333,  0.18039216),
#                 (0.1372549 ,  0.40784314,  0.60784314)]
# purple #9467bd
# light blue #17becf


#    colorlist = ['#2ca02c', '#ff7f0e', '#d62728']
    colorlist = ['gold', '#ff7f0e', '#d62728']
    ax.set_prop_cycle(cycler('color', colorlist))
    
    print('from setuppsd:', fig.axes)
   # fig, ax = plt.subplots(figsize=(8,6)) 
    ax.plot(nhnm[0], nhnm[1], linewidth=2, color='black', label='NHNM/NLNM')
    ax.plot(nlnm[0], nlnm[1], linewidth=2, color='black')
    #ax.plot(nhnb[0], nhnb[1], linewidth=2, ls=':', color='black')
    #ax.plot(nlportb[0], nlportb[1], linewidth=3, ls='--', color='grey', label='Low $\geq$200 sps Baseline')
    ax.plot(nlportb[0], nlportb[1], linewidth=3, ls='--', color='grey', label='Low Portable Baseline')
    #ax.plot(nlpermb[0], nlpermb[1], linewidth=2, ls='-', color='black')
    ax.semilogx()
    ax.set_xlim(0.05, 200)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel(r'Power (dB[m$^2$/s$^4$/Hz])')
    ax.set_ylim(-200, -50)

# experimental...piecewise linear fit to baselines
# low perm:
#    ax.plot([0.02,0.110485],[-160,-160], color='grey', ls='-.', lw=2)
#    ax.plot([0.110485,0.340784],[-160,-166.7], color='grey', ls='-.', lw=2)
    ax.plot([0.02,0.110485,0.340784], [-160, -160, -166.7], color='grey', ls='-.', lw=3, label='Low Permanent Baseline')
# high:
    #ax.plot([0.1, 0.01], [-91.5, -91.5], color='grey', ls=':', lw=2)
 #   ax.plot([0.01, 0.03], [-90, -90], color='grey', ls=':', lw=2)
 #   ax.plot([0.03, 0.04], [-90, -88], color='grey', ls=':', lw=2)
 #   ax.plot([0.04, 0.065], [-88, -88], color='grey', ls=':', lw=2)
#    ax.plot([0.065, 0.1], [-88, -91.5], color='grey', ls=':', lw=2)

    ax.plot([0.01, 0.03, 0.04, 0.065, 0.1], [-90, -90, -88, -88, -91.5], color='grey', ls=(0,(1,1)), lw=3, label='High Baseline')

#    ax.plot([0.01, 0.110485, 0.340784], [-137.267, -160, -166.7], color='red', ls='-', lw=1)

    
#    carlpts = [(0.07,-167.0),(0.08,-168.0),(0.09,-169.0),(0.10,-169.5),(0.11,-170.5),(0.13,-171.0),(0.14,-171.5),(0.17,-172.0),(0.20,-172.5),(0.25,-173.0),(0.30,-173.5),(0.40,-173.0),(0.50,-172.0),(0.60,-171.0),(0.70,-170.0),(0.80,-169.2)]
#    for pt in carlpts:
#        ax.plot(*pt, color='red', marker='o', ls='None')

    plotBrune(ax)

    print('end of setuppsd:', fig.axes)
    return fig, ax


def statsFromPDF(noisefile):
    '''Given a PDF PSD from IRIS noise-pdf web service, calculate stats.
       Writes file in PQLX .mod format
       (currently only mean and median)
       Assumes noisefile is named following NSLC (eg MM.MDY..HHZ)'''
    freqs, dbs, hits = np.loadtxt(noisefile, skiprows=5, unpack=True, delimiter=',')
    freqlist = np.unique(freqs)
    mean = np.zeros(len(freqlist))
    median = np.zeros(len(freqlist))
# Insert dashes in location for compatibility with GLASS format
    #modfile = open('{0}.mod'.format(noisefile),'w')
    nslc = noisefile.split('.')
    print(len(nslc))
    modfile = open('{0}.{1}.--.{2}.mod'.format(nslc[0], nslc[1],nslc[3]),'w')
    mode = np.zeros(len(freqlist))
    perc90 = np.zeros(len(freqlist))
    perc50 = np.zeros(len(freqlist))
    perc10 = np.zeros(len(freqlist))
    perc2 = np.zeros(len(freqlist))
    n = np.zeros(len(freqlist))
    n_cml = np.zeros(len(freqlist))
    for i, freq in enumerate(freqlist):
        print(i,freq)
        indices, = np.where(freqs==freq)
        dbslice = dbs[indices]
        hitslice = hits[indices]
# we don't need to do this, use numpy.average instead
#        for j in indices:
#            mean[i] += dbs[j]*hits[j]
#            n[i] += hits[j]
#        mean[i] = mean[i]/n[i]
#        print('mean', mean[i], np.average(dbslice, weights=hitslice))
        mean[i] = np.average(dbslice, weights=hitslice)
        n[i] = sum(hitslice)
#        print('n=',n[i])
        # Find mode (highest # of hits for a given freq)
#        highhit = np.max(hits[indices])
#        print(highhit)
#        db_subset = dbs[indices]
#        hit_subset = hits[indices]
#        print(db_subset)
#        k_mode, = np.where(hit_subset == highhit)
#        mode[i] = db_subset[k_mode[0]] 
# expand histogram into array with one entry per value 
        db_expand = np.empty(int(sum(hitslice)))
        m = 0
        for j,val in enumerate(dbslice):
            for k in range(hitslice[j].astype(int)):
                db_expand[m] = val
                m += 1
        mode[i] = stats.mode(db_expand).mode[0] 
        perc90[i] = np.percentile(db_expand, 90)
        perc10[i] = np.percentile(db_expand, 10)
        perc2[i] = np.percentile(db_expand, 2)
        perc50[i] = np.percentile(db_expand, 50)
        median[i] = perc50[i]

       # tmp_hist = array
#        # Find bin containing median sample
#        if n[i] % 2 != 0:
#            k = (n[i]+1)/2
#        else:
#            k = n[i]/2
#        print(k)
#        n_cuml = 0
#        j_median = 0
#        j_before = 0
#        db_singlefreq = []
#        # This is a kludgey way to find the median, 
#        # must be a more elegant way?
#        for j in indices:
#            for nhits in range(int(hits[j])):
#                db_singlefreq.append(dbs[j])
#        median[i] = np.median(db_singlefreq)
#        print('***median: ',median[i],perc50[i])
        modfile.write('{0} {1} {2} {3} {4} {5}\n'.format(1./freqlist[i], perc10[i], perc90[i], mode[i], median[i], mean[i]))
#        print(freqlist[i], np.mean(db_singlefreq)-mean[i], median[i])
        print('------')
    modfile.close()
    return freqlist, mean, median, mode, perc10, perc50, perc90, perc2
