import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

    #fig, ax = plt.subplots(figsize=(6.5,6)) 
    fig, ax = plt.subplots(figsize=(8,6)) 
    ax.plot(nhnm[0], nhnm[1], linewidth=2, color='black')
    ax.plot(nlnm[0], nlnm[1], linewidth=2, color='black')
    #ax.plot(nhnb[0], nhnb[1], linewidth=2, ls=':', color='black')
    ax.plot(nlportb[0], nlportb[1], linewidth=2, ls='--', color='pink')
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
    ax.plot([0.02,0.110485,0.340784], [-160, -160, -166.7], color='grey', ls='-.', lw=2)
# high:
    #ax.plot([0.1, 0.01], [-91.5, -91.5], color='grey', ls=':', lw=2)
 #   ax.plot([0.01, 0.03], [-90, -90], color='grey', ls=':', lw=2)
 #   ax.plot([0.03, 0.04], [-90, -88], color='grey', ls=':', lw=2)
 #   ax.plot([0.04, 0.065], [-88, -88], color='grey', ls=':', lw=2)
#    ax.plot([0.065, 0.1], [-88, -91.5], color='grey', ls=':', lw=2)

    ax.plot([0.01, 0.03, 0.04, 0.065, 0.1], [-90, -90, -88, -88, -91.5], color='grey', ls=':', lw=2)

    ax.plot([0.01, 0.110485, 0.340784], [-137.267, -160, -166.7], color='red', ls='-', lw=1)

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
