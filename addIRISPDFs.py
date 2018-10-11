#!/usr/bin/env python
# NOTE: run getlistoffreqs.sh to make freqlist_uniq.txt and dblist_uniq.txt first!

import sys
import numpy as np
from glob import glob
import requests


from noiseplot import setupPSDPlot
from obspy.imaging.cm import pqlx
from obspy import UTCDateTime
import matplotlib.pyplot as plt

args = sys.argv
if len(sys.argv) > 1:
    if sys.argv[1] == 'recalc':
        recalc = True
else:
    recalc = False

workdir = '/Users/ewolin/research/NewNewNoise/Get200/GetEachPDF'


# Read text file returned from IRIS' fedcat service to build list of channel on/off dates
# consider using requests to get this instead of a separate shell script?
infile = workdir+'/irisfedcat_HZ_gt200.txt'
chanfile = open(infile, 'r')
targets = []
nslcs = []
channel_startdates = []
channel_enddates = []
lines = chanfile.readlines()
for line in lines:
    l = line.strip().split('|')
    nslc = '.'.join(l[0:4])
    target = nslc+'.M'
    s1 = UTCDateTime(l[15])
    e1 = UTCDateTime(l[16])
    print(target, s1, e1)
    targets.append(target)
    nslcs.append(nslc)
    channel_startdates.append(s1)
    channel_enddates.append(e1)

def checkAboveLNM(targets, channel_startdates, channel_enddates):
    pdffiles = []
    notusedfile = open('notused.txt', 'w')
    usedfile = open('used.txt', 'w')
    for i in range(len(targets)):
# for each channel, request pct_below_nlnm metric
        channel_totaltime = channel_enddates[i] - channel_startdates[i]
        res = requests.get('http://service.iris.edu/mustang/measurements/1/query?metric=pct_below_nlnm&target={0}&format=text&value_gt=10&orderby=start_asc&nodata=404&start={1}&end={2}'.format(targets[i], channel_startdates[i].format_iris_web_service().split('T')[0],(channel_enddates[i]+1).format_iris_web_service().split('T')[0]))
        below_startdates = []
        below_enddates = []
#        print(res.text)
        text = res.text.split('\n')[2:-1]
        text2 = [i.split(',') for i in text]
#        print('len of text2', len(text2))
        if len(text2) > 0:
            for t in text2:
                #print(t)
                s1 = UTCDateTime(t[2].strip('"'))
                e1 = UTCDateTime(t[3].strip('"'))
                below_startdates.append(s1)
                below_enddates.append(e1)
            total_below_time = 0.
            for j in range(len(below_startdates)):
                dt = below_enddates[j] - below_startdates[j]
                total_below_time += dt
            lifetime_pct_below = total_below_time/channel_totaltime*100.
        else: 
            lifetime_pct_below = 0.
        if lifetime_pct_below <= 10:
            pdffiles.append(workdir+'/'+targets[i]+'.txt')
            usedfile.write('{0} {1} {2}\n'.format(targets[i], channel_startdates[i], channel_enddates[i]))
        else:
            notusedfile.write('{0} {1} {2}\n'.format(targets[i], channel_startdates[i], channel_enddates[i]))

        print(targets[i], '% days in lifetime >10% below LNM:', lifetime_pct_below)


    usedfile.close()
    notusedfile.close()
    return pdffiles

# write out list of all channels used...

pdffiles = checkAboveLNM(targets, channel_startdates, channel_enddates)
# or if we don't want to wait for re-requested data
# last request: don't use anything that's >10% below LNM for more than 20% of lifetime
#usedfile = open('used.txt') 
#lines = usedfile.readlines()
#pdffiles = [ workdir+'/'+i.split()[0]+'.txt' for i in lines ]
#usedfile.close()

print(pdffiles)

def findPDFBounds():
    # Get lists of unique frequencies and dbs
    freqfile = '/Users/ewolin/research/NewNewNoise/Get200/GetEachPDF/freqs_uniq.txt'
    freq_u = np.loadtxt(freqfile, unpack=True)

    dbfile = '/Users/ewolin/research/NewNewNoise/Get200/GetEachPDF/dbs_uniq.txt'
    db_u = np.loadtxt(dbfile, unpack=True)
    return freq_u, db_u
    
def calcMegaPDF(freq_u, db_u, pdffiles, outpdffile='megapdf.npy'):
    # Set up dictionaries to convert freq and db to integers
    freq_s = [str(f) for f in freq_u ]
    i_f = np.arange(len(freq_u))
    fd = dict(zip(freq_s, i_f))

    i_db = np.arange(len(db_u))
    dbd = dict(zip(db_u, i_db))
    pdf = np.zeros((len(i_f), len(i_db)), dtype=np.int_)
    
    # Make mega-pdf
    for infile in pdffiles:
        print(infile.split('/')[-1])
    # Read input file
        try:
            freq = np.loadtxt(infile, unpack=True, delimiter=',', usecols=0)
            db, hits = np.loadtxt(infile, unpack=True, delimiter=',', usecols=[1,2], dtype=np.int_)
                
            for i in range(len(hits)):
                f1 = freq[i]
                db1 = db[i]
                hit1 = hits[i]
            
                i_f1 = fd[str(f1)]
                i_db1 = dbd[db1]
                pdf[i_f1, i_db1] += hit1
        except:
            pass
    # Save PDF to a numpy file so we can plot it easily later
    np.save(outpdffile, pdf)
    return pdf

freq_u, db_u = findPDFBounds()
if recalc:
    pdf = calcMegaPDF(freq_u, db_u, pdffiles)
else:
    pdf = np.load('megapdf.npy')

# Plot PDF
cmap = pqlx
fig, ax = setupPSDPlot()

newpdf_norm = np.zeros(shape=pdf.shape, dtype=np.float_)
for i in range(len(freq_u)):
    print(freq_u[i], np.sum(pdf[i,:]))
    if np.sum(pdf[i,:]) > 0:
        newpdf_norm[i,:] = pdf[i,:]/np.sum(pdf[i,:])
    else:
        newpdf_norm[i,:] = pdf[i,:]*1e-10 # np.nan*pdf[i,:]

im = ax.pcolormesh(1./freq_u, db_u, newpdf_norm.T, cmap=cmap, vmax=.30)
#ax.set_ylim(-367, 278)

def find_percentile(perc):
    db_perc = -999*np.ones(len(freq_u))
    for i in range(len(freq_u)):
        fslice = newpdf_norm[i,:]
        dum = 0
        for j in range(len(fslice)):
            dum += fslice[j]
            if(dum >= perc):
                db_perc[i] = db_u[j]
#                print(dum)
                break
#        print(freq_u[i], db_perc[i])
    ax.plot(1./freq_u, db_perc, label='{0:.1f}%'.format(100*perc))

ax.plot([0.01, 0.3], [-137, -162], 'k--', label='200 sps noise model')
ax.plot([0.01, 0.731139], [-137,-168.536389686], 'c:', lw=5, label='200 sps noise model')
ax.plot([0.01, 1], [-137,-170.849624626], 'y:', lw=2, label='200 sps noise model')
dlogperiod = np.log10(0.3) - np.log10(0.01)
ddb = -137 - -162
y = -137 + ddb/dlogperiod * (np.log10(0.01)-np.log10(0.73))
print(y)
#ax.plot(1, y, 'ko')

ax.grid()

find_percentile(0.01)
find_percentile(0.02)
find_percentile(0.1)
find_percentile(0.5)
find_percentile(0.9)
#find_percentile(1.0)

ax.legend(ncol=3, loc='lower center', fontsize='small')
ax.set_xlim(0.005,10)
#ax.set_xlim(0.01,10)
fig.savefig('meh.png')
