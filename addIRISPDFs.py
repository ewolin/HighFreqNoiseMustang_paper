#!/usr/bin/env python
# NOTE: run getlistoffreqs.sh to make freqlist_uniq.txt and dblist_uniq.txt first!

import sys
import os.path
import numpy as np
from glob import glob
import requests

import pandas as pd
from io import StringIO

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


####################################
def readIRISfedcat(fedcatfile):
# read file returned by fedcat, clean up headers, add columns holding start/end y-m-d for MUSTANG requests
    df = pd.read_csv(fedcatfile, sep='|')
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df.rename(columns=lambda x: x.strip('#'), inplace=True)

# check for junk row with datacenter comment and delete it
    if df.iloc[0].Network[0] == '#':
        df = df.drop(0).reset_index(drop=True)

# Replace NaN Locations with a blank character
    df = df.replace(pd.np.nan, '')

# Make a new column w/NSCLQ target for MUSTANG
    #df['Target'] = stns.Network+'.'+stns.Station+'.'+stns.Location+'.'+stns.Channel + '.M'
    df['Target'] = df[['Network', 'Station', 'Location', 'Channel']].apply(lambda row: '.'.join(row.values.astype(str))+'.M', axis=1)

# Add columns holding start/end y-m-d for use with MUSTANG requests
# Round end time up to start of next day since MUSTANG noise-pdf works by day
# Could use 'today' as stopdate for most up-to-date data, but will set to a fixed date for now for reproducibility
# NOTE: pandas datetime overflows for dates far in future (eg 2599 etc) so we have to work around this
# Store StartDate and EndDate as strings for easy use in building MUSTANG URLs.
#    stopdate = pd.to_datetime('today')
    stopdate = pd.to_datetime('2018-10-12')
    stopstring = stopdate.strftime('%Y-%m-%dT%H:%M:%S')

    starttimes = pd.to_datetime(df.StartTime)

# errors = 'coerce' makes overflow values (2500, 2599, 2999 etc) get set to pd.NaT 
# then convert, and replace all future dates w/stopdate
    endtimes = pd.to_datetime(df.EndTime, errors='coerce') 
    endtimes[endtimes > stopdate ] = stopdate
    endtimes = endtimes.replace(pd.NaT, stopdate)

    df['StartDate'] = starttimes.dt.strftime('%Y-%m-%d')
    df['EndDate'] = endtimes.dt.ceil('D').dt.strftime('%Y-%m-%d')

    df['TotalTime'] = (endtimes - starttimes).dt.total_seconds()

    return df
####################################


####################################
def getIRISfedcat():
# use requests to get fedcat file...
# in bash: used
# curl "http://service.iris.edu/irisws/fedcatalog/1/query?net=*&cha=HHZ,CHZ,DHZ,EHZ&format=text&includeoverlaps=true&nodata=404&datacenter=IRISDMC" > irisfedcat.txt
    reqstring = 'http://service.iris.edu/irisws/fedcatalog/1/query?net=*&cha=*HZ&format=text&includeoverlaps=true&nodata=404&datacenter=IRISDMC'
    print('requesting list of stations')
    res = requests.get(reqstring)
    outname = 'irisfedcat_allHZ.txt'
    outfile = open(outname, 'w')
    outfile.write(res.text)
    outfile.close()

# parse returned text just like we'd read a CSV
    df = readIRISfedcat(StringIO(res.text))

# get only sample rates >= 200 sps, remember to reset indices!!
    df = df[df.SampleRate >= 200].reset_index(drop=True)
    df.to_csv('irisfedcat_allHZ_ge200.txt', sep='|', index=False)
#    print(df.Target)
# surely we can do this with requests, StringIO, and pandas...
    return df
####################################    

workdir = '/Users/ewolin/research/NewNewNoise/Get200/GetEachPDF'


# Read text file returned from IRIS' fedcat service to build list of channel on/off dates
# consider using requests to get the fedcat file, and maybe process directly with pandas, instead of requesting w/a separate shell script?
#infile = workdir+'/irisfedcat_HZ_gt200.txt'
#infile = '/Users/ewolin/research/NewNewNoise/Get200/TMP/irisfedcat_HZ_gt200_small.txt'
#infile = '/Users/ewolin/research/NewNewNoise/Get200/TMP/savehere.csv'
#infile = '/Users/ewolin/research/NewNewNoise/Get200/TMP/irisfedcat_allHZ_ge200.txt'
#df = readIRISfedcat(infile)
#kdf = df[df.SampleRate >= 200]
df = getIRISfedcat()
#df = df[0:5]
useindex = []

####################################
def checkAboveLNM(df): 
    pdffiles = []
    notusedfile = open('notused.txt', 'w')
    usedfile = open('used.txt', 'w')
    for i in range(len(df.Target)):
    # for each channel, request pct_below_nlnm metric
        reqbase = 'http://service.iris.edu/mustang/measurements/1/query?metric=pct_below_nlnm&format=text&nodata=404&orderby=start_asc'
        reqstring = reqbase+'&target={0}&value_gt=10&start={1}&end={2}'.format(df.Target[i],df.StartDate[i],df.EndDate[i])
        res = requests.get(reqstring)
 #       print(reqstring)
        below_startdates = []
        below_enddates = []
        text = res.text.split('\n')[2:-1]
        text2 = [k.split(',') for k in text]
#        print(text2)
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
            lifetime_pct_below = (total_below_time/df.TotalTime[i])*100.
        else: 
            lifetime_pct_below = 0.
        if lifetime_pct_below <= 10:
            #pdffiles.append(workdir+'/'+df.Target[i]+'_'+df.StartTime+'_'+df.EndTime+'.txt') 
# maybe I should make some kind of option to not re-request a file if it already exists on disk
# and to deal w/not using certain PDF files if they don't match the list returned here...like if we've already downloaded the file but I decide to add an option to change the pct_below_nlnm value??
# might also be good to store a list of targets/times we want so can re-start w/o nlnm check 
            usedfile.write('{0} {1} {2}\n'.format(df.Target[i], df.StartTime[i], df.EndTime[i]))
            useindex.append(i)
        else:
            notusedfile.write('{0} {1} {2}\n'.format(df.Target[i], df.StartTime[i], df.EndTime[i]))

        print(df.Target[i], '% days in lifetime >10% below LNM:', lifetime_pct_below)

    usedfile.close()
    notusedfile.close()
    np.save('useindices.npy', useindex)
    df_selected = df.iloc[useindex]
    df_selected.to_csv('irisfedcat_selected.txt', index=False, sep='|')
    return df_selected
####################################

# write out list of all channels used...

#pdffiles = checkAboveLNM(targets, channel_startdates, channel_enddates)
#pdffiles = checkAboveLNM(df)
df_selected = checkAboveLNM(df)

# or if we've already run checkAboveLNM to produce a 'selected' file (or we've made it some other way): 
#df_selected = pd.read_csv('irisfedcat_selected.txt', sep='|')
#df_selected = pd.read_csv('irisfedcat_selected_gt200.txt', sep='|')
#df_selected = readIRISfedcat('irisfedcat_selected_gt250.txt')

####################################
def requestPDFs(df):
# request PDF PSDs from MUSTANG
# but first check to see if PDF file of date/time range of interest already exists in outpdfdir
    outpdfdir = '/Users/ewolin/research/NewNewNoise/Get200/TMP'
    for i in range(len(df.Target)):
        outname = outpdfdir+'/{0}_{1}_{2}.txt'.format(df.Target[i], df.StartDate[i], df.EndDate[i])
        if not os.path.exists(outname):
            print('requesting PDF PSD for:', df.Target[i])
            reqbase = 'http://service.iris.edu/mustang/noise-pdf/1/query?format=text&nodata=404'
            reqstring = reqbase + '&target={0}&starttime={1}&endtime={2}'.format(df.Target[i], df.StartDate[i], df.EndDate[i])
#        print(reqstring)
            res = requests.get(reqstring)
#        df2 = pd.read_csv(StringIO(res.text), skiprows=4)
            outfile = open(outname, 'w')
            outfile.write(res.text)
            outfile.close()
        else:
            print('PDF PSD file exists, will not re-request for {0}'.format(df.Target[i]))
####################################

requestPDFs(df_selected)
#sys.exit()


####################################
def listPDFFiles(df):
# to do: check to see if all pdf files exist in outpdfdir, and if not, request missing ones?
    outpdfdir = '/Users/ewolin/research/NewNewNoise/Get200/TMP'
    pdffiles = []
    for i in range(len(df.Target)):
        pdffile = outpdfdir+'/'+df.Target[i]+'_'+df.StartDate[i]+'_'+df.EndDate[i]+'.txt'
# todo        if pdffile exists:
        pdffiles.append(pdffile)
# todo    else:
# todo        make slice of dataframe for that pdf file
# todo        requestPDF()
    return pdffiles
####################################

pdffiles = listPDFFiles(df_selected)

#print(pdffiles)

####################################
def findPDFBounds():
    # Get lists of unique frequencies and dbs
    freqfile = '/Users/ewolin/research/NewNewNoise/Get200/GetEachPDF/freqs_uniq.txt'
    freq_u = np.loadtxt(freqfile, unpack=True)

    dbfile = '/Users/ewolin/research/NewNewNoise/Get200/GetEachPDF/dbs_uniq.txt'
    db_u = np.loadtxt(dbfile, unpack=True)
    return freq_u, db_u
####################################
    
####################################
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
        print('adding to PDF:',infile.split('/')[-1])
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
####################################

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
#    print(freq_u[i], np.sum(pdf[i,:]))
    if np.sum(pdf[i,:]) > 0:
        newpdf_norm[i,:] = pdf[i,:]/np.sum(pdf[i,:])
    else:
        newpdf_norm[i,:] = pdf[i,:]*1e-10 # np.nan*pdf[i,:]

im = ax.pcolormesh(1./freq_u, db_u, newpdf_norm.T, cmap=cmap, vmax=.30)
#ax.set_ylim(-367, 278)

####################################
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
####################################

ax.plot([0.01, 0.3], [-137, -162], 'k--', label='200 sps noise model')
ax.plot([0.01, 0.731139], [-137,-168.536389686], 'c:', lw=5, label='200 sps noise model')
ax.plot([0.01, 1], [-137,-170.849624626], 'y:', lw=2, label='200 sps noise model')
dlogperiod = np.log10(0.3) - np.log10(0.01)
ddb = -137 - -162
y = -137 + ddb/dlogperiod * (np.log10(0.01)-np.log10(0.73))
#print(y)
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
