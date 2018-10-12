#!/usr/bin/env python
# NOTE: run getlistoffreqs.sh to make freqlist_uniq.txt and dblist_uniq.txt first!

import sys
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

workdir = '/Users/ewolin/research/NewNewNoise/Get200/GetEachPDF'


# Read text file returned from IRIS' fedcat service to build list of channel on/off dates
# consider using requests to get the fedcat file, and maybe process directly with pandas, instead of requesting w/a separate shell script?
infile = workdir+'/irisfedcat_HZ_gt200.txt'
df = readIRISfedcat(infile)
useindex = []

def readIRISfedcat(fedcatfile):
# read file returned by fedcat, clean up headers, add columns holding start/end y-m-d for MUSTANG requests
    df = pd.read_csv(fedcatfile, sep='|')
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df.rename(columns=lambda x: x.strip('#'), inplace=True)

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

# replace 2599 dates BEFORE converting to datetime to avoid errors
# then convert, and replace all future dates w/stopdate
    endtimes = pd.to_datetime(df.EndTime.replace(to_replace='2599-12-31T23:59:59', value=stopstring)) 
    endtimes[endtimes > stopdate ] = stopdate

    df['StartDate'] = starttimes.dt.strftime('%Y-%m-%d')
    df['EndDate'] = endtimes.dt.ceil('D').dt.strftime('%Y-%m-%d')

    df['TotalTime'] = (endtimes - starttimes).dt.total_seconds()

    return df



#def checkAboveLNM(targets, channel_startdates, channel_enddates):
def checkAboveLNM(df): 
    pdffiles = []
    notusedfile = open('notused.txt', 'w')
    usedfile = open('used.txt', 'w')
    for i in range(len(df.Target)):
# for each channel, request pct_below_nlnm metric
        starttime = UTCDateTime(df.StartTime[i])
        endtime = UTCDateTime(df.EndTime[i])
        channel_totaltime = endtime - starttime 
#        startstring = starttime.date.strftime('%Y-%m-%d')
# add one day to endstring, bc endtimes often are stored as 23:59:59 so we want to round up
#        endstring = (endtime+86400).date.strftime('%Y-%m-%d') 
#        print(startstring, endstring)
        reqbase = 'http://service.iris.edu/mustang/measurements/1/query?metric=pct_below_nlnm&format=text&nodata=404&orderby=start_asc'
        reqstring = reqbase+'&target={0}&value_gt=10&start={1}&end={2}'.format(df.Target[i],df.StartDate[i],df.EndDate[i])
#        res = requests.get('http://service.iris.edu/mustang/measurements/1/query?metric=pct_below_nlnm&target={0}&format=text&value_gt=10&orderby=start_asc&nodata=404&start={1}&end={2}'.format(df.Target[i], startstring,endstring))
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
            lifetime_pct_below = total_below_time/channel_totaltime*100.
        else: 
            lifetime_pct_below = 0.
        if lifetime_pct_below <= 10:
            pdffiles.append(workdir+'/'+df.Target[i]+'_'+df.StartTime+'_'+df.EndTime+'.txt') # STOPPED 11 Oct 2018: I know this is going to break the workflow so let's fix it tomorrow: need to add a section to this script that requests & names PDF files according to this convention.  Currently assumes we've already requested PDF files, but why bother requesting the ones we don't want?
# although maybe I should make some kind of option to not re-request a file if it already exists on disk
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
#    return pdffiles
    return df_selected

# write out list of all channels used...

#pdffiles = checkAboveLNM(targets, channel_startdates, channel_enddates)
#pdffiles = checkAboveLNM(df)
#df_selected = checkAboveLNM(df)

# or if we've already run checkAboveLNM to produce a 'selected' file (or we've made it some other way): 
#df_selected = pd.read_csv('irisfedcat_selected.txt', sep='|')
#df_selected = pd.read_csv('irisfedcat_selected_gt200.txt', sep='|')
df_selected = pd.read_csv('irisfedcat_selected_gt250.txt', sep='|')
df_selected.rename(columns=lambda x: x.strip(), inplace=True)
df_selected.rename(columns=lambda x: x.strip('#'), inplace=True) 

def requestPDFs(df):
# to do: add check to see if PDF file of date/time range of interest already exists in outpdfdir?
    outpdfdir = '/Users/ewolin/research/NewNewNoise/Get200/TMP'
    for i in range(len(df.Target)):
        print(df.Target[i])
        starttime = UTCDateTime(df.StartTime[i])
        endtime = UTCDateTime(df.EndTime[i])
        startstring = starttime.date.strftime('%Y-%m-%d')
# add one day to endstring, bc endtimes often are stored as 23:59:59 so we want to round up
        endstring = (endtime+86400).date.strftime('%Y-%m-%d') 
        reqbase = 'http://service.iris.edu/mustang/noise-pdf/1/query?format=text&nodata=404'
        reqstring = reqbase + '&target={0}&starttime={1}&endtime={2}'.format(df.Target[i], startstring, endstring)
#        print(reqstring)
        res = requests.get(reqstring)
#        df2 = pd.read_csv(StringIO(res.text), skiprows=4)
        outname = outpdfdir+'/{0}_{1}_{2}.txt'.format(df.Target[i], startstring, endstring)
        outfile = open(outname, 'w')
        outfile.write(res.text)
        outfile.close()

#requestPDFs(df_selected)
#sys.exit()


def listPDFFiles(df):
# to do: check to see if all pdf files exist in outpdfdir, and if not, request missing ones?
    outpdfdir = '/Users/ewolin/research/NewNewNoise/Get200/TMP'
    pdffiles = []
    for i in range(len(df.Target)):
        pdffile = outpdfdir+'/'+df.Target[i]+'_'+df.StartDate+'_'+df.EndDate+'.txt'
# todo        if pdffile exists:
        pdffiles.append(pdffile)
# todo    else:
# todo        make slice of dataframe for that pdf file
# todo        requestPDF()
    return pdffiles

pdffiles = listPDFFiles(df_selected)

#print(pdffiles)

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
