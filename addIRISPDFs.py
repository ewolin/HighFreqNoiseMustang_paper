#!/usr/bin/env python
# NOTE: run getlistoffreqs.sh to make freqlist_uniq.txt and dblist_uniq.txt first!

import sys
import os
import numpy as np
from glob import glob
import requests

import pandas as pd
from io import StringIO

import json

from noiseplot import setupPSDPlot
from scipy import stats
from obspy.imaging.cm import pqlx
from obspy import UTCDateTime
import matplotlib.pyplot as plt

####
# To do:
# Read from config or supply as command line arg:
#     - make list and map of nets/stns?
####################################
def readConfig(jsonfile):
    '''Parse JSON configuration file, which should contain:
       - workdir (can be '.' or path to a directory that MUST ALREADY EXIST)
       Several wildcard-able text strings that will be plugged directly into irisws fedcat URL:
       - networks
       - stations
       - locations
       - channels
       And a few other values for selecting data:
       - min_samp_rate : 
       - daily_perc_cutoff : goes into pct_below_nlnm request as value_gt parameter 
       - life_perc_cutoff : reject station if this % of days fall below NLNM
    '''
    fh = open(jsonfile)
    raw = fh.read()
    try: 
        config = json.loads(raw)
        return config
    except:
        print("Error in JSON config file, please check formatting")
        sys.exit()
####################################

####################################
def readIRISfedcat(fedcatfile):
# read file returned by fedcat, clean up headers, add columns holding start/end y-m-d for MUSTANG requests
# specify dtypes explicitly so we know what we're working with later
# note that we have to give column names including spaces etc. 
# to match format returned by IRIS fedcat request!
# Then we clean them up using df.rename
    use_dtypes = {'#Network' : 'str',
                  ' Station ' : 'str',
                  ' Location ' : 'str',
                  ' Channel ' : 'str',
                  ' Latitude ' : 'np.float64',
                  ' Longitude ' : 'np.float64', 
                  ' Elevation ' : 'np.float64',
                  ' Depth ' : 'np.float64',
                  ' Azimuth ' : 'np.float64',
                  ' Dip ' : 'np.float64', 
                  ' SensorDescription ' : 'str',
                  ' Scale ' : 'np.float64',
                  ' ScaleFreq ' : 'np.float64',
                  ' ScaleUnits' : 'str',
                  ' SampleRate ' : 'int',
                  ' StartTime ' : 'str',
                  ' EndTime ' : 'str'}
    df = pd.read_csv(fedcatfile, sep='|', dtype={' Location ':'str'})
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df.rename(columns=lambda x: x.strip('#'), inplace=True)

# check for junk row with datacenter comment and delete it
    if df.iloc[0].Network[0] == '#':
        df = df.drop(0).reset_index(drop=True)

# Replace NaN Locations with a blank character
    df = df.replace(pd.np.nan, '')

# Make a new column w/NSCLQ target for MUSTANG
    df['Target'] = df[['Network', 'Station', 'Location', 'Channel']].apply(lambda row: '.'.join(row.values.astype(str))+'.M', axis=1)

# Add columns holding start/end y-m-d for use with MUSTANG requests
# Round end time up to start of next day since MUSTANG noise-pdf works by day
# Use 'today' as stopdate for most up-to-date data
# although for reproducibility, we probably want to specify a specific end date.  Set in a config file?
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
def getIRISfedcat(config, outname):
# use requests to get fedcat file...
# in bash: used
# curl "http://service.iris.edu/irisws/fedcatalog/1/query?net=*&cha=HHZ,CHZ,DHZ,EHZ&format=text&includeoverlaps=true&nodata=404&datacenter=IRISDMC" > irisfedcat.txt
# set NSLC target(s) and start/end dates in config file?
    #reqstring = 'http://service.iris.edu/irisws/fedcatalog/1/query?net=*&cha=*HZ&format=text&includeoverlaps=true&nodata=404&datacenter=IRISDMC'
    #reqstring = 'http://service.iris.edu/irisws/fedcatalog/1/query?net=II,IU,CU,IC,US&cha=*HZ&format=text&includeoverlaps=true&nodata=404&datacenter=IRISDMC'
    #reqstring = 'http://service.iris.edu/irisws/fedcatalog/1/query?net=GS&cha=*HZ&format=text&includeoverlaps=true&nodata=404&datacenter=IRISDMC'
    reqstring = 'http://service.iris.edu/irisws/fedcatalog/1/query?net={0}&sta={1}&loc={2}&cha={3}&format=text&includeoverlaps=true&nodata=404&datacenter=IRISDMC'.format(config['networks'],config['stations'], config['locations'], config['channels'])
    print('requesting list of stations')
    res = requests.get(reqstring)
    outfile = open(outname, 'w')
    outfile.write(res.text)
    outfile.close()

# parse returned text just like we'd read a CSV
    df = readIRISfedcat(StringIO(res.text))

# get only sample rates >= min_samp_rate, remember to reset indices!!
# to do: set sample rate in config file?
# to do: change this so we can specify 'selected' text file name!
    df = df[df.SampleRate >= config['min_samp_rate']].reset_index(drop=True)
    savecsv = 'irisfedcat_samprate_ge{0:d}.txt'.format(config['min_samp_rate'])
    df.to_csv(savecsv, sep='|', index=False)
    return df
####################################    

####################################
# TO DO: catch errors in requesting data and find a way to gracefully restart after errors instead of re-requesting everything
# to do: read config for daily % cutoff and lifetime % cutoff
def checkAboveLNM(df, config, getplots=False, outpbndir='PctBelowNLNM'): 
    '''For each Target and Start/EndTime in the dataframe,
       request pct_below_nlnm metric from IRIS MUSTANG
       and check that the # of days below a given percentange do not exceed
       the specified % of the station lifetime.
       df should be a pandas dataframe with AT MINIMUM the following columns:
       Target (string, N.S.L.C.Q)
       StartDate (string, YYYY-MM-DD)
       EndDate (string, YYYY-MM-DD)'''
    useindex = []
    notusedfile = open('notused.txt', 'w')
    usedfile = open('used.txt', 'w')
    for i in df.index: 
        outpbnname = outpbndir+'/{0}_{1}_{2}.txt'.format(df.Target[i],df.StartDate[i],df.EndDate[i])
        if not os.path.exists(outpbnname):
            reqbase = 'http://service.iris.edu/mustang/measurements/1/query?metric=pct_below_nlnm&format=text&nodata=404&orderby=start_asc'
            reqstring = reqbase+'&target={0}&value_gt={1}&start={2}&end={3}'.format(df.Target[i],config['daily_perc_cutoff'],df.StartDate[i],df.EndDate[i])
            res = requests.get(reqstring)
            print(reqstring)
            outpbnfile = open(outpbnname, 'w')
            outpbnfile.write(res.text)
            outpbnfile.close()
            text = res.text.split('\n')[2:-1]
        else:
            print('already requested pct_below_nlnm for {0}, will not re-request'.format(df.Target[i]))
            text = open(outpbnname, 'r').readlines()
            text = [ line.strip('\n') for line in text ][2:-1]

        # get pdf psd plots too for QC
        reqbase2 = 'http://service.iris.edu/mustang/noise-pdf/1/query?format=plot&nodata=404'
        reqstring2 = reqbase2+'&target={0}&starttime={1}&endtime={2}'.format(df.Target[i],df.StartDate[i],df.EndDate[i])
        print(reqstring2)
        if getplots:
            reqbase2 = 'http://service.iris.edu/mustang/noise-pdf/1/query?format=plot&nodata=404'
            reqstring2 = reqbase2+'&target={0}&starttime={1}&endtime={2}'.format(df.Target[i],df.StartDate[i],df.EndDate[i])
            res2 = requests.get(reqstring2)
            imgfile = open(df.Target[i]+'_'+df.StartDate[i]+'_'+df.EndDate[i]+'.png', 'wb')
            if res2.status_code == 200:
                for chunk in res2:
                    imgfile.write(chunk)
            else:
                print(res2.status_code)
                print(reqstring2)
            imgfile.close()

        below_startdates = []
        below_enddates = []
# pct_below_nlnm returns text w/2 header lines
# and only returns days where pct_below_nlnm > 0 
#        text = res.text.split('\n')[2:-1]
        text2 = [k.split(',') for k in text]
        if len(text2) > 0:
            for t in text2:
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
        if lifetime_pct_below <= config['life_perc_cutoff']:
            usedfile.write('{0} {1} {2}\n'.format(df.Target[i], df.StartTime[i], df.EndTime[i]))
            useindex.append(i)
            termcolor=32
        else:
            notusedfile.write('{0} {1} {2}\n'.format(df.Target[i], df.StartTime[i], df.EndTime[i]))
            termcolor=31

#        print(df.Target[i], '% days in lifetime >{0}% below LNM:'.format(value_gt), lifetime_pct_below)
        print('\033[1;{3};40m {0}: % days in lifetime >{1}% below LNM: {2:.2f}\033[0;37;40m '.format(df.Target[i], config['daily_perc_cutoff'], lifetime_pct_below, termcolor))

    usedfile.close()
    notusedfile.close()
    np.save('useindices.npy', useindex)
    df_selected = df.loc[useindex]
    df_selected.to_csv('irisfedcat_selected.txt', index=False, sep='|')
    return df_selected
####################################

####################################
def requestPDFs(df, outpdfdir):
# request PDF PSDs from MUSTANG
# but first check to see if PDF file of date/time range of interest already exists in outpdfdir
#    outpdfdir = '/Users/ewolin/research/NewNewNoise/Get200/TMP'
    for i in df.index: #range(len(df.Target)):
        print(i, df.Target[i])
        #outname = outpdfdir+'/{0}_{1}_{2}_gt_{3}.txt'.format(df.Target[i], df.StartDate[i], df.EndDate[i])
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

####################################
def listPDFFiles(df, outpdfdir):
# to do: check to see if all pdf files exist in outpdfdir, and if not, request missing ones?
#    outpdfdir = '/Users/ewolin/research/NewNewNoise/Get200/TMP'
    pdffiles = []
    for i in df.index: #range(len(df.Target)):
        pdffile = outpdfdir+'/'+df.Target[i]+'_'+df.StartDate[i]+'_'+df.EndDate[i]+'.txt'
# todo        if pdffile exists:
        pdffiles.append(pdffile)
# todo    else:
# todo        make slice of dataframe for that pdf file
# todo        requestPDF()
    return pdffiles
####################################

####################################
# TO DO: find freq bounds in Python!!
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

####################################
def find_percentile(freq_u, db_u, newpdf_norm, perc, ax):
#    db_perc = -999*np.ones(len(freq_u))
    nfreq = len(freq_u)
    db_perc = -999*np.ones(nfreq) 
    for i in range(nfreq):
        fslice = newpdf_norm[i,:]
        dum = 0
        for j in range(len(fslice)):
            dum += fslice[j]
            if(dum >= perc):
                db_perc[i] = db_u[j]
                break
    ax.plot(1./freq_u, db_perc, label='{0:.1f}%'.format(100*perc))
    return db_perc
####################################

####################################
# ^.v.^
#   o
# m___m
####################################
def main():
    args = sys.argv
    if len(sys.argv) > 1:
        if sys.argv[1] == 'recalc':
            recalc = True
    else:
        recalc = False

    config = readConfig('config.json')

# Create output directories if they don't exist
#    workdir = '/Users/ewolin/research/NewNewNoise/PermWithPython/' # to do : read this from config and/or default to '.'
#    workdir = '/Users/ewolin/research/NewNewNoise/GSwithPython/' # to do : read this from config and/or default to '.'
#    workdir = '.'

    workdir = config['workdir']
    outpdfdir = workdir+'/IndividualPDFs'
    outpercdir = workdir+'/Percentiles' # to do: supply as arg to checkAboveLNM, write percentile files to this dir
    outpbndir = workdir+'/PctBelowNLNM'
    for outdir in [outpdfdir, outpercdir, outpbndir]:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
            print('created', outdir)
        else:
            print('exists', outdir)


    outname = 'irisfedcat_initial.txt'
    df = getIRISfedcat(config, outname)
    
    resume_lnm_check = False
    if resume_lnm_check:
    # find last target+start/end time that we checked
        df_checked_already = pd.read_csv('used.txt', names=['Target', 'StartTime', 'EndTime'], sep=' ')
        lastnscl = df_checked_already.iloc[-1].Target
        laststart = df_checked_already.iloc[-1].StartTime
        lastend = df_checked_already.iloc[-1].EndTime
        i_restart = df[(df.Target == lastnscl) & (df.StartTime == laststart) & (df.EndTime == lastend)].index[0]
        print('restarting from', i_restart)
        df_restart = df.loc[i_restart:]
        df_selected = checkAboveLNM(df_restart, config)
        df_selected = df[:i_restart-1].append(df_selected)
    else:
        df_selected = checkAboveLNM(df, config)
        

    
# or if we've already run checkAboveLNM to produce a 'selected' file (or we've made it some other way): 
    #df_selected = pd.read_csv('irisfedcat_selected.txt', sep='|')
    #df_selected = pd.read_csv('irisfedcat_selected_gt200.txt', sep='|')
    #df_selected = readIRISfedcat('irisfedcat_selected_gt250.txt')

    
    requestPDFs(df_selected, outpdfdir)

    pdffiles = listPDFFiles(df_selected, outpdfdir)

# Find min/max freqs and dBs and sum PDFs    
    freq_u, db_u = findPDFBounds()
    if recalc:
        pdf = calcMegaPDF(freq_u, db_u, pdffiles)
    else:
        pdf = np.load('megapdf.npy')

    
    
    # Plot PDF
    cmap = pqlx
    fig, ax = setupPSDPlot()
    
    newpdf_norm = np.zeros(shape=pdf.shape, dtype=np.float_)
# normalize PDF since MUSTANG returns hit counts not %
    for i in range(len(freq_u)):
    #    print(freq_u[i], np.sum(pdf[i,:]))
        if np.sum(pdf[i,:]) > 0:
            newpdf_norm[i,:] = pdf[i,:]/np.sum(pdf[i,:])
        else:
            newpdf_norm[i,:] = pdf[i,:]*1e-10 # np.nan*pdf[i,:]
    
    im = ax.pcolormesh(1./freq_u, db_u, newpdf_norm.T, cmap=cmap, vmax=.30)
    #ax.set_ylim(-367, 278)
    
# Plot PDF and save    
    ax.plot([0.01, 0.3], [-137, -162], 'k--', label='200 sps noise model')
    ax.plot([0.01, 0.731139], [-137,-168.536389686], 'c:', lw=5, label='200 sps noise model')
    ax.plot([0.01, 1], [-137,-170.849624626], 'y:', lw=2, label='200 sps noise model')
    dlogperiod = np.log10(0.3) - np.log10(0.01)
    ddb = -137 - -162
    y = -137 + ddb/dlogperiod * (np.log10(0.01)-np.log10(0.73))
    #print(y)
    #ax.plot(1, y, 'ko')

    ax.plot([0.01, 0.1], [-91, -91], 'r--', label='GS high noise model?')
    
    ax.grid()
    
    find_percentile(freq_u, db_u, newpdf_norm, 0.01, ax)
    y_full = find_percentile(freq_u, db_u, newpdf_norm, 0.02, ax)
    iwhere, = np.where((freq_u >= 3)&(freq_u<=40))
    x = freq_u[iwhere]
    y = y_full[iwhere]
    find_percentile(freq_u, db_u, newpdf_norm, 0.1, ax)
    find_percentile(freq_u, db_u, newpdf_norm, 0.5, ax)
    find_percentile(freq_u, db_u, newpdf_norm, 0.9, ax)
    #find_percentile(newpdf_norm, 1.0)

# need to get db, log(f or T) out of percentiles and do a linear fit
    print(x)
    x_log = np.log10(1./x)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log,y)
    y_new = x_log*slope+intercept
    plt.plot(1./x, y_new, lw=5, color='pink')
    print(min(1./x), max(1./x), min(y), max(y))
    plt.plot(1./x, np.ones(len(x))*-180, lw=10, color='purple')
   # for i,x in enumerate(x):
   #     print(x[i], x[i]*slope+intercept)
    
    ax.legend(ncol=3, loc='lower center', fontsize='small')
    ax.set_xlim(0.005,10)
    #ax.set_xlim(0.01,10)
    fig.savefig('meh.png')

if __name__ == "__main__":
    main()
