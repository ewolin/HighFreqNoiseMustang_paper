#!/usr/bin/env python
# pretty sure this will cover all dependencies:
# conda create -n hfnoise python numpy requests pandas scipy obspy matplotlib
# (make sure you are subscribed to conda-forge channel first)

import sys
import os
import numpy as np
#from glob import glob
import requests

import pandas as pd
from io import StringIO

import json
import argparse

import functools

from noiseplot import setupPSDPlot
from scipy import stats
from obspy.imaging.cm import pqlx
from obspy import UTCDateTime
import matplotlib.pyplot as plt

####
# To do:
# Read from config or supply as command line arg:
#     - make list and map of nets/stns?
#     - tradeoff: download pct_below_nlnm with value_gt > 0 the first time, and re-evaluate for various lifetime %s, 
#                 *OR* request only for one value_gt, but would have to re-request if we decrease value_gt???
# put some control flow in main() so we can skip straight to plotting if desired!! 
####################################
def readConfig(jsonfile):
    '''Parse JSON configuration file, which should contain:
       - workdir (can be '.' or path to a directory)
       Several wildcard-able text strings that will be plugged directly into irisws fedcat URL:
       - networks
       - stations
       - locations
       - channels
       A few other values for selecting data:
       - min_samp_rate : 
       - daily_perc_cutoff : goes into pct_below_nlnm request as value_gt parameter 
       - life_perc_cutoff : reject station if this % of days fall below NLNM
       Can also contain parameters for calculating, plotting, and fitting percentiles to PDF:
       - plot_percs: list of percentiles you want to plot. ex: "plot_percs" : [1, 2, 50, 90]
       - fit_percs: list of percentiles to fit from fmin to fmax.  Specify like so:
       	"fit_percs" : { "perc2": 
                {
		"perc" : 2.0,
		"fmin" : 10,
		"fmax" : 50
		},
	        	"perc1": 
                { 
		"perc" : 1.0,
		"fmin" : 10,
		"fmax" : 50
		}
                     }
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
# note that we also give column names including # and spaces 
# to match format returned by IRIS fedcat request!
# Then we clean them up using df.rename
    use_dtypes = {'#Network' : 'str',
                  ' Station ' : 'str',
                  ' Location ' : 'str',
                  ' Channel ' : 'str',
                  ' Latitude ' : np.float64,
                  ' Longitude ' : np.float64, 
                  ' Elevation ' : np.float64,
                  ' Depth ' : np.float64,
                  ' Azimuth ' : np.float64,
                  ' Dip ' : np.float64, 
                  ' SensorDescription ' : 'str',
                  ' Scale ' : np.float64,
                  ' ScaleFreq ' : np.float64,
                  ' ScaleUnits ' : 'str',
                  ' SampleRate ' : np.float64,
                  ' StartTime ' : 'str',
                  ' EndTime ' : 'str',
                  'Network' : 'str',
                  'Station' : 'str',
                  'Location' : 'str',
                  'Channel' : 'str',
                  'Latitude' : np.float64,
                  'Longitude' : np.float64, 
                  'Elevation' : np.float64,
                  'Depth' : np.float64,
                  'Azimuth' : np.float64,
                  'Dip' : np.float64, 
                  'SensorDescription' : 'str',
                  'Scale' : np.float64,
                  'ScaleFreq' : np.float64,
                  'ScaleUnits' : 'str',
                  'SampleRate' : np.float64,
                  'StartTime' : 'str',
                  'EndTime' : 'str'}
    df = pd.read_csv(fedcatfile, sep='|', dtype=use_dtypes)
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
    '''Get list of desired station-epochs from IRIS fedcat web service.
       Use networks, stations, locations, and channels from config file to build fedcat request.
       Then select only those where SampleRate > min_samp_rate in config file.'''

    reqstring = 'http://service.iris.edu/irisws/fedcatalog/1/query?net={0}&sta={1}&loc={2}&cha={3}&format=text&includeoverlaps=true&nodata=404&datacenter=IRISDMC'.format(config['networks'],config['stations'], config['locations'], config['channels'])
    print('requesting list of stations')
    res = requests.get(reqstring)
    if res.status_code == 200:
        print('List of stations received from IRIS fedcat')
        outfile = open(outname, 'w')
        outfile.write(res.text)
        outfile.close()
        print('Wrote initial list to {0}'.format(outname))

# parse returned text just like we'd read a CSV
    df = readIRISfedcat(StringIO(res.text))

# get only sample rates >= min_samp_rate, remember to reset indices!!
    df = df[df.SampleRate >= config['min_samp_rate']].reset_index(drop=True)
    savecsv = 'irisfedcat_samprate_ge{0:d}.txt'.format(config['min_samp_rate'])
    df.to_csv(savecsv, sep='|', index=False)
    print('Wrote list of stns with sample rate >= {0} to {1}'.format(config['min_samp_rate'],savecsv))
    return df
####################################    

####################################
def checkAboveLNM(df, config, args, getplots=False, outpbndir='PctBelowNLNM'): 
    '''For each Target and Start/EndTime in the dataframe,
       request pct_below_nlnm metric from IRIS MUSTANG
       and check that the # of days below a given percentage do not exceed
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
        if args.force_get_PDFs or (not os.path.exists(outpbnname)):
            reqbase = 'http://service.iris.edu/mustang/measurements/1/query?metric=pct_below_nlnm&format=text&nodata=404&orderby=start_asc'
            reqstring = reqbase+'&target={0}&value_gt={1}&start={2}&end={3}'.format(df.Target[i],config['daily_perc_cutoff'],df.StartDate[i],df.EndDate[i])
            res = requests.get(reqstring)
            print(reqstring)
            outpbnfile = open(outpbnname, 'w')
            outpbnfile.write(res.text)
            outpbnfile.close()
            text = res.text.split('\n')[2:-1]
        else:
            print('Will not re-request file for pct_below_nlnm for {0}'.format(df.Target[i]))
            text = open(outpbnname, 'r').readlines()
            print(outpbnname, df.Target[i])
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
# and only returns days where pct_below_nlnm > value_gt (or 0 if value_gt == 0) 
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
            usedfile.write('{0} {1} {2}\n'.format(df.Target[i], df.StartDate[i], df.EndDate[i]))
            useindex.append(i)
            termcolor=32
        else:
            notusedfile.write('{0} {1} {2}\n'.format(df.Target[i], df.StartDate[i], df.EndDate[i]))
            termcolor=31

        # fancy color prompt :)
        print('\033[1;{3};40m {0} {4} {5}: % days in lifetime >{1}% below LNM: {2:.2f}\033[0;37;40m '.format(df.Target[i], config['daily_perc_cutoff'], lifetime_pct_below, termcolor, df.StartDate[i], df.EndDate[i]))

    usedfile.close()
    notusedfile.close()
    np.save('useindices.npy', useindex)
    df_selected = df.loc[useindex]
    df_selected.to_csv('irisfedcat_selected.txt', index=False, sep='|')
    return df_selected
####################################


####################################
def requestPDFs(df, outpdfdir):
    '''Request PDF PSDs from MUSTANG
       but first, check to see if PDF file for stn epoch already exists in outpdfdir
       If so, don't re-request unless forced'''
    df_successful = df.copy()
    errorfile = open(outpdfdir+'/error.log', 'w')
    for i in df.index: 
        outname = outpdfdir+'/{0}_{1}_{2}.txt'.format(df.Target[i], df.StartDate[i], df.EndDate[i])
        if not os.path.exists(outname):
            print('requesting PDF PSD for:', df.Target[i])
            reqbase = 'http://service.iris.edu/mustang/noise-pdf/1/query?format=text&nodata=404'
            reqstring = reqbase + '&target={0}&starttime={1}&endtime={2}'.format(df.Target[i], df.StartDate[i], df.EndDate[i])
            print(reqstring)
            res = requests.get(reqstring)
            if res.status_code == 200:
                outfile = open(outname, 'w')
                outfile.write(res.text)
                outfile.close()
            else: 
                print(res.status_code)
                errorfile.write('Error {0}: {1} {2} {3} \n'.format(res.status_code, df.Target[i], df.StartDate[i], df.EndDate[i]))
                errorfile.write(reqstring+'\n')
                errorfile.write('\n')
                df_successful = df_successful.drop(i)
        else:
            print('PDF PSD file exists, will not re-request for {0}'.format(df.Target[i]))
    df_successful.to_csv('irisfedcat_PDFs-exist.txt', index=False, sep='|')
    errorfile.close()
    return df_successful
####################################

####################################
def listPDFFiles(df, outpdfdir):
    '''Make a list of all the PDF PSD files to read based on contents of a dataframe '''
    pdffiles = []
    for i in df.index:
        pdffile = outpdfdir+'/'+df.Target[i]+'_'+df.StartDate[i]+'_'+df.EndDate[i]+'.txt'
        pdffiles.append(pdffile)
    return pdffiles
####################################

####################################
def findPDFBounds(pdffiles):
# need to find a way to write out unique frequency STRINGS as well for lookup in calcMegaPDF
    '''Get lists of unique frequencies and dBs
       from all individual PDF files'''
    print('finding list of unique freq, dB')
    df_single = pd.read_csv(pdffiles[0], skiprows=5, names=['freq', 'db', 'hits'])
    freq_u = df_single.freq.unique()
    db_u = df_single.db.unique()
    for i in range(1,len(pdffiles)):
        df_single =  pd.read_csv(pdffiles[i], skiprows=5, names=['freq', 'db', 'hits'])
        freq_u = np.unique(np.append(freq_u, df_single.freq.unique()))
        db_u = np.unique(np.append(db_u, df_single.db.unique()))
    np.save('freq_u.npy', freq_u)
    np.save('db_u.npy', db_u)
    return freq_u, db_u
####################################

####################################
def findUniqFreq(pdffiles):
    '''Find unique frequency values as *strings* for quick lookup in calcMegaPDF''' 
    df_single = pd.read_csv(pdffiles[0], skiprows=5, names=['freq', 'db', 'hits'], dtype={'freq':'str'})
    freq_u = df_single.freq.unique()
    for i in range(1,len(pdffiles)):
        df_single =  pd.read_csv(pdffiles[i], skiprows=5, names=['freq', 'db', 'hits'], dtype={'freq':'str'})
        freq_u = np.unique(np.append(freq_u, df_single.freq.unique()))
    outfile = open('freq_u_str.txt', 'w')
    for i in range(len(freq_u)):
        outfile.write('{0}\n'.format(freq_u[i]))
    outfile.close()
    isort = np.argsort(freq_u.astype('float'))
#    freq_u.sort()
    freq_u = freq_u[isort]
    np.save('freq_u_str.npy', freq_u)
    return freq_u
####################################
def calcMegaPDF(freq_u, freq_u_str, db_u, pdffiles, outpdffile='megapdf.npy'):
    '''Add together all PDF PSDs in pdffiles!
       And save as .npy file for easier reading later'''
    # Set up dictionaries to convert freq and db to integers
    i_f = np.arange(len(freq_u_str))
    fd = dict(zip(freq_u_str, i_f))
    print(fd)

    i_db = np.arange(len(db_u))
    dbd = dict(zip(db_u, i_db))
    pdf = np.zeros((len(i_f), len(i_db)), dtype=np.int_)
    
    # Make mega-pdf
    for infile in pdffiles:
        print('adding to PDF:',infile.split('/')[-1])
    # Read input file
        #try:
        freq = np.loadtxt(infile, unpack=True, delimiter=',', usecols=0)
        db, hits = np.loadtxt(infile, unpack=True, delimiter=',', usecols=[1,2], dtype=np.int_)
            
        for i in range(len(hits)):
            #print(freq[i], db[i], hits[i])
            f1 = freq[i]
            db1 = db[i]
            hit1 = hits[i]
        
            i_f1 = fd[str(f1)]
            i_db1 = dbd[db1]
            pdf[i_f1, i_db1] += hit1
        #except:
            #print('trouble adding {0} to pdf'.format(infile))
# tried using pandas to add PDF files but I think it's faster w/loadtxt (the way I did it originally)
#        df_pdf = pd.read_csv(infile, skiprows=5, names=['freq', 'db', 'hits'], dtype={'freq':'str', 'db':'int', 'hits':'int'})
#        for i in df_pdf.index:
#            i_f1 = fd[df_pdf.freq[i]]
#            i_db1 = dbd[df_pdf.db[i]]
#            pdf[i_f1, i_db1] += df_pdf.hits[i]
    # Save PDF to a numpy file so we can plot it easily later
    np.save(outpdffile, pdf)
    outpdftext = open('megapdf.txt', 'w')
    outpdftext.write('#freq db hits\n')
    for i_f in range(len(freq_u)):
        for i_db in range(len(db_u)):
            outpdftext.write('{0} {1} {2}\n'.format(freq_u[i_f], db_u[i_db], pdf[i_f, i_db]))
    outpdftext.close()
            
    return pdf
####################################

####################################
def find_percentile(freq_u, db_u, newpdf_norm, perc, ax, plotline=True):
    '''Given a (normalized) PDF, find the dB levels of a given percentile'''
# surely there must be something in numpy or scipy that does this but I haven't hunted it down yet.
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
    if plotline:
        ax.plot(1./freq_u, db_perc, label='{0:.1f}%'.format(100*perc))
    outname = 'Percentiles/percentile_{0:.1f}.txt'.format(100*perc)
    outfile = open(outname,'w')
    outfile.write('#freq dB\n')
    for i in range(len(db_perc)):
        outfile.write('{0} {1}\n'.format(freq_u[i], db_perc[i]))
    outfile.close()
    print('Wrote {0} percentile to {1}'.format(100*perc, outname))
    return db_perc
####################################

####################################
def linregressHighFreqs(f, db, fnyqs, ax, f_min=3, f_max=100):
    '''Fit a line to a given PDF percentile between f_max and f_min.
       Ignore frequencies from 0.75*fnyq and fnyq to cut out spikes'''
    iclip, = np.where((f >= f_min) & (f <= f_max))
    print(iclip)
    i_use = iclip
    for i in range(len(fnyqs)):
        i_nonyq, = np.where((f>fnyqs[i]) | (f<0.75*fnyqs[i]))
        print(i_nonyq)
        print(fnyqs[i],np.intersect1d(i_use, i_nonyq))
        i_use = np.intersect1d(i_use, i_nonyq)
    x = f[i_use]
    y = db[i_use]
    for i in range(len(x)):
        print(x[i], y[i])
    # convert to period and take log10 so we can do a linear fit and then easily plot with setupPSDPlot
    #ax.plot(1./x, y, 'ko')
    x_log = np.log10(1./x)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log,y)
    y_new = x_log*slope+intercept
    #ax.plot(1./x, y_new, lw=5, color='blue')

# extend a bit to plot between 0.001-0.1 s (10-100 Hz)
    T_extend = np.logspace(-2,np.log10(0.731139))
    y_extend = np.log10(T_extend)*slope+intercept
 #   ax.plot(T_extend, y_extend, lw=3)#, color='orange')
    
#    return x, y_new
    return slope, intercept
####################################


####################################
# ^.v.^
#   o
# m___m
####################################
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--doall", help="Restart from beginning: request stn-epochs and select by sample rate, check below NLNM, request PDFPSDs, sum, and plot", action='store_true')
    parser.add_argument("--request_stns", help="Request list of stn-epochs defined in config file from IRIS fedcat", action='store_true', default=False)
    parser.add_argument("--read_stns", help="Read list of stns from IRIS fedcat webservice (pipe-separated values).  If you do not use --request_stns then you MUST use this argument and supply a filename.")
    parser.add_argument("--lnm_check", help="Check that PDF PSD values don't drop daily_perc_cutoff below Peterson NLNM for life_perc_cutoff of lifetime", action='store_true', default=False)
    parser.add_argument("--force_lnm_check", help="Re-request pct_below_nlnm output from MUSTANG even if files already exist in PctBelowNLNM",
                        action='store_true', default=False)
    parser.add_argument("--get_PDFs", help="Request text PDF PSDs from MUSTANG for each stn-epoch", action='store_true', default=False)
    parser.add_argument("--force_get_PDFs", help="Re-request text PDF PSDs from MUSTANG even if files already exist in IndividualPDFs", action='store_true', default=False)
    parser.add_argument("--calc_PDF", help="Sum all PDF PSD files into one composite mega-PDF", action='store_true', default=False)
    parser.add_argument("--plot_PDF", help="Plot composite PDF", action='store_true', default=False)

    args = parser.parse_args()
    if args.doall:
        args.request_stns = True
        args.lnm_check = True
        args.get_PDFs = True
        args.calc_PDF = True
        args.plot_PDF = True

    config = readConfig('config.json')

# Create output directories if they don't exist
    workdir = config['workdir']
    outpdfdir = workdir+'/IndividualPDFs'
    outpercdir = workdir+'/Percentiles' # to do: supply as arg to checkAboveLNM, write percentile files to this dir
    outpbndir = workdir+'/PctBelowNLNM'
    for outdir in [workdir, outpdfdir, outpercdir, outpbndir]:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
            print('created', outdir)
        else:
            print('exists', outdir)


    if args.request_stns:
        outname = 'irisfedcat_initial.txt'
        df = getIRISfedcat(config, outname)
    else:
        df = readIRISfedcat(args.read_stns)

    if args.lnm_check:
        df_selected = checkAboveLNM(df, config, args)
    else:
        df_selected = df.copy()

    if args.get_PDFs:
        df_selected = requestPDFs(df_selected, outpdfdir)

    pdffiles = listPDFFiles(df_selected, outpdfdir)

# Find min/max freqs and dBs and sum PDFs    
    if args.calc_PDF:
        freq_u, db_u = findPDFBounds(pdffiles)
        freq_u_str = findUniqFreq(pdffiles)
        pdf = calcMegaPDF(freq_u, freq_u_str, db_u, pdffiles)
    elif args.plot_PDF:
        freq_u = np.load('freq_u.npy')
        freq_u_str = np.load('freq_u_str.npy')
        db_u = np.load('db_u.npy')
        pdf = np.load('megapdf.npy')

# Plot PDF
    if args.plot_PDF:
#        cmap = pqlx
#        vmax=0.30
        cmap = 'bone_r'
        vmax=0.05
        fig, ax = setupPSDPlot()
    
# Normalize PDF since MUSTANG returns hit counts not %
        newpdf_norm = np.zeros(shape=pdf.shape, dtype=np.float_)
        for i in range(len(freq_u)):
            if np.sum(pdf[i,:]) > 0:
                newpdf_norm[i,:] = pdf[i,:]/np.sum(pdf[i,:])
            else:
                newpdf_norm[i,:] = pdf[i,:]*1e-10 # np.nan*pdf[i,:]
    
        im = ax.pcolormesh(1./freq_u, db_u, newpdf_norm.T*100, cmap=cmap, vmax=vmax*100)
        fig.colorbar(im, ax=ax, label='Probability (%)')
    
# Plot PDF and save    
# vertical dashed line at 100 Hz
        ax.plot([0.01, 0.01], [-200, -50], ls='--', color='grey')
        #ax.plot([0.01, 0.3], [-137, -162], 'k--', label='200 sps noise model')
        #ax.plot([0.01, 0.731139], [-137,-168.536389686], 'c:', lw=3, label='200 sps noise model')
        #ax.plot([0.01, 1], [-137,-170.849624626], 'y:', lw=2, label='200 sps noise model')
        dlogperiod = np.log10(0.3) - np.log10(0.01)
        ddb = -137 - -162
        y = -137 + ddb/dlogperiod * (np.log10(0.01)-np.log10(0.73))
        #print(y)
        #ax.plot(1, y, 'ko')
        #ax.plot(0.794328, -169.170, 'ko')
    
        #ax.plot([0.01, 0.1], [-91, -91], 'r--', label='GS high noise model?')
    
        #ax.grid()

        fnyqs = 0.5*df.SampleRate.unique()

# Plot specified percentiles
        try:
            for perc in config['plot_percs']:
                frac = perc/100.
                db_perc = find_percentile(freq_u, db_u, newpdf_norm, frac, ax, plotline=True)
        except KeyError:
            print('no plot percentiles specified in config.json')
            print('ex: "plot_percs" : [1, 2, 50, 90]')

# Fit a line to specified percentiles and plot
        try:
            for perc in config['fit_percs']:
                frac = config['fit_percs'][perc]['perc']/100.
                f_min = config['fit_percs'][perc]['fmin']
                f_max = config['fit_percs'][perc]['fmax']
                db_perc = find_percentile(freq_u, db_u, newpdf_norm, frac, ax, plotline=False)
#                x, y = linregressHighFreqs(freq_u, db_perc, fnyqs, ax, f_min=f_min, f_max=f_max)
                slope, intercept = linregressHighFreqs(freq_u, db_perc, fnyqs, ax, f_min=f_min, f_max=f_max)
                x_write = np.logspace(-2, np.log10(0.794328), num=9)
                y_write = np.log10(x_write)*slope+intercept
                
                outfile = open('Percentiles/fit_{0}_perc.txt'.format(config['fit_percs'][perc]['perc']), 'w')
                for i in range(len(x_write)):
                    outfile.write('{0} {1}\n'.format(x_write[i], y_write[i]))
                outfile.close

        except KeyError:
            print('no fit percentiles specified in config.json')

        ax.legend(ncol=3, loc='lower center', fontsize='medium')
        #ax.set_xlim(0.005,10)
        ax.set_xlim(0.01,10)

        plt.tight_layout()
        fig.savefig('pdf.png')
        fig.savefig('pdf.eps')

if __name__ == "__main__":
    main()
