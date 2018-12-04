#!/usr/bin/env python
# pretty sure this will cover all dependencies:
# (make sure you are subscribed to conda-forge channel before trying to create env)
# conda create -n hfnoise python numpy requests pandas scipy obspy matplotlib
# then use
# conda activate hfnoise
# to activate it.

import os
import sys
import json
import requests
import argparse
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from obspy import UTCDateTime
from obspy.imaging.cm import pqlx

from noiseplot import setupPSDPlot

def readConfig(jsonfile):
    '''Parse JSON configuration file config.json in current working dir.
    Needs to contain:
    - workdir (can be '.' or path to a directory)
    Several wildcard-able text strings
    that will be plugged directly into irisws fedcat URL:
    - networks
    - stations
    - locations
    - channels
    A few other values for selecting data:
    - min_samp_rate : 
    - daily_perc_cutoff : goes into pct_below_nlnm request 
                          as value_gt parameter 
    - life_perc_cutoff : reject station if this % of days fall below NLNM
    Can also contain parameters for calculating, plotting, and fitting 
    percentiles to PDF:
    - plot_percs: list of percentiles you want to plot. 
    ex: "plot_percs" : [1, 2, 50, 90]
    - fit_percs: list of percentiles to fit from fmin to fmax.  
    Specify like so:
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


def readIRISfedcat(fedcatfile):
    '''Read file returned by fedcat, clean up headers, add columns holding
    start/end y-m-d for MUSTANG requests.
    Specify dtypes explicitly so we know what we're working with later.
    Note that we also give column names including # and spaces 
    to match format returned by IRIS fedcat request!
    Then we clean them up using df.rename
    '''
    #wait a minute...could we not just specify names upon reading?
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

    # Check for junk row with datacenter comment and delete it
    if df.iloc[0].Network[0] == '#':
        df = df.drop(0).reset_index(drop=True)

    # Replace NaN Locations with a blank character
    df = df.replace(pd.np.nan, '')

    # Make a new column w/NSCLQ target for MUSTANG
    df['Target'] = df[['Network', 'Station', 'Location', 'Channel']].apply(lambda row: '.'.join(row.values.astype(str))+'.M', axis=1)

    # Add columns holding start/end y-m-d for use with MUSTANG requests.
    # Round end time up to start of next day since MUSTANG noise-pdf works 
    # day by day (cannot specify end hr/min/sec).
    # Store StartDate and EndDate as strings for easy use in building 
    # MUSTANG URLs.
    # Use 'today' as stopdate for most up-to-date data.
    # For reproducibility, we specify a specific end date.  
    # NOTE: pandas datetime overflows for dates far in future (eg 2599 etc)
    # so we have to work around this
#    stopdate = pd.to_datetime('today')
    stopdate = pd.to_datetime('2018-10-12')
    stopstring = stopdate.strftime('%Y-%m-%dT%H:%M:%S')

    starttimes = pd.to_datetime(df.StartTime)

    # Using argument errors = 'coerce' makes overflow values 
    # (2500, 2599, 2999 etc) get set to pd.NaT 
    # Then we can convert, and replace all future dates w/stopdate.
    endtimes = pd.to_datetime(df.EndTime, errors='coerce') 
    endtimes[endtimes > stopdate ] = stopdate
    endtimes = endtimes.replace(pd.NaT, stopdate)

    df['StartDate'] = starttimes.dt.strftime('%Y-%m-%d')
    df['EndDate'] = endtimes.dt.ceil('D').dt.strftime('%Y-%m-%d')

    df['TotalTime'] = (endtimes - starttimes).dt.total_seconds()

    return df


def getIRISfedcat(config, outname):
    '''Get list of desired station-epochs from IRIS fedcat web service.
    Use networks, stations, locations, and channels from config file to
    build fedcat request.
    Then select only those where SampleRate > min_samp_rate specified in
    config file.
    '''
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


def checkAboveLNM(df, config, args, getplots=False, 
                  outpbndir='PctBelowNLNM'): 
    '''For each Target and Start/EndTime in the dataframe,
    request pct_below_nlnm metric from IRIS MUSTANG
    and check that the # of days below a given percentage do not exceed
    the specified % of the station lifetime.
    df should be a pandas dataframe 
    with AT MINIMUM the following columns:
    Target (string, N.S.L.C.Q)
    StartDate (string, YYYY-MM-DD)
    EndDate (string, YYYY-MM-DD)
    '''
    useindex = []
    notusedfile = open('notused.txt', 'w')
    usedfile = open('used.txt', 'w')
    for i in df.index: 
        outpbnname = outpbndir+'/{0}_{1}_{2}.txt'.format(df.Target[i],df.StartDate[i],df.EndDate[i])
        if args.force_lnm_check or (not os.path.exists(outpbnname)):
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
        if getplots:
            reqbase2 = 'http://service.iris.edu/mustang/noise-pdf/1/query?format=plot&nodata=404'
            reqstring2 = reqbase2+'&target={0}&starttime={1}&endtime={2}'.format(df.Target[i],df.StartDate[i],df.EndDate[i])
            res2 = requests.get(reqstring2)
            imgname = '_'.join([df.Target[i], df.StartDate[i], df.EndDate[i]])+'.png'
            imgfile = open(imgname, 'wb') 
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
        # and only returns days where pct_below_nlnm > value_gt 
        # (or 0 if value_gt == 0) 
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
            usedfile.write('{0} {1} {2}\n'.format(df.Target[i], 
                           df.StartDate[i], df.EndDate[i]))
            useindex.append(i)
            termcolor=32
        else:
            notusedfile.write('{0} {1} {2}\n'.format(df.Target[i], 
                              df.StartDate[i], df.EndDate[i]))
            termcolor=31

        # fancy color prompt :)
        print('\033[1;{3};40m {0} {4} {5}: % days in lifetime >{1}% below LNM: {2:.2f}\033[0;37;40m '.format(df.Target[i], config['daily_perc_cutoff'], lifetime_pct_below, termcolor, df.StartDate[i], df.EndDate[i]))

    usedfile.close()
    notusedfile.close()
    np.save('useindices.npy', useindex)
    df_selected = df.loc[useindex]
    df_selected.to_csv('irisfedcat_selected.txt', index=False, sep='|')
    return df_selected


def requestPDFs(df, args, outpdfdir):
    '''Request PDF PSDs from MUSTANG.  
    Checks to see if PDF file for stn epoch already exists in outpdfdir. 
    If so, will not re-request unless forced.
    '''
    df_successful = df.copy()
    errorfile = open(outpdfdir+'/error.log', 'w')
    for i in df.index: 
        outname = outpdfdir+'/{0}_{1}_{2}.txt'.format(df.Target[i], df.StartDate[i], df.EndDate[i])
        if args.force_get_PDFs or (not os.path.exists(outname)):
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


def listPDFFiles(df, outpdfdir):
    '''Make a list of all the PDF PSD files to read 
    based on contents of a dataframe.
    '''
    pdffiles = []
    for i in df.index:
        pdffile = outpdfdir+'/'+df.Target[i]+'_'+df.StartDate[i]+'_'+df.EndDate[i]+'.txt'
        pdffiles.append(pdffile)
    return pdffiles


def findPDFBounds(pdffiles):
    '''Get lists of unique frequencies and dBs
    from all individual PDF files
    '''
    print('finding list of unique freq, dB')
    df_single = pd.read_csv(pdffiles[0], skiprows=5, 
                            names=['freq', 'db', 'hits'])
    freq_u = df_single.freq.unique()
    db_u = df_single.db.unique()
    for i in range(1,len(pdffiles)):
        df_single =  pd.read_csv(pdffiles[i], skiprows=5, 
                                 names=['freq', 'db', 'hits'])
        freq_u = np.unique(np.append(freq_u, df_single.freq.unique()))
        db_u = np.unique(np.append(db_u, df_single.db.unique()))
    db_u.sort()
    freq_u.sort()
    np.save('freq_u.npy', freq_u)
    np.save('db_u.npy', db_u)
    return freq_u, db_u


def findUniqFreq(pdffiles):
    '''Find unique frequency values as *strings* 
    for quick lookup in calcMegaPDF.
    ''' 
    df_single = pd.read_csv(pdffiles[0], skiprows=5, 
                            names=['freq', 'db', 'hits'], 
                            dtype={'freq':'str'})
    freq_u = df_single.freq.unique()
    for i in range(1,len(pdffiles)):
        df_single =  pd.read_csv(pdffiles[i], skiprows=5, 
                                 names=['freq', 'db', 'hits'], 
                                 dtype={'freq':'str'})
        freq_u = np.unique(np.append(freq_u, df_single.freq.unique()))
    outfile = open('freq_u_str.txt', 'w')
    for i in range(len(freq_u)):
        outfile.write('{0}\n'.format(freq_u[i]))
    outfile.close()
    isort = np.argsort(freq_u.astype('float'))
    freq_u = freq_u[isort]
    np.save('freq_u_str.npy', freq_u)
    return freq_u


def calcMegaPDF(freq_u, freq_u_str, db_u, pdffiles, 
                outpdffile='megapdf.npy'):
    '''Add together all PSDPDFs in pdffiles!
    And save as .npy file for easier reading later
    '''
    # Set up dictionaries to convert freq and db to integers.
    # Use integers for freq to avoid floating point errors
    # and make binning faster.
    i_f = np.arange(len(freq_u_str))
    fd = dict(zip(freq_u_str, i_f))

    i_db = np.arange(len(db_u))
    dbd = dict(zip(db_u, i_db))
    pdf = np.zeros((len(i_f), len(i_db)), dtype=np.int_)
    
    # Sum all files to make mega-pdf
    print('Adding individual PDFs to composite, please wait...')
    logfile = open('pdffiles.txt','w')
    for infile in pdffiles:
        logfile.write('{0}\n'.format(infile.split('/')[-1]))
        freq = np.loadtxt(infile, unpack=True, delimiter=',', usecols=0)
        db, hits = np.loadtxt(infile, unpack=True, delimiter=',', 
                              usecols=[1,2], dtype=np.int_)
        for i in range(len(hits)):
            f1 = freq[i]
            db1 = db[i]
            hit1 = hits[i]
        
            i_f1 = fd[str(f1)]
            i_db1 = dbd[db1]
            pdf[i_f1, i_db1] += hit1
    logfile.close()

    # Save PDF to a numpy file so we can read+plot it easily later
    np.save(outpdffile, pdf)
    outpdftext = open('megapdf.txt', 'w')
    outpdftext.write('#freq db hits\n')
    for i_f in range(len(freq_u)):
        for i_db in range(len(db_u)):
            outpdftext.write('{0} {1} {2}\n'.format(freq_u[i_f], db_u[i_db],
                             pdf[i_f,i_db]))
    outpdftext.close()
    print('Finished calculating composite PDF.') 
    print('See pdffiles.txt for list of individual PDFs summed.')
            
    return pdf


def find_percentile(freq_u, db_u, newpdf_norm, perc, ax, fnyqs=[], 
                    plotline=True):
    '''Given a (normalized) PDF, find the dB levels of a given percentile.
    Ignore frequencies between 0.75*fnyq and fnyq to cut out spikes.
    '''
    # surely there must be something in numpy or scipy that does this 
    # but I haven't hunted it down yet.
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
    # plot and/or write percentile line 
    # ignoring spikes near Nyquist frequency(ies)
    i_use, = np.where(freq_u < 200)
    freq_u = freq_u[i_use]
    db_perc = db_perc[i_use]
    i_use, = np.where(freq_u < 100) 
    for fnyq in fnyqs:
        i_nonyq, = np.where((freq_u > fnyq) | (freq_u <0.75*fnyq))
        i_use = np.intersect1d(i_use, i_nonyq)
    freq_perc = freq_u[i_use]
    db_perc = db_perc[i_use]
    if plotline:
        ax.plot(1./freq_perc, db_perc, 
                label='{0:.1f}%'.format(100*perc), linewidth=2)
    outname = 'Percentiles/percentile_{0:.1f}.txt'.format(100*perc)
    outfile = open(outname,'w')
    outfile.write('#freq dB\n')
    for i in range(len(db_perc)):
        outfile.write('{0} {1}\n'.format(freq_perc[i], db_perc[i]))
    outfile.close()
    print('Wrote {0} percentile to {1}'.format(100*perc, outname))
    return freq_perc, db_perc


def linregressHighFreqs(f, db, fnyqs, perc_name, ax, f_min=3, f_max=100, 
                        plotline=False):
    '''Fit a line to a given PDF percentile between f_max and f_min.
    Ignore frequencies between 0.75*fnyq and fnyq to cut out spikes
    '''
    iclip, = np.where((f >= f_min) & (f <= f_max))
    i_use = iclip
    for i in range(len(fnyqs)):
        i_nonyq, = np.where((f>fnyqs[i]) | (f<0.75*fnyqs[i]))
        #print(i_nonyq)
        #print(fnyqs[i],np.intersect1d(i_use, i_nonyq))
        i_use = np.intersect1d(i_use, i_nonyq)
    x = f[i_use]
    y = db[i_use]
    # convert to period and take log10 so we can do a linear fit 
    x_log = np.log10(1./x)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log,y)
    print('slope, intercept of best fit line:', slope, intercept)
    y_new = x_log*slope+intercept
    if plotline:
        ax.plot(1./x, y_new, lw=1, color='black', ls=':')

    # Extend linear fit to 1.25 Hz (0.8 s) 
    # (for matching portable 1st percentile to NLNM)
    log_extend = np.linspace(np.log10(1.25), 2, 8)
    f_extend = 10**log_extend
    y_extend = np.log10(1./f_extend)*slope+intercept
    outfile = open('Percentiles/fit_{0}_perc.txt'.format(perc_name), 'w')
    outfile.write('T_s, dB\n')
    for i in range(len(x)):
        outfile.write('{0} {1}\n'.format(1./x[i], y_new[i]))
    outfile.close()
    outfile = open('Percentiles/fit_{0}_extend_perc.txt'.format(perc_name), 'w')
    outfile.write('T_s, dB\n')
    for i in range(len(f_extend)):
        outfile.write('{0} {1}\n'.format(1./f_extend[i], y_extend[i]))
    outfile.close()

    return slope, intercept


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
        args.force_lnm_check = True
        args.get_PDFs = True
        args.force_get_PDFs = True
        args.calc_PDF = True
        args.plot_PDF = True
    if args.force_lnm_check:
        args.lnm_check = True
    if args.force_get_PDFs:
        args.get_PDFs = True

    config = readConfig('config.json')

    # Create output directories if they don't exist
    workdir = config['workdir']
    outpdfdir = workdir+'/IndividualPDFs'
    outpercdir = workdir+'/Percentiles' 
    outpbndir = workdir+'/PctBelowNLNM'
    for outdir in [workdir, outpdfdir, outpercdir, outpbndir]:
        if not os.path.exists(outdir):
            os.mkdir(outdir)

    # Request or read list of stations to use in PDF calculation
    if args.request_stns:
        outname = 'irisfedcat_initial.txt'
        df = getIRISfedcat(config, outname)
    else:
        try:
            df = readIRISfedcat(args.read_stns)
        except ValueError as e:
            print('ERROR: no list of stations provided. Use --request_stns to query IRIS, or use --read_stns <file.txt> to supply a file in iris fedcatalog format')
            sys.exit(1)
        except FileNotFoundError as e:
            print('ERROR: could not find file {0}'.format(args.read_stns))
            sys.exit(1)

    # Check if stations are above Peterson NLNM
    if args.lnm_check:
        df_selected = checkAboveLNM(df, config, args)
    else:
        df_selected = df.copy()

    # Request PDFs from IRIS
    if args.get_PDFs:
        df_selected = requestPDFs(df_selected, args, outpdfdir)

    # Get list of existing PDFs for individual stations
    pdffiles = listPDFFiles(df_selected, outpdfdir)

    # Sum all individual station PDFs to produce composite pdf
    # or read results from a previous calculation
    if args.calc_PDF:
        freq_u, db_u = findPDFBounds(pdffiles)
        freq_u_str = findUniqFreq(pdffiles)
        pdf = calcMegaPDF(freq_u, freq_u_str, db_u, pdffiles)
    elif args.plot_PDF:
        try:
            freq_u = np.load('freq_u.npy')
            freq_u_str = np.load('freq_u_str.npy')
            db_u = np.load('db_u.npy')
            pdf = np.load('megapdf.npy')
        except FileNotFoundError as e:
            print(e)
            print('Could not find an input file needed for PDF plotting.')
            print('Run again with --calc_PDF option.')
            sys.exit(1)
    # Plot composite PDF
    if args.plot_PDF:
        cmap = 'gray_r'
        vmax=0.05
        fig, ax = setupPSDPlot()
    
        # Normalize PDF since MUSTANG returns hit counts not %
        newpdf_norm = np.zeros(shape=pdf.shape, dtype=np.float_)
        for i in range(len(freq_u)):
            if np.sum(pdf[i,:]) > 0:
                newpdf_norm[i,:] = pdf[i,:]/np.sum(pdf[i,:])
            else:
                newpdf_norm[i,:] = pdf[i,:]*0 
        outpdfnorm = open('megapdf_norm.txt', 'w')
        outpdfnorm.write('#freq db hits \n')
        for i_f in range(len(freq_u)):
            for i_db in range(len(db_u)):
                outpdfnorm.write('{0} {1} {2}\n'.format(freq_u[i_f], 
                                  db_u[i_db], newpdf_norm[i_f, i_db]))
        outpdfnorm.close()

        # Plot normalized PDF!
        im = ax.pcolormesh(1./freq_u, db_u, newpdf_norm.T*100, cmap=cmap, 
                           vmax=vmax*100) 
        fig.colorbar(im, cax=fig.axes[1], label='Probability (%)')
    

        # Calculate and plot specified percentiles
        fnyqs = 0.5*df.SampleRate.unique()
        try:
            for perc in config['plot_percs']:
                frac = perc/100.
                freq_perc, db_perc = find_percentile(freq_u, db_u, 
                                                     newpdf_norm, frac, 
                                                     ax, plotline=True, 
                                                     fnyqs=fnyqs)
        except KeyError as e:
            print(e)
            print('no plot percentiles specified in config.json')

        # Compute a linear fit specified percentiles and plot line
        try:
            for perc in config['fit_percs']:
                frac = config['fit_percs'][perc]['perc']/100.
                f_min = config['fit_percs'][perc]['fmin']
                f_max = config['fit_percs'][perc]['fmax']
                freq_perc, db_perc = find_percentile(freq_u, db_u, 
                                                     newpdf_norm, frac, 
                                                     ax, plotline=False,
                                                     fnyqs=fnyqs)
                slope, intercept = linregressHighFreqs(freq_perc, db_perc, 
                                                       fnyqs, perc, ax, 
                                                       f_min=f_min, 
                                                       f_max=f_max,  
                                                       plotline=False)
        except KeyError:
            print('no fit percentiles specified in config.json')


        # Add frequency axis to plot
        ax_freq = ax.twiny()
        ax_freq.set_xlabel('Frequency (Hz)')
        ax_freq.semilogx()
        ax_freq.set_xlim(1/0.01, 1/10)
        ax_freq.xaxis.set_label_position('top')
        ax_freq.tick_params(axis='x', top=True, labeltop=True) 
        
        # Final setup: Draw legend and set period axis limits
        ax.legend(ncol=2, loc='lower center', fontsize='6', handlelength=3)
        ax.set_xlim(0.01,10)
        plt.tight_layout()

        # Save images of plots
        for imgtype in ['png', 'eps']:
            fig.savefig('pdf.'+imgtype)
            print('saved plot as pdf.{0}'.format(imgtype))

    print('Finished')

if __name__ == "__main__":
    main()
