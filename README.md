# HighFreqNoiseMustang_paper
Download power spectral density probability functions (PDFPSDs) from stations of interest from IRIS MUSTANG.

Sum PSDPDFs from individual stations, calculate statistics, and plot.

## Setting up your Python environment
Dependencies: numpy, scipy, obspy, requests, pandas, matplotlib.

I use [Anaconda](https://www.anaconda.com/download/) to manage dependencies.  You can create a new conda environment called ```hfnoise``` with all the necessary dependencies like so:
```bash
conda config --add channels conda-forge
conda create -n hfnoise python numpy requests pandas scipy obspy matplotlib
conda activate hfnoise
```

## Reproducing plots in paper
Plots in Figure 1 can be reproduced by running the code as follows:

Each subfigure a-d has its own directory:

GS | PERM | PORT | TA

Each folder contains a file called config.json and irisfedcat_[GS|PERM|PORT|TA].txt.

```bash
addIRISPDFs.py --read_stns irisfedcat_[GS|PERM|PORT|TA].txt --lnm_check --get_PDFs --calc_PDFs --plot_PDF
```

## Using the code on your own
To download data for your own network(s)/station(s)/channel(s) etc. of interest, you can either:

(a) copy and edit config.json, then run 
```bash
addIRISPDFs.py --doall
```
The script will fetch a list of stations for you and then do all of the other processing.

(b) use the [irisws-fedcatalog](http://service.iris.edu/irisws/fedcatalog/1/) web service to download a pipe-separated list of stations you are interested in, edit if desired, and supply the name of this text file as an argument to the --read_stns argument:
```bash
addIRISPDFs.py --read_stns myfedcatfile.txt --lnm_check --get_PDFs --calc_PDF --plot_PDF
```

Once you have calculated the composite PDF you can re-plot it with
```bash
addIRISPDFs.py --read_stns myfedcatfile.txt --plot_PDF
```
(useful if you just want to tweak plotting parameters)

### Simple example
A quick way to test the script and config file is to run it for a single station.  The config.json file in Examples/SingleStation will fetch and calculate a single PDF for SPREE station XI.SN54.  (This was one of my favorite stations to visit, located on a beautiful sand prairie in Wisconsin.) 
```bash
cd Examples/SingleStation
addIRISPDF.py --doall
```

## Sample config.json file
plot_percs and fit_percs are optional; all other keys must be defined.
```json
{
	"workdir" : ".",
	"networks" : "*",
	"stations" : "*",
	"locations" : "*",
	"channels" : "*HZ",
	"min_samp_rate" : 200,
	"daily_perc_cutoff" : 10.0,
	"life_perc_cutoff" : 10.0,
	"plot_percs" : [1, 50, 90],
	"fit_percs" : {
		"perc1": {
		"perc" : 1.0,
		"fmin" : 5,
		"fmax" : 100
		}}
}
```

## Processing steps

### Request or read list of stations
The IRIS [fedcatalog](http://service.iris.edu/irisws/fedcatalog/1/) web service can be used to obtain a list of stations matching various criteria.
It returns a file with the following headers:
```
#Network | Station | Location | Channel | Latitude | Longitude | Elevation | Depth | Azimuth | Dip | SensorDescription | Scale | ScaleFreq | ScaleUnits | SampleRate | StartTime | EndTime
```
The script will read a fedcatalog file into a pandas dataframe, clean up the header names, select only stations with sample rates >= min_samp_rate, and add a few columns for use later in the script.  Other irisfedcat_*txt files written out by the script will have these headers: 
```
Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime|Target|StartDate|EndDate|TotalTime
```
You can also supply your own list of stations in fedcatalog format. 

### Check pct\_below\_nlnm metric
Request MUSTANG's [pct\_below\_nlnm](http://service.iris.edu/mustang/metrics/docs/1/desc/pct_below_nlnm/) metric and reject stations with more than life_perc_cutoff percent of days exceeding daily_perc_cutoff pct\_below\_nlnm.  (Set cutoffs in config.json.)  Ex: Reject stations for which more than 10% days in their lifetime have more than 10% of points below the NLNM.  Write pct\_below\_nlnm results for individual stations to directory PctBelowNLNM.  Write irisfedcat_passLNMcheck.txt to workdir.

### Get PDFs
Request PSDPDFs from MUSTANG's [noise-pdf](http://service.iris.edu/mustang/noise-pdf/1/) metric.  Write to directory IndividualPDFs.  Write irisfedcat_PDFs-exist.txt to workdir.

### Calculate composite PDF
Sum all specified individual PDFs. Save composite pdf as megapdf.npy (really a histogram, with number of counts in each bin) and megapdf\_norm.npy (an actual PDF, where all values at a given frequency sum to 1).  Also save list of freq and dB as freq\_u.npy and db\_u.npy.

If you want to read and plot these output files later: 
```python
import numpy as np
from noiseplot import setupPSDPlot

freq_u = np.load('freq_u.npy')
db_u = np.load('db_u.npy')
pdf = np.load('megapdf.npy') # PDF in terms of bin counts
pdf_norm = np.load('megapdf.npy') # PDF in percent

# PDFs are indexed as pdf[freq, dB]
# so you can get the histogram slice at the lowest frequency with pdf[0,:]
# handy trick to calculate # of PSDs:
sum(pdf[0,:])

# To plot with setupPSDPlot you must transpose the pdf:
# (in this example we also multiply by 100 so we get probability in percent)
fig, ax = setupPSDPlot()
im = ax.pcolormesh(1./freq_u, db_u, newpdf_norm.T*100, cmap='gray_r', 
                   vmax=0.05*100)
fig.colorbar(im, cax=fig.axes[1], label='Probability (%)')
```

### Plot composite PDF and selected percentiles
Compute percentiles from composite PDF.  
Fit a linear trend to percentiles in a given range, if specified in config.json.  
Save percentiles and linear fits in directory Percentiles.  
Save plot of composite PDF, percentiles, as pdf.[png,eps]


