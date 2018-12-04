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

GS | PERM | GE200 | TA

Each folder contains a file called config.json and irisfedcat_[GS|PERM|GE200|TA].txt.

```bash
addIRISPDFs.py --read_stns irisfedcat_[GS|PERM|GE200|TA].txt --lnm_check --get_PDFs --calc_PDFs --plot_PDF
```

## Using the code on your own
To download data for your own network(s)/station(s)/channel(s) etc. of interest, you can either:

(a) use the [irisws-fedcatalog](http://service.iris.edu/irisws/fedcatalog/1/) web service to download a pipe-separated list of stations you are interested in, edit as desired, and supply the name of this text file as an argument to the --read_stns argument;

(b) copy and edit config.json, then run 
```bash
addIRISPDFs.py --doall
```
The script will fetch the list of stations for you and then do all of the other processing.

## Sample config.json file
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
The script will read a fedcatalog file into a pandas dataframe, clean up the header names, select only stations with sample rates >= min_samp_rate, and add a few columns for use later in the script.  Other irisfedcat_*txt will have these headers: 
```
Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime|Target|StartDate|EndDate|TotalTime
```
You can also supply your own list of stations in fedcatalog format. 

### Check pct\_below\_nlnm metric
Request MUSTANG's [pct\_below\_nlnm](http://service.iris.edu/mustang/metrics/docs/1/desc/pct_below_nlnm/) metric and flag stations with more than life_perc_cutoff percent of days exceeding daily_perc_cutoff pct\_below\_nlnm.  (Set cutoffs in config.json.).  Write pct\_below\_nlnm results for individual stations to directory PctBelowNLNM.  Write irisfedcat_passLNMcheck.txt to workdir.

### Get PDFs
Request PSDPDFs from MUSTANG's [noise-pdf](http://service.iris.edu/mustang/noise-pdf/1/) metric.  Write to directory IndividualPDFs.  Write irisfedcat_PDFs-exist.txt to workdir.

### Calculate composite PDF
Sum all individual PDFs (that passed the pct\_below\_nlnm check, if applied)
 and save as megapdf.npy.

### Plot composite PDF and selected percentiles
Compute percentiles from composite PDF.  
Fit a linear trend to percentiles in a given range, if specified in config.json.  
Save percentiles and linear fits in directory Percentiles.  
Save plot of composite PDF, percentiles, as pdf.[png,eps]


