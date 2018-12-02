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

## Overview of code

### Request or read list of stations

irisws fedcatalog service returns a file with these headers:
```
#Network | Station | Location | Channel | Latitude | Longitude | Elevation | Depth | Azimuth | Dip | SensorDescription | Scale | ScaleFreq | ScaleUnits | SampleRate | StartTime | EndTime
```
The script will output other files named irisfedcat_* with slightly cleaner headers:
```
Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime|Target|StartDate|EndDate|TotalTime
```

### Check pct\_below\_nlnm metric
Write to directory PctBelowNLNM

### Get PDFs
Write to directory IndividualPDFs

### Calculate composite PDF
Save as megapdf.npy

### Plot composite PDF and selected percentiles
Save percentiles in directory Percentiles
Save plot as pdf.*

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
