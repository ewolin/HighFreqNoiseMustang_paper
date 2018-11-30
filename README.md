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
(Run in the appropriate directory so you have the correct config file as well)
Each subfigure a-d has its own directory:
GS | PERM | GE200 | TA
Each folder contains a file called config.json and irisfedcat_[GS|PERM|GE200|TA].txt.

```bash
addIRISPDFs.py --read_stns irisfedcat_[GS|PERM|GE200|TA].txt --lnm_check --get_PDFs --calc_PDFs --plot_PDF
```

## Using the code on your own
To download data for your own network(s)/station(s)/channel(s) etc. of interest, you can:

(1) use the [irisws-fedcatalog](http://service.iris.edu/irisws/fedcatalog/1/) web service to download a pipe-separated list of stations you are interested in, edit as desired, and supply the name of this text file as an argument to the --read_stns argument;

(2) copy and edit config.json, and let the script fetch the list of stations for you.
