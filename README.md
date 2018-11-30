# HighFreqNoiseMustang_paper
Download power spectral density probability functions (PDFPSDs) from stations of interest from IRIS MUSTANG, sum, calculate statistics, and plot

## Setting up your Python environment
Dependencies: numpy, scipy, obspy, requests, pandas, matplotlib.

I use [Anaconda](https://www.anaconda.com/download/) to manage dependencies.  You can create a new conda environment called ```hfnoise``` with all the necessary dependencies like so:
```bash
conda config --add channels conda-forge
conda create -n hfnoise python numpy requests pandas scipy obspy matplotlib
conda activate hfnoise
```

## Running the code
Plots in Figure 1 can be reproduced by running the code as follows:

```bash
addIRISPDFs.py --read_stns irisfedcat_[GS|PERM|GE200|TA].txt --lnm_check --get_PDFs --calc_PDFs --plot_PDF
```
