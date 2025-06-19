import numpy as np 
import pandas as pd
import seaborn as sns
import scipy as scp
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import segyio
import matplotlib.pyplot as plt



def segy_inspect(filename):
    """
    Read a segy file and return the data, the trace headers and the bin headers
    """

    with segyio.open(filename) as segyfile:

        # Memory map file for faster reading (especially if file is big...)
        segyfile.mmap()

        # Print binary header info
        print(segyfile.bin)
        print(segyfile.bin[segyio.BinField.Traces])

        # Read headerword inline for trace 10
        #print(segyfile.header[10][segyio.TraceField.INLINE_3D])

        # Print inline and crossline axis
        print(segyfile.xlines)
        print(segyfile.ilines)



def plot_segy(data):#, traces, bin_headers,headers):
    """
    Plot the  STACKED segy data
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(data.T, cmap="seismic", aspect="auto")
    plt.title("SEGY Data")
    plt.show()
    return


def read_segy(segyfile):
    """
    Read a segy file and return the data, the trace headers and the bin headers
    """
    with segyio.open(segyfile, "r",ignore_geometry=True) as segyfile:
        data = segyfile.trace.raw[:]
        headers=segyfile.header
        text_headers = segyfile.text[0]
        bin_headers = segyfile.bin
    return data, text_headers, bin_headers, headers


def SEGY_extract_sbp(file):
    """
    READ SEGY FILE AND EXTRACT THE SBP DATA FROM IT
    
    INPUT: SEGY FILE
    
    OUTPUT: NUMBER OF TRACES, SAMPLE RATE, NUMBER OF SAMPLES, TWT
    """
    
    with segyio.open(file, ignore_geometry=True) as f:
        # Get basic attributes
        n_traces = f.tracecount
        sample_rate = segyio.tools.dt(f) / 1000
        n_samples = f.samples.size
        twt = f.samples
        #data = f.trace.raw[:]  # Get all data into memory (could cause on big files)
        record_length=(n_samples-1)*sample_rate
    return n_traces, sample_rate, n_samples, record_length,twt

#########################################################################################################