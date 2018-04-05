#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 14:34:11 2018

@author: woelfle
"""

# %% Import modules as needed

import numpy as np  # for handling arrays
# import pandas as pd  # for handling 2d things
import xarray as xr  # for handling nd things (netcdfs)

# from scipy import interpolate    # interpolation functions

import matplotlib.pyplot as plt  # for plotting things
import matplotlib.gridspec as gridspec  # for subplot management

from socket import gethostname   # used to determine which machine we are
#                                #   running on

import multiprocessing as mp  # Allow use of multiple cores
import datetime  # For keeping track of run times

from mdwtools import mdwfunctions as mwfn  # For averaging things
from mdwtools import mdwplots as mwp  # For plotting things

import cesm1to2plotter as c1to2p

import os  # operating system things.
# import matplotlib.cm as cm
# from scipy.stats import linregress

# %% Define functions as needed


def setfilepaths():
    """
    Set host specific variables and filepaths

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2017-10-17

    Args:
        N/A

    Returns:
        ncDir - directory in which netcdf case directories are stored
        ncSubDir - directory within case directory to search for netcdfs
        saveDir - directory to which figures will be saved

    Notes:
        fullPathForHistoryFileDirectory = (ncDir + fullCaseName +
                                           os.sep + ncSubDir)
    """

    if gethostname() in ['stable', 'challenger', 'p', 'fog']:
        ncDir = '/home/disk/eos9/woelfle/cesm/nobackup/cesm1to2/'
        ncSubDir = '0.9x1.25/'
        saveDir = ('/home/disk/user_www/woelfle/cesm1to2/')

    elif gethostname() == 'woelfle-laptop':
        ncDir = 'C:\\Users\\woelfle\\Documents\\UW\\CESM\\hist\\'
        ncSubDir = ''
        saveDir = 'C:\\Users\\woelfle\\Documents\\UW\\CESM\\figs\\'

    elif gethostname()[0:6] in ['cheyen', 'geyser']:
        ncDir = '/glade/p/cesmLE/CESM-CAM5-BGC-LE/'
        ncSubDir = ''
        saveDir = '/glade/p/work/woelfle/figs/cesm1to2/LENS/'

    return (ncDir, ncSubDir, saveDir)

# %% Main section
if __name__ == '__main__':

    # Set variables to compute at runtime
    newVars = ['PRECT']

    # Set flags for loading/plotting/saving
    ncAtmSubDir = 'atm/proc/tseries/monthly/'
    lenMean = 10  # Length to use when computing variability chunks
    ocnOnly_flag = False  # True to only use ocean points
    prect_flag = True  # True to compute PRECT
    save_flag = True  # True to save figures
    saveSubDir = '' # subdirectory for saving figures
   
    # Set more details for period to be loaded
    fileBase = 'b.e11.B1850C5CN.f09_g16.005.cam.h0.'
    startYrs = np.arange(1600, 1601, 100)
    verbose_flag = False

    # Get directory of file(s) to load 
    ncDir, ncSubDir, saveDir = setfilepaths()

    loadPeriods = ['{:04.0f}00-{:04.0f}12'.format(yr1, yr1+99)
                   for yr1 in startYrs]

    # Open the datasets
    if any([prect_flag, 'PRECT' in newVars]):
        # Create full paths to files to be loaded
        loadFiles = [ncDir + ncAtmSubDir + varName + '/' +
                     fileBase + varName +
                     '.{:04.0f}01-'.format(yid) +
                     '{:04.0f}12.nc'.format(yid + 99)
                     for yid in startYrs
                     for varName in ['PRECC', 'PRECL']
                     ]

        # Load precipitaiton fields to be used for computing PRECT
        dataSet = xr.open_mfdataset(loadFiles,
                                    decode_times=False)

        # Compute PRECT
        dataSet['PRECT'] = mwfn.calcprectda(dataSet)

    # %% Make plots

    indexName = 'dITCZ'

    rmAnnMean_flag = False

    # Make plot of double ITCZ index through time
    if indexName in ['dITCZ']:
        ds = dataSet
        plotObs_flag = False
        plotVar = 'PRECT'
        ocnOnly_flag = False
        title = 'Double-ITCZ Index'
        yLim = np.array([0, 6])
        yLim_annMean = np.array([0, 3])

    # Compute given index through time
    indexDa = c1to2p.calcregmeanindex(ds,
                                      indexName,
                                      indexType=None,
                                      indexVar=plotVar,
                                      ocnOnly_flag=ocnOnly_flag,
                                      )

    # Create figure for plotting
    hf = plt.figure()
    hf.set_size_inches(12, 5)

    # Pull regional mean through time and plot
    pData = (indexDa.values - indexDa.mean(dim='time').values
             if rmAnnMean_flag
             else indexDa.values)
    hl, = plt.plot(np.arange(0, len(pData)),
                   pData,
                   color='k',
                   label='LENS1850',
                   # marker='o',
                   )

    # Compute N-year smoothed means
    for nYrs in [5, 10, 20, 50]:
        pDataN = np.array([np.mean(pData[j:j+nYrs])
                            for j in np.arange(0, len(pData)-nYrs)])
    
        hl, = plt.plot(np.arange(nYrs/2., len(pDataN)+nYrs/2.),
                       pDataN,
                       # lor='r',
                       label='LENS1850_{:0.0f}'.format(nYrs),
                       # marker='o',
                       )


    # Dress figure
    plt.xlabel('time')
    plt.ylabel(title)

    plt.legend()

    try:
        plt.ylim(yLim)
    except NameError:
        pass

    plt.title('Time evolution of {:s}'.format(title))

    plt.grid()

    plt.tight_layout()
    
    plt.draw()

    # Save figure
    if save_flag:
        # Set directory for saving
        if saveDir is None:
            saveDir = setfilepaths()[2]

        # Set file name for saving
        tString = 'mon'
        saveFile = ('full_' + indexName.lower() + '_'
                    '{:04.0f}-{:04.0f}'.format(startYrs[0], startYrs[-1])
                    )

        # Set saved figure size (inches)
        fx, fy = hf.get_size_inches()

        # Save figure
        print(saveDir + saveFile + '.png')
        mwp.savefig(saveDir + saveSubDir + saveFile,
                    shape=np.array([fx, fy]))
        plt.close('all')
        
