#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:37:39 2018

@author: woelfle
"""

# %% Import packages as needed

import numpy as np  # for handling arrays
import xarray as xr  # for handing netcdfs and data sets

import matplotlib.pyplot as plt  # for plotting
import matplotlib.gridspec as gridspec  # for setting up subplots

from socket import gethostname  # for defining local machine

from mdwtools import mdwfunctions as mwfn  # for data processing
from mdwtools import mdwplots as mp  # for plotting things

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

    elif gethostname()[0:6] in ['yslogi', 'geyser']:
        ncDir = '/glade/p/cgd/amp/people/hannay/amwg/climo/'
        ncSubDir = '0.9x1.25/'
        saveDir = '/glade/p/work/woelfle/figs/cesm1to2/'

    return (ncDir, ncSubDir, saveDir)


# %% Main section
if __name__ == '__main__':

    # Set options/flags
    prect_flag = True
    reload_flag = True
    save_flag = False
    saveSubDir = 'testfigs/'
    stdUnits_flag = True
    verbose_flag = False

    latLim = np.array([-20, 0])
    lonLim = np.array([210, 260])

# %% Loading section

    # Get directories for loading and saving files/figures
    ncDir, ncSubDir, saveDir = setfilepaths()

    # Load data
    versionIds = ['01',
                  '28',  '36',
                  'ga7.66', '119', '125',
                  '161', '194', '195'
                  ]
    fileBases = ['b.e15.B1850G.f09_g16.pi_control.01',
                 'b.e15.B1850G.f09_g16.pi_control.28',
                 'b.e15.B1850.f09_g16.pi_control.36',
                 'b.e15.B1850.f09_g16.pi_control.all_ga7.66',
                 'b.e15.B1850.f09_g16.pi_control.all.119',
                 'b.e20.B1850.f09_g16.pi_control.all.125',
                 'b.e20.BHIST.f09_g17.20thC.161_01',
                 'b.e20.B1850.f09_g17.pi_control.all.194',
                 'b.e20.B1850.f09_g17.pi_control.all.195',
                 ]
    loadSuffixes = ['_' + '{:02d}'.format(mon + 1) + '_climo.nc'
                    for mon in range(12)]

    # Create list of files to load
    loadFileLists = {versionIds[j]: [ncDir + fileBases[j] + '/' +
                                     ncSubDir +
                                     fileBases[j] + loadSuffix
                                     for loadSuffix in loadSuffixes]
                     for j in range(len(versionIds))}

    # Open netcdf file(s)
    try:
        if not all([vid in dataSets.keys() for vid in versionIds]):
            load_flag = True
        else:
            load_flag = False
    except NameError:
        load_flag = True

    if load_flag or reload_flag:
        dataSets = {versionId: xr.open_mfdataset(loadFileLists[versionId])
                    for versionId in versionIds}

    # Compute PRECT if needed
    if prect_flag:
        for vid in versionIds:
            if verbose_flag:
                print(vid)
            dataSets[vid]['PRECT'] = mwfn.calcprectda(dataSets[vid])

    # Add version id to dataSets for easy access and bookkeeping
    for vid in versionIds:
        dataSets[vid].attrs['id'] = vid

# %% Plotting section

    # Here we want to make a plot of SST versus overlying low level wind speed.
    #   For starters, we will use U10 as the low level wind speed as every
    #   other output for wind is provided only in component form.

    # Set lat/lon limits

    # Set variables to plot
    xVar = 'TS'
    yVar = 'U10'

    for vid in [versionIds[0]]:
        # Subset to region of interest
        regDs = dataSets[vid].loc[dict(lat=slice(latLim[0], latLim[-1]),
                                       lon=slice(lonLim[0], lonLim[-1]))]

        # Convert to standard units if necessary
        if stdUnits_flag:
            for var in [xVar, yVar]:
                (regDs[var].values,
                 regDs[var].attrs['units']) = mwfn.convertunit(
                     regDs[var].values,
                     regDs[var].units,
                     mwfn.getstandardunits(var)
                     )

        # Convert to one big vector and plot U10 as a function of SST
        hf = plt.figure()

        plt.scatter(regDs[xVar].data[:, :, :].ravel(),
                    regDs[yVar].data[:, :, :].ravel(),
                    15,
                    # np.arange(regDs[xVar].data[:, :, :].size)
                    )
        plt.xlabel('{:s} ({:s})'.format(xVar, regDs[xVar].units))
        plt.ylabel('{:s} ({:s})'.format(yVar, regDs[yVar].units))
