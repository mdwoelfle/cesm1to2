#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:06:15 2017

@author: woelfle
"""

# %% Import modules as needed

import os                        # import operating system functions
import sys
try:
    if os.isatty(sys.stdout.fileno()):
        import matplotlib
        matplotlib.use('Agg')
except:
    pass

from socket import gethostname   # used to determine which machine we are
#                                #   running on

import numpy as np  # for handling arrays
# import pandas as pd  # for handling 2d things
import xarray as xr  # for handling nd things (netcdfs)

# from scipy import interpolate    # interpolation functions

import matplotlib.pyplot as plt  # for plotting things

from mdwtools import mdwfunctions as mwfn  # For averaging things
from mdwtools import mdwplots as mwp  # For plotting things

import cesm1to2plotter as c1to2p
# import matplotlib.cm as cm
from scipy.stats import linregress

# %% Define funcitons as needed


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

    elif gethostname()[0:6] in ['yslogi', 'geyser', 'cheyen']:
        ncDir = '/glade/p/cgd/amp/people/hannay/amwg/climo/'
        ncSubDir = '0.9x1.25/'
        saveDir = '/glade/p/work/woelfle/figs/cesm1to2/'

    return (ncDir, ncSubDir, saveDir)


def getcolordict():
    return {'01': '#1f77b4',
            '119': '#9467bd',
            '125': '#8c564b',
            '161': '#e377c2',
            '194': '#7f7f7f',
            '195': '#bcbd22',
            '28': '#ff7f0e',
            '36': '#2ca02c',
            'ga7.66': '#d62728',
            'obs': [0, 0, 0]}


# %%
if __name__ == '__main__':

    # Set new variables to compute when loading
    newVars = 'PRECT'

    # Set flags for loading/plotting/doing things
    loadGpcp_flag = False
    loadHadIsst_flag = False
    ocnOnly_flag = False
    prect_flag = True
    save_flag = True
    verbose_flag = False

    # Get directory of file to load
    ncDir, ncSubDir, saveDir = setfilepaths()

    # Set name(s) of file(s) to load
    for versionId in ['28', '36',
                      # 'ga7.66', '119', '125',
                      # '161', '194', '195'
                      ]:
        print('\n\n---\nProcessing {:s}\n---\n\n'.format(versionId))
        caseBase = c1to2p.getcasebase(versionId)
        yrIds = c1to2p.getavailableyearslist(versionId)
        yrDirs = c1to2p.getyearsubdirs(versionId)
        # subDirs = ['yrs_' + yr for yr in yrIds]
        fileName = caseBase + '_ANN_climo.nc'

        # Create list of files to load
        loadFiles = {yid: (ncDir + caseBase + '/' +
                           yrDirs[jId] + '/' + fileName)
                     for jId, yid in enumerate(yrIds)}

        # Open netcdf file(s)
        dataSets = {yid: xr.open_dataset(loadFiles[yid])
                    for yid in yrIds}

        # Compute PRECT if needed
        if prect_flag:
            for yid in yrIds:
                if verbose_flag:
                    print(yid)
                dataSets[yid]['PRECT'] = mwfn.calcprectda(dataSets[yid])

        # Add version id to dataSets for easy access and bookkeeping
        for yid in yrIds:
            dataSets[yid]['PRECT'] = mwfn.calcprectda(dataSets[yid])

        # Load GPCP
        if loadGpcp_flag:
            # Set directories for GPCP
            gpcpDir = '/home/disk/eos9/woelfle/dataset/GPCP/climo/'
            gpcpClimoFile = 'gpcp_197901-201012_annclimo.nc'

            # Load GPCP from both climo and add id
            gpcpDs = xr.open_dataset(gpcpDir + gpcpClimoFile)
            gpcpDs.attrs['id'] = 'GPCP'

        if loadHadIsst_flag:
            hadIsstDs = mwfn.loadhadisst(daNewGrid=None,
                                         kind='linear',
                                         newGridFile=None,
                                         newGridName='0.9x1.25',
                                         newLat=None,
                                         newLon=None,
                                         qc_flag=False,
                                         regrid_flag=True,
                                         whichHad='pd_monclimo',
                                         )

    # %% Plot bias indices across averaging period

        if True:  # plotRegMean_flag:

            # Set variable for plotting
            plotVar = 'TS'

            # Set flags and other options based on variable to be plotted
            if plotVar == 'TS':
                latLim = np.array([-3, 3])
                lonLim = np.array([180, 220])
                try:
                    obsDs = hadIsstDs
                except NameError:
                    obsDs = None
                obsVar = 'sst'
                ocnOnly_flag = True
                rmRefRegMean_flag = True
                refLatLim = np.array([-20, 20])
                refLonLim = np.array([150, 250])
                yLim = np.array([-2.0, 1.0])
            elif plotVar in ['PRECT', 'PRECC', 'PRECL']:
                latLim = np.array([-20, 0])
                lonLim = np.array([210, 260])
                try:
                    obsDs = gpcpDs
                except NameError:
                    obsDs = None
                obsVar = 'precip'
                ocnOnly_flag = ocnOnly_flag
                rmRefRegMean_flag = False
                yLim = np.array([1.00, 2.75])
            else:
                latLim = np.array([-20, 20])
                lonLim = np.array([120, 270])
                ocnOnly_flag = False
                rmRefRegMean_flag = False
                yLim = None

            # Create dictionary to hold annual mean value (and colors)
            annMean = dict()
            colorDict = dict()

            # Determine if obs are available to plot
            if obsDs is None:
                plotObs_flag = False

            # Compute regional mean over double-ITCZ region as defined in
            #   Bellucci et al. (2010, J Clim)
            # latLim = np.array([-20, 0])
            # lonLim = np.array([210, 260])
            # latLim = np.array([-3, 3])
            # lonLim = np.array([180, 220])
            # latLim = np.array([-20, 20])
            # lonLim = np.array([150, 250])

            for yid in yrIds:
                # Compute regional mean through time
                regMeanDs = mwfn.calcdsregmean(dataSets[yid][plotVar],
                                               gwDa=dataSets[yid]['gw'],
                                               latLim=latLim,
                                               lonLim=lonLim,
                                               ocnOnly_flag=ocnOnly_flag,
                                               qc_flag=False,
                                               landFracDa=(
                                                   dataSets[yid]['LANDFRAC']),
                                               stdUnits_flag=True,
                                               )

                # Compute reference regional mean if needed
                if rmRefRegMean_flag:
                    refRegMeanDs = mwfn.calcdsregmean(
                        dataSets[yid][plotVar],
                        gwDa=dataSets[yid]['gw'],
                        latLim=refLatLim,
                        lonLim=refLonLim,
                        ocnOnly_flag=ocnOnly_flag,
                        landFracDa=dataSets[yid]['LANDFRAC'],
                        qc_flag=False,
                        stdUnits_flag=True,
                        )
                    regMeanDs = regMeanDs - refRegMeanDs

                annMean[yid] = regMeanDs.mean(dim='time')

            # Compute regional mean of observations
            if plotObs_flag:
                obsRegMeanDs = mwfn.calcdsregmean(obsDs[obsVar],
                                                  gwDa=None,
                                                  latLim=latLim,
                                                  lonLim=lonLim,
                                                  stdUnits_flag=True,
                                                  )
                # Compute regional mean of reference region for observations
                if rmRefRegMean_flag:
                    obsRefRegMeanDs = mwfn.calcdsregmean(obsDs[obsVar],
                                                         gwDa=None,
                                                         latLim=refLatLim,
                                                         lonLim=refLonLim,
                                                         stdUnits_flag=True,
                                                         )

                    obsRegMeanDs = obsRegMeanDs - obsRefRegMeanDs

                annMean['obs'] = obsRegMeanDs.mean(dim='time')

            # Plot annual mean values
            plt.figure()
            plt.scatter(np.arange(1, len(annMean) + 1),
                        np.array([annMean[yid] for yid in yrIds]),
                        marker='o',
                        c=[getcolordict()[versionId] for yid in yrIds],
                        s=80,
                        )
            if plotObs_flag:
                plt.scatter([len(annMean)],
                            annMean['obs'],
                            marker='^',
                            c=getcolordict()['obs'],
                            s=80,
                            )

            if plotObs_flag:
                plt.xticks(np.arange(1, len(annMean) + 2),
                           yrIds + ['obs'])
            else:
                plt.xticks(np.arange(1, len(annMean) + 1),
                           yrIds)
            plt.xlabel('Averaging Period')

            plt.ylabel(plotVar + ' (' +
                       mwp.getlatlimstring(latLim) + ', ' +
                       mwp.getlonlimstring(lonLim, lonFormat='EW') + ')'
                       )
            plt.ylim(yLim)

            plt.grid(ls='--')
            plt.gca().set_axisbelow(True)

            if plotVar in ['PRECT', 'PRECC', 'PRECL']:
                plt.title('Annual mean 2xITCZ index ({:s}) for v{:s}'.format(
                    plotVar, versionId))
            elif plotVar in ['TS']:
                plt.title('Annual mean CT index ({:s}) for v{:s}'.format(
                    plotVar, versionId))

           # plt.show()

            if save_flag:
                if plotVar in ['PRECT', 'PRECC', 'PRECL']:
                    mwp.savefig(saveDir +
                                'annmean_dITCZindex_v{:s}'.format(versionId))
                elif plotVar in ['TS']:
                    mwp.savefig(saveDir +
                                'annmean_CTI_v{:s}'.format(versionId))
