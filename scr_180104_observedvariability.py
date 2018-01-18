#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:50:11 2018

@author: woelfle
"""

# %% Import modules as needed

import numpy as np  # for handling arrays
# import pandas as pd  # for handling 2d things
import xarray as xr  # for handling nd things (netcdfs)

# from scipy import interpolate    # interpolation functions

import matplotlib.pyplot as plt  # for plotting things
import matplotlib.gridspec as gridspec  # for subplot management
import matplotlib.colors as colors
import matplotlib.cm as cmx

from socket import gethostname   # used to determine which machine we are
#                                #   running on

import multiprocessing as mp  # Allow use of multiple cores
import datetime  # For keeping track of run times

from mdwtools import mdwfunctions as mwfn  # For averaging things
from mdwtools import mdwplots as mwp  # For plotting things

import cesm1to2plotter as c1to2p

# import matplotlib.cm as cm
# from scipy.stats import linregress

# %% Define functions as needed


def setfilepaths():
    """
    Set host specific variables and filepaths

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2018-01-04

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


# %% Main section (do stuff)

# %% Main section
if __name__ == '__main__':

    # Set options/flags
    loadClimo_flag = False  # True to only load climatologies
    loadErai_flag = False  # True to load ERAI fields
    loadGpcp_flag = True
    loadHadIsst_flag = True
    mp_flag = True  # True to use multiprocessing when regridding
    ocnOnly_flag = True  # Need to implement to confirm CTindex is right.
    plotSeasonalvTime_flag = True
    plotTSeries_flag = False
    prect_flag = True
    regridVertical_flag = False
    reload_flag = False
    save_flag = False
    saveSubDir = 'testfigs/'
    verbose_flag = False

    # Set new variables to compute when loading
    newVars = 'PRECT'

    # Get directory of file to load
    ncDir, ncSubDir, saveDir = setfilepaths()

    # # Load GPCP
    if loadGpcp_flag:
        # Set directories for GPCP
        gpcpDir = '/home/disk/eos9/woelfle/dataset/GPCP/climo/'
        gpcpFile = 'gpcp_197901-201012.nc'
        gpcpClimoFile = 'gpcp_197901-201012_climo.nc'

        # Load GPCP for all years and add id
        gpcpDs = xr.open_dataset(gpcpDir + gpcpFile)
        gpcpDs.attrs['id'] = 'GPCP_all'

        # Load GPCP from both climo and add id
        if loadClimo_flag:
            gpcpClimoDs = xr.open_dataset(gpcpDir + gpcpClimoFile)
            gpcpClimoDs.attrs['id'] = 'GPCP'

    if loadHadIsst_flag:
        hadIsstDs = mwfn.loadhadisst(daNewGrid=None,
                                     kind='linear',
                                     newGridFile=None,
                                     newGridName='0.9x1.25',
                                     newLat=None,
                                     newLon=None,
                                     qc_flag=False,
                                     regrid_flag=True,
                                     # whichHad='pd_monclimo',
                                     whichHad='all',
                                     )

    if loadErai_flag:
        eraiDs = mwfn.loaderai(daNewGrid=None,
                               kind='linear',
                               loadClimo_flag=True,
                               newGridFile=None,
                               newGridName='0.9x1.25',
                               newLat=None,
                               newLon=None,
                               regrid_flag=False,
                               whichErai='monmean',
                               )
        erai3dDs = mwfn.loaderai(daNewGrid=None,
                                 kind='linear',
                                 loadClimo_flag=True,
                                 newGridFile=None,
                                 newGridName='0.9x1.25',
                                 newLat=None,
                                 newLon=None,
                                 regrid_flag=False,
                                 whichErai='monmean.3d',
                                 )

    # Set variable of interest

    # Conver things to reasonable units if needed
#    newUnits = {'PRECC': 'mm/d',
#                'PRECL': 'mm/d'}
#    if plotVar in newUnits.keys():
#        for vid in versionIds:
#            (dataSets[vid][plotVar].values,
#             dataSets[vid][plotVar].attrs['units']) = \
#                mwfn.convertunit(dataSets[vid][plotVar].values,
#                                 dataSets[vid][plotVar].units,
#                                 newUnits[plotVar]
#                                 )

# %% Make plots
    plot_flag = True
    plotVar = 'TS'

    if any([plot_flag,
            plotSeasonalvTime_flag,
            plotTSeries_flag
            ]):

        divideByTropMean_flag = False
        tropLatLim = np.array([-30, 30])
        tropLonLim = np.array([0, 360])

        rmAnnMean_flag = False

        plev = 250
        diffPlev = 850

        if plotVar == 'PRECT':
            try:
                obsDs = gpcpDs
            except NameError:
                raise NameError('gpcpDs not loaded')
            obsVar = 'precip'
            latLim = np.array([-20, 0])
            lonLim = np.array([210, 260])
            ocnOnly_flag = False
            rmRefRegMean_flag = False
            tickStep_yrs = 2
            yLim = np.array([1, 2.75])
        elif plotVar == 'TS':
            try:
                obsDs = hadIsstDs
            except NameError:
                raise NameError('hadIsstDs not loaded')
            obsVar = 'sst'
            ocnOnly_flag = True
            latLim = np.array([-3, 3])
            lonLim = np.array([180, 220])
            rmRefRegMean_flag = True
            refLatLim = np.array([-20, 20])
            refLonLim = np.array([150, 250])
            tickStep_yrs = 10
            yLim = np.array([-2, 1])
        else:
            rmRefRegMean_flag = False

        lRunMean = 10

        # Copmute regional mean to be plotted
        obsRegMeanDa = mwfn.calcdaregmean(obsDs[obsVar],
                                          gwDa=None,
                                          latLim=latLim,
                                          lonLim=lonLim,
                                          stdUnits_flag=True,
                                          )
        if divideByTropMean_flag:
            obsTropMeanDa = mwfn.calcdaregmean(obsDs[obsVar],
                                               gwDa=None,
                                               latLim=tropLatLim,
                                               lonLim=tropLonLim,
                                               stdUnits_flag=True,
                                               )
        if rmRefRegMean_flag:
            obsRefRegMeanDa = mwfn.calcdaregmean(obsDs[obsVar],
                                                 gwDa=None,
                                                 latLim=refLatLim,
                                                 lonLim=refLonLim,
                                                 stdUnits_flag=True,
                                                 )

            # Subtract off reference regional mean, but assume this to be
            #   a difference between levels as opposed to a difference of
            #   regions
            if np.ndim(obsRefRegMeanDa) == 2:
                obsRefRegMeanDa = obsRefRegMeanDa.loc[
                    dict(plev=slice(diffPlev, diffPlev))]
                obsRefRegMeanDa['plev'].values = np.array([plev])
                obsRegMeanDa = obsRegMeanDa - obsRefRegMeanDa
            else:
                obsRegMeanDa = obsRegMeanDa - obsRefRegMeanDa

        # Pull appropriate level if needed
        if np.ndim(obsRegMeanDa) == 2:
            obsRegMeanDa = obsRegMeanDa.loc[dict(plev=slice(plev, plev))]
            try:
                obsTropMeanDa = obsTropMeanDa.loc[dict(plev=slice(plev,
                                                                  plev))]
            except NameError:
                pass

        # Get data for plotting and plot it
        if divideByTropMean_flag:
            obsRegMeanDa = obsRegMeanDa/obsTropMeanDa
        pData = (obsRegMeanDa.values - obsRegMeanDa.mean(dim='time').values
                 if rmAnnMean_flag
                 else obsRegMeanDa.values)

    # %% Plot seasonal cycle through time
    if plotSeasonalvTime_flag:
        cycData = pData[:-np.mod(pData.size, 12)].reshape(
            [int(pData.size/12), 12])
        mons = np.arange(0, pData.size)
        tCyc = mons[:-np.mod(mons.size, 12)].reshape(
            [int(mons.size/12), 12]).mean(axis=1)

        hf = plt.figure()

        plt.pcolormesh(np.arange(12),
                       tCyc,
                       (cycData.transpose() - cycData.mean(axis=1)
                        ).transpose(),
                       cmap='RdBu_r')

        plt.colorbar()
        # %%
        hf = plt.figure()
        for l, lRunMean in enumerate([10, 20, 30]):
            for d, plotDir in enumerate([-1, 1]):
                ax = hf.add_subplot(231 + l + 3*d)

                tStarts = np.arange(0, len(tCyc)-lRunMean, lRunMean)

                cm = plt.get_cmap('RdYlBu_r')
                cNorm = colors.Normalize(vmin=0, vmax=len(tCyc))
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
                print(scalarMap.get_clim())

                for j in tStarts[::plotDir]:
                    colorVal = scalarMap.to_rgba(j)
                    plt.plot(np.arange(12)+1,
                             (cycData[j:j+lRunMean].mean(axis=0) -
                              cycData[j:j+lRunMean].mean(axis=0).mean()),
                             color=colorVal,
                             label='{:d}-{:d}'.format(j+1870, j+lRunMean+1870),
                             )

                plt.xlabel('Month')
                plt.xlim([1, 12])

                plt.ylabel('CTI, Ann. Mean Removed (K)')
                plt.ylim([-2., 2.])

                plt.grid()

                plt.legend(ncol=3)

                plt.title('Averaged over {:d} yrs.'.format(lRunMean))

        # plt.legend()

    # %% Plot time series of regional means
    if plotTSeries_flag:

        lw = 2
        lc = [[0, 0, 0],
              [1, 0, 0],
              [0, 0, 1]]
        marker = '^'

        hf = plt.figure()
        # Plot raw time series
        t = np.arange(0, pData.size)
        hl, = plt.plot(t,
                       pData,
                       lw=lw,
                       c=lc[0],
                       label='Mon. mean',
                       marker=marker,
                       )

        # Plot annual mean time series
        # hf = plt.figure()
        annMean = pData[:-np.mod(pData.size, 12)].reshape(
            [int(pData.size/12), 12]).mean(axis=1)
        tAnnMean = t[:-np.mod(t.size, 12)].reshape(
            [int(t.size/12), 12]).mean(axis=1)
        hl, = plt.plot(tAnnMean,
                       annMean,
                       lw=lw,
                       c=lc[1],
                       label='Ann. mean',
                       marker=marker,
                       )

        # Plot ten year running mean of annual mean time series
        runMean = np.zeros_like(annMean)
        runMean = runMean[:-lRunMean]
        tRunMean = np.zeros_like(runMean)
        for j in np.arange(0, annMean.size-lRunMean):
            runMean[j] = np.mean(annMean[j:j+lRunMean])
            tRunMean[j] = np.mean(tAnnMean[j:j+lRunMean])
        hl, = plt.plot(tRunMean,
                       runMean,
                       lw=lw,
                       c=lc[2],
                       label='10-yr. mean',
                       marker=marker,
                       )

        plt.xlim([0, pData.size])
        plt.xticks(np.arange(0, pData.size, 12*tickStep_yrs))
        plt.xlabel('Months since {:d}-{:02d}'.format(
            obsDs['time'][0].data.astype('datetime64[Y]').astype(int) + 1970,
            np.mod(obsDs['time'][0].data.astype('datetime64[M]').astype(int),
                   12) + 1,
            # (obsDs['time'][0].data -
            #  obsDs['time'][0].data.astype('datetime64[M]').astype(int) + 1)
            ))

        try:
            plotVarString = {'precip': 'Precipitation',
                             'sst': 'SST',
                             }[obsVar]
        except KeyError:
            plotVarString = obsVar
        plt.ylabel('{:s} ({:s})'.format(plotVarString,
                                        obsDs[obsVar].units))

        plt.legend()
