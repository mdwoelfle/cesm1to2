#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:10:25 2017

@author: woelfle
"""

# %% Import modules

# Correction for running remotely
import os                        # import operating system functions
import sys
try:
    if os.isatty(sys.stdout.fileno()):
        import matplotlib
        matplotlib.use('Agg')
except:
    pass

import matplotlib.pyplot as plt  # import matplotlib for plotting
import matplotlib.gridspec as gridspec  # pretty subplots

import mdwtools.mdwfunctions as mwfn      # import personal processing functions
import mdwtools.mdwplots as mwp           # import personal plotting functions
import netCDF4 as nc4            # import netCDF4 as nc4
import numpy as np               # import numpy as np
from socket import gethostname   # used to determine which machine we are
#                                #   running on
from datetime import datetime    # for working with dates and stuff
import time

# Functions for plotlatloncontsovertime
from mpl_toolkits.basemap import Basemap  # import tool for lat/lon plotting
from matplotlib import cm  # import access to colormaps

from scipy import interpolate    # import interpolation functions from scipy
# import scr_150720_popbasics as popbasics
# import scr_150806_atmbasics as atmbasics

import multiprocessing as mp  # Allow use of multiple cores

# %% Define functions


def getavailableyearslist(versionId):
    """
    Get list of averaging periods available for a given model version
    """
    return {'01': None,
            '28': ['2-10', '2-20', '50-74', '75-99'],
            '36': ['2-10', '2-20', '21-40', '60-60', '75-99'],
            'ga7.66': ['2-20', '20-39', '55-74'],
            '119': ['2-9', '2-21', '21-40', '30-49', '75-99'],
            '125': ['2-9', '2-21', '11-30', '21-40', '30-49', '70-89', '80-99',
                    '100-109', '100-119'],
            '161': ['1850-1869', '1920-1939', '1980-1999'],
            '194': ['14-33', '15-29', '50-69', '100-119'],
            '195': ['15-29', '50-69', '80-99', '100-119', '122-141'],
            }[versionId]


def getcasebase(versionId):
    """
    Get long form nave for a given version ID for cesm1to2 cases
    """
    return {'01': 'b.e15.B1850G.f09_g16.pi_control.01',
            '28': 'b.e15.B1850G.f09_g16.pi_control.28',
            '36': 'b.e15.B1850.f09_g16.pi_control.36',
            'ga7.66': 'b.e15.B1850.f09_g16.pi_control.all_ga7.66',
            '119': 'b.e15.B1850.f09_g16.pi_control.all.119',
            '125': 'b.e20.B1850.f09_g16.pi_control.all.125',
            '161': 'b.e20.BHIST.f09_g17.20thC.161_01',
            '194': 'b.e20.B1850.f09_g17.pi_control.all.194',
            '195': 'b.e20.B1850.f09_g17.pi_control.all.195',
            }[versionId]


def getcompcont(plotVar,
                diff_flag=False):
    """
    Determine comparison contour value for given plotVar

    Author:
        Matthew Woelfle

    Version Date:
        2017-10-17

    Args:
        plotVar - name of variable (in CESM parlance) for which comparison
            contour is to be retrieved

    Kwargs:
        diff_flag - true if plotting difference in variable
    """
    if not diff_flag:
        try:
            compcont = {'FLNS': np.array([0.]),
                        'FNS': np.array([0.]),
                        'FSNS': np.array([0.]),
                        'LHFLX': np.array([0.]),
                        'OMEGA500': np.array([0.]),
                        'OMEGA850': np.array([0.]),
                        'PRECC': np.array([2.]),
                        'PRECL': np.array([2.]),
                        'PRECT': np.array([2.]),
                        'PS': np.array([1008.]),
                        'SHFLX': np.array([0]),
                        'TAUX': np.array([0]),
                        'TAUY': np.array([0]),
                        'TS': np.array([300]),
                        'curlTau': np.array([0]),
                        'curlTau_y': np.array([0]),
                        'divTau': np.array([0]),
                        'ekmanx': np.array([0]),
                        'ekmany': np.array([0]),
                        'precip': np.array([2.]),
                        'sst': np.array([300]),
                        'sverdrupx': np.array([0]),
                        'MGx': np.array([0])
                        }[plotVar]
        except KeyError:
            compcont = None
    else:
        compcont = np.array([0])

    return compcont


def getmapcontlevels(plotVar,
                     diff_flag=False):
    """
    Determine contour values for given plotVar

    Author:
        Matthew Woelfle

    Version Date:
        2017-10-17

    Args:
        plotVar - name of variable (in CESM parlance) for which contours are to
            be retrieved

    Kwargs:
        diff_flag - true if plotting difference in variable
    """
    if not diff_flag:
        try:
            levels = {'FLNS': np.arange(0., 120.1, 10),
                      'FNS': np.arange(-600., 600.1, 100),
                      'FSNS': np.arange(0, 400.1, 20.),
                      'LHFLX': np.arange(0, 200.1, 10),
                      'OMEGA500': np.arange(-0.125, 0.1251, 0.0125),
                      'OMEGA850': np.arange(-0.125, 0.1251, 0.0125),
                      'PRECC': np.arange(0, 20.1, 2),
                      'PRECL': np.arange(0, 20.1, 2),
                      'PRECT': np.arange(0, 20.1, 2),
                      'PS': np.arange(1004., 1013.1, 1),
                      'SHFLX': np.arange(0, 20., 1.),
                      'TAUX': np.arange(-0.2, 0.201, 0.02),
                      'TAUY': np.arange(-0.1, 0.101, 0.01),
                      'TS': np.arange(290, 305, 1),
                      'curlTau': np.arange(-3e-7, 3.001e-7, 3e-8),
                      'curlTau_y': np.arange(-4e-13, 4.01e-13, 4e-14),
                      'divTau': np.arange(-2e-7, 2.01e-7, 2e-8),
                      'ekmanx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'ekmany': np.arange(-3e4, 3.01e4, 3e3),
                      'precip': np.arange(0, 20.1, 2),
                      'sst': np.arange(290, 305, 1),
                      'sverdrupx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'MGx': np.arange(-1.5e5, 1.501e5, 1.5e4)
                      }[plotVar]
        except KeyError:
            levels = None
    else:
        try:
            levels = {'FLNS': np.arange(-30., 30.1, 3),
                      # 'FNS': np.arange(-600., 600.1, 100),
                      'FNS': np.arange(-200, 200.1, 20),
                      'FSNS': np.arange(-50, 50.1, 5.),
                      'LHFLX': np.arange(-50, 50.1, 5),
                      'OMEGA500': np.arange(-0.125, 0.1251, 0.0125),
                      'OMEGA850': np.arange(-0.125, 0.1251, 0.0125),
                      'PRECC': np.arange(-10, 10.1, 1),
                      'PRECL': np.arange(-10, 10.1, 1),
                      'PRECT': np.arange(-10, 10.1, 1),
                      'PS': np.arange(-4., 4.01, 0.5),
                      'SHFLX': np.arange(-10, 10., 1.),
                      'TAUX': np.arange(-0.1, 0.101, 0.01),
                      'TAUY': np.arange(-0.1, 0.101, 0.01),
                      'TS': np.arange(-2, 2.1, 0.2),
                      'curlTau': np.arange(-1.5e-7, 1.51e-7, 1.5e-8),
                      'curlTau_y': np.arange(-4e-13, 4.01e-13, 4e-14),
                      'divTau': np.arange(-1e-7, 1.01e-7, 1e-8),
                      'ekmanx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'ekmany': np.arange(-1e4, 1.01e4, 1e3),
                      'precip': np.arange(-10, 10.1, 1),
                      'sst': np.arange(-2, 2.1, 0.2),
                      'sverdrupx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'MGx': np.arange(-1.5e5, 1.501e5, 1.5e4)
                      }[plotVar]
        except KeyError:
            levels = None

    return levels


def getzonmeancontlevels(plotVar,
                         diff_flag=False):
    """
    Determine contour values for given plotVar

    Author:
        Matthew Woelfle

    Version Date:
        2017-10-31

    Args:
        plotVar - name of variable (in CESM parlance) for which contours are to
            be retrieved

    Kwargs:
        diff_flag - true if plotting difference in variable
    """
    if not diff_flag:
        try:
            levels = {'FLNS': np.arange(0., 120.1, 10),
                      'FNS': np.arange(-600., 600.1, 100),
                      'FSNS': np.arange(0, 400.1, 20.),
                      'LHFLX': np.arange(0, 200.1, 10),
                      'OMEGA500': np.arange(-0.125, 0.1251, 0.0125),
                      'OMEGA850': np.arange(-0.125, 0.1251, 0.0125),
                      'PRECC': np.arange(0, 20.1, 2),
                      'PRECL': np.arange(0, 20.1, 2),
                      'PRECT': np.arange(0, 20.1, 2),
                      'PS': np.arange(1004., 1013.1, 1),
                      'SHFLX': np.arange(0, 20., 1.),
                      'TAUX': np.arange(-0.2, 0.201, 0.02),
                      'TAUY': np.arange(-0.1, 0.101, 0.01),
                      'TS': np.arange(290, 305, 1),
                      'curlTau': np.arange(-3e-7, 3.001e-7, 3e-8),
                      'curlTau_y': np.arange(-4e-13, 4.01e-13, 4e-14),
                      'divTau': np.arange(-2e-7, 2.01e-7, 2e-8),
                      'ekmanx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'ekmany': np.arange(-3e4, 3.01e4, 3e3),
                      'precip': np.arange(0, 20.1, 2),
                      'sst': np.arange(290, 305, 1),
                      'sverdrupx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'MGx': np.arange(-1.5e5, 1.501e5, 1.5e4)
                      }[plotVar]
        except KeyError:
            levels = None
    else:
        try:
            levels = {'FLNS': np.arange(-30., 30.1, 3),
                      # 'FNS': np.arange(-600., 600.1, 100),
                      'FNS': np.arange(-200, 200.1, 20),
                      'FSNS': np.arange(-50, 50.1, 5.),
                      'LHFLX': np.arange(-50, 50.1, 5),
                      'OMEGA500': np.arange(-0.125, 0.1251, 0.0125),
                      'OMEGA850': np.arange(-0.125, 0.1251, 0.0125),
                      'PRECC': np.arange(-10, 10.1, 1),
                      'PRECL': np.arange(-10, 10.1, 1),
                      'PRECT': np.arange(-10, 10.1, 1),
                      'PS': np.arange(-4., 4.01, 0.5),
                      'SHFLX': np.arange(-10, 10., 1.),
                      'TAUX': np.arange(-0.1, 0.101, 0.01),
                      'TAUY': np.arange(-0.1, 0.101, 0.01),
                      'TS': np.arange(-2, 2.1, 0.2),
                      'curlTau': np.arange(-1.5e-7, 1.51e-7, 1.5e-8),
                      'curlTau_y': np.arange(-4e-13, 4.01e-13, 4e-14),
                      'divTau': np.arange(-1e-7, 1.01e-7, 1e-8),
                      'ekmanx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'ekmany': np.arange(-1e4, 1.01e4, 1e3),
                      'precip': np.arange(-10, 10.1, 1),
                      'sst': np.arange(-2, 2.1, 0.2),
                      'sverdrupx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'MGx': np.arange(-1.5e5, 1.501e5, 1.5e4)
                      }[plotVar]
        except KeyError:
            levels = None

    return levels


def plotlatlon(ds,
               plotVar,
               box_flag=False,
               caseString=None,
               cbar_flag=True,
               cbar_dy=0.001,
               cbar_height=0.02,
               cMap=None,
               compcont=None,
               compcont_flag=True,
               convertUnits_flag=True,
               diff_flag=False,
               diffDs=None,
               diffTSteps=None,
               diffVar=None,
               fontSize=12,
               latLim=np.array([-30, 30]),
               latlbls=None,
               levels=None,
               lonLim=np.array([119.5, 270.5]),
               lonlbls=None,
               newUnits=None,
               ocnOnly_flag=False,
               qc_flag=False,
               quiver_flag=False,
               quiverScale=0.4,
               quiverUnits='inches',
               rmRegLatLim=None,
               rmRegLonLim=None,
               rmRegMean_flag=False,
               rmse_flag=False,
               stampDate_flag=True,
               tSteps=None,
               tStepLabel_flag=True,
               uVar='TAUX',
               vVar='TAUY',
               **kwargs
               ):
    """
    Plot a map of a given dataset averaged over the specified timesteps

    Version Date:
        2017-10-17
    """

    # Set lats/lons to label if not provided
    if latlbls is None:
        latlbls = mwp.getlatlbls(latLim)
    if lonlbls is None:
        lonlbls = mwp.getlonlbls(lonLim)

    # Get levels for contouring if not provided
    if levels is None:
        levels = getmapcontlevels(plotVar,
                                  diff_flag=any([diff_flag,
                                                 rmRegMean_flag])
                                   )
    if compcont is None:
        compcont = getcompcont(plotVar,
                               diff_flag=any([diff_flag,
                                              rmRegMean_flag])
                               )

    # Find colormap for means
    if cMap is None:
        cMap = mwp.getcmap(plotVar,
                           diff_flag=any([diff_flag,
                                          rmRegMean_flag])
                           )

    # Determine time steps for plotting
    if tSteps is None:
        tSteps = np.arange(0, ds[plotVar].shape[0], dtype=int)
    if diffTSteps is None:
        diffTSteps = tSteps

    # Set caseString for plotting
    if caseString is None:
        if diff_flag:
            caseString = ds.id + '-' + diffDs.id
        else:
            caseString = ds.id

    # Ensure diffVar is defined as non-None
    if diffVar is None:
        diffVar = plotVar

    # Convert units as needed
    (ds[plotVar].values,
     ds[plotVar].attrs['units']) = mwfn.convertunit(
        ds[plotVar].values,
        ds[plotVar].units,
        mwfn.getstandardunits(plotVar)
        )
    if diff_flag:
        try:
            (diffDs[diffVar].values,
             diffDs[diffVar].attrs['units']) = mwfn.convertunit(
                diffDs[diffVar].values,
                diffDs[diffVar].units,
                mwfn.getstandardunits(diffVar)
                )
        except TypeError:
            pass

    # Pull data for plotting
    if diff_flag:
        pData = (ds[plotVar].values[tSteps, :, :].mean(axis=0) -
                 diffDs[diffVar].values[diffTSteps, :, :].mean(axis=0))
        if quiver_flag:
            uData = (ds[uVar].data[tSteps, :, :].mean(axis=0) -
                     diffDs[uVar].data[diffTSteps, :, :].mean(axis=0))
            vData = (ds[vVar].data[tSteps, :, :].mean(axis=0) -
                     diffDs[vVar].data[diffTSteps, :, :].mean(axis=0))
        else:
            uData = None
            vData = None
    else:
        pData = ds[plotVar].values[tSteps, :, :].mean(axis=0)
        if quiver_flag:
            uData = ds[uVar].data[tSteps, :, :].mean(axis=0)
            vData = ds[vVar].data[tSteps, :, :].mean(axis=0)
        else:
            uData = None
            vData = None

    # Compute and subtract off regional mean if requested
    if rmRegMean_flag:
        # Get averaging lat/lon limits if not explicity provided
        if rmRegLatLim is None:
            rmRegLatLim = latLim
        if rmRegLonLim is None:
            rmRegLonLim = lonLim

        # Compute regional mean through time
        regMeanDs = mwfn.calcdsregmean(ds[plotVar],
                                       gwDa=(ds['gw']
                                             if 'gw' in ds
                                             else None),
                                       latLim=rmRegLatLim,
                                       lonLim=rmRegLonLim,
                                       ocnOnly_flag=ocnOnly_flag,
                                       landFracDa=(ds['LANDFRAC']
                                                   if 'LANDFRAC' in ds
                                                   else None),
                                       qc_flag=qc_flag,
                                       stdUnits_flag=False,
                                       )

        # Compute time mean regional mean to be subtracted
        regMean = regMeanDs.values[tSteps].mean(axis=0)

        # Subtract off regional mean
        pData = pData - regMean
        if qc_flag:
            print(regMean)

    # Set variables to only extend one end of colorbar
    maxExtendVars = ['PRECT', 'PRECC', 'PRECL', 'precip']

    # if plotting difference, extend both ends for all variables
    if diff_flag:
        maxExtendVars = []

    # Plot map
    im1, ax = mwp.plotmap(ds.lon,
                          ds.lat,
                          pData,
                          box_flag=box_flag,
                          caseString=caseString,
                          cbar_flag=cbar_flag,
                          cbar_dy=cbar_dy,
                          cbar_height=cbar_height,
                          cMap=cMap,
                          compcont=(compcont
                                    if compcont_flag
                                    else None),
                          extend=['both', 'max'][plotVar in maxExtendVars],
                          fill_color=[0.3, 0.3, 0.3],
                          fontsize=10,  # fontSize,
                          latlbls=latlbls,
                          latLim=latLim,
                          levels=levels,
                          lonlbls=lonlbls,
                          lonLim=lonLim,
                          varName=mwp.getplotvarstring(ds[plotVar].name),
                          varUnits=ds[plotVar].units,
                          quiver_flag=quiver_flag,
                          quiverScale=quiverScale,
                          quiverUnits=quiverUnits,
                          U=uData,
                          Uname=(ds[uVar].name
                                 if uVar in ds.data_vars
                                 else None),
                          Uunits=(ds[uVar].units
                                  if uVar in ds.data_vars
                                  else None),
                          Uref=0.1,
                          V=vData,
                          subSamp=(3  # ds['TAUX'].shape[1]/36
                                   if uVar in ds.data_vars
                                   else None),
                          tStepLabel_flag=False,
                          **kwargs
                          )

    # Add removed regional mean to annotations
    if rmRegMean_flag:
        rmMeanString = '\nRm Reg Mean = {:0.1f} {:s}'.format(regMean,
                                                             ds[plotVar].units)
    else:
        rmMeanString = ''

    # Add time steps used to annotations
    if tStepLabel_flag:
        tStepString = 't = [{:0d}, {:0d}]'.format(tSteps[0], tSteps[-1])
    else:
        tStepString = ''

    # Add annotations to plot
    ax.annotate(
            tStepString + rmMeanString,
            xy=(1, 1),
            xycoords='axes fraction',
            horizontalalignment='right',
            verticalalignment='bottom'
            )

    if rmse_flag:
        print('need to add back RMSE computation')

    if stampDate_flag:
        mwp.stampdate(x=1, y=0)

    return (im1, ax, compcont)


def plotmultilatlon(dsDict,
                    plotIdList,
                    plotVar,
                    box_flag=False,
                    cbar_flag=True,
                    cbarOrientation='vertical',
                    compcont_flag=True,
                    diff_flag=False,
                    diffIdList=None,
                    diffDs=None,
                    diffVar=None,
                    fontSize=12,
                    latLim=np.array([-30, 30]),
                    latlbls=None,
                    lonLim=np.array([119.5, 270.5]),
                    lonlbls=None,
                    quiver_flag=False,
                    quiverScale=0.4,
                    quiverUnits='inches',
                    rmse_flag=False,
                    rmRegMean_flag=False,
                    save_flag=False,
                    saveDir=None,
                    stampDate_flag=False,
                    subFigCountStart='a',
                    tSteps=None,
                    **kwargs
                    ):
    """
    Plot maps from multiple cases for comparison

    Version Date:
        2017-10-17
    """

    # Set lat/lon labels
    if latlbls is None:
        latlbls = mwp.getlatlbls(latLim)
    if lonlbls is None:
        lonlbls = mwp.getlonlbls(lonLim)

    # Ensure diffIdList or diffDs provided if diff_flag
    if all([diff_flag, (diffIdList is None), (diffDs is None)]):
        raise ValueError('diffIdList or diffDs must be provided to plot ' +
                         'differences')

    # Set variable for differencing; assumes plotVar if not provided
    if diffVar is None:
        diffVar = plotVar

    # Determine time step parameters
    if tSteps is None:
        tSteps = np.arange(0, dsDict[plotIdList[0]][plotVar].shape[0])

    # Ensure box_flag is iterable
    if isinstance(box_flag, bool):
        box_flag = [box_flag]*len(plotIdList)

    # Create figure for plotting
    hf = plt.figure()
    if len(plotIdList) == 3:
        if cbarOrientation == 'vertical':
            # Set figure window size
            hf.set_size_inches(9, 10, forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(5, 2,
                                   height_ratios=[20, 1, 20, 1, 20],
                                   width_ratios=[30, 1])

            # Set gridspec colorbar location
            cbColInd = 1
            cbRowInd = 0

        elif cbarOrientation == 'horizontal':
            # Set figure window size
            hf.set_size_inches(7.5, 10, forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(6, 1,
                                   height_ratios=[20, 1, 20, 1, 20, 1],
                                   )

            # Set gridspec colorbar location
            cbColInd = 0
            cbRowInd = 5

        # Set gridpsec index order
        colInds = [0, 0, 0]
        rowInds = [0, 2, 4]
    elif len(plotIdList) == 9:
        if cbarOrientation == 'vertical':
            # Set figure window size
            hf.set_size_inches(16.25, 7.75, forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(3, 4,
                                   height_ratios=[1, 1, 1],
                                   hspace=0.05,
                                   width_ratios=[30, 30, 30, 1],
                                   wspace=0.15)

            # Set gridspec colorbar location
            cbColInd = 3
            cbRowInd = 0
        elif cbarOrientation == 'horizontal':
            # Set figure window size
            hf.set_size_inches(14, 12, forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(4, 3,
                                   height_ratios=[20, 20, 20, 1],
                                   hspace=0.5,
                                   width_ratios=[1, 1, 1],
                                   wspace=0.5)

            # Set gridspec colorbar location
            cbColInd = 0
            cbRowInd = 3

        # Set gridspec index order
        colInds = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        rowInds = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    # Set figure window title
    hf.canvas.set_window_title(('d' if diff_flag else '') +
                               'complatlon: ' + plotVar
                               )

    # Plot maps
    for jSet, plotId in enumerate(plotIdList):
        plt.subplot(gs[rowInds[jSet], colInds[jSet]])
        if diff_flag:
            im1, ax, compcont = plotlatlon(
                dsDict[plotId],
                plotVar,
                box_flag=box_flag[jSet],
                cbar_flag=False,
                compcont_flag=compcont_flag,
                diff_flag=True,
                diffDs=(dsDict[diffIdList[jSet]]
                        if diffDs is None
                        else diffDs),
                diffVar=diffVar,
                fontSize=fontSize,
                latLim=latLim,
                latlbls=latlbls,
                lonLim=lonLim,
                lonlbls=lonlbls,
                quiver_flag=quiver_flag,
                quiverScale=quiverScale,
                quiverUnits=quiverUnits,
                rmse_flag=rmse_flag,
                stampDate_flag=stampDate_flag,
                tSteps=tSteps,
                tStepLabel_flag=(jSet == 0),
                **kwargs
                )
        else:
            im1, ax, compcont = plotlatlon(
                dsDict[plotId],
                plotVar,
                box_flag=box_flag[jSet],
                cbar_flag=False,
                compcont_flag=compcont_flag,
                fontSize=fontSize,
                latLim=latLim,
                latlbls=latlbls,
                lonLim=lonLim,
                lonlbls=lonlbls,
                quiver_flag=quiver_flag,
                quiverScale=quiverScale,
                quiverUnits=quiverUnits,
                rmRegMean_flag=rmRegMean_flag,
                stampDate_flag=stampDate_flag,
                tSteps=tSteps,
                tStepLabel_flag=(jSet == 0),
                **kwargs
                )

        # Add subplot label (subfigure number)
        ax.annotate('(' + chr(jSet + ord(subFigCountStart)) + ')',
                    # xy=(-0.12, 1.09),
                    xy=(-0.08, 1.05),
                    xycoords='axes fraction',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    fontweight='bold',
                    )

    # Add common colorbar
    # Create axis for colorbar
    cbar_flag = True
    if cbar_flag:
        cbar_ax = plt.subplot(gs[cbRowInd:, cbColInd:])
        # cbar_ax = plt.subplot(gs[0:, 3:])

        # Create colorbar and set position
        if plotVar in ['PS']:
            hcb = plt.colorbar(im1,
                               cax=cbar_ax,
                               format='%0.0f',
                               orientation=cbarOrientation,
                               )
        else:
            hcb = plt.colorbar(im1,
                               cax=cbar_ax,
                               orientation=cbarOrientation,
                               )
        pcb = cbar_ax.get_position()
        # Create colorbar
        if cbarOrientation == 'vertical':
            # Place colorbar on figure
            cbar_ax.set_position([pcb.x0-0.01, pcb.y0 + pcb.height/6.,
                                  0.015, pcb.height*2./3.])

            # Label colorbar with variable name and units
            cbar_ax.set_ylabel(
                (r'$\Delta$' if diff_flag else '') +
                mwp.getplotvarstring(dsDict[plotIdList[0]][plotVar].name) +
                ' (' +
                mwfn.getstandardunitstring(
                    dsDict[plotIdList[0]][plotVar].units) +
                ')')

        elif cbarOrientation == 'horizontal':
            # Place colorbar on figure
            cbar_ax.set_position([pcb.x0, pcb.y0 - 0.015,
                                  pcb.width*1., 0.015])

            # Label colorbar with variable name and units
            cbar_ax.set_xlabel(
                (r'$\Delta$' if diff_flag else '') +
                mwp.getplotvarstring(dsDict[plotIdList[0]][plotVar].name) + ' (' +
                mwfn.getstandardunit(dsDict[plotIdList[0]][plotVar].units) +
                ')')

        # Add colorbar ticks and ensure compcont is labeled
        if (not diff_flag) and (plotVar == 'PRECT'):
            hcb.set_ticks(im1.levels[::2])
        else:
            try:
                hcb.set_ticks(im1.levels[::2]
                              if np.min(np.abs(im1.levels[::2] - compcont)) < 1e-10
                              else im1.levels[1::2])
            except TypeError:
                pass

        # Prevent colorbar from using offset
        # hcb.ax.yaxis.get_major_formatter().set_useOffset(False)

        # Plot bold contour on colorbar for reference
        if compcont_flag:
            try:
                boldLoc = ((compcont - im1.levels[0]) /
                           float((im1.levels[-1]) - im1.levels[0]))
                if cbarOrientation == 'vertical':
                    cbar_ax.hlines(boldLoc, 0, 1,
                                   colors='k',
                                   linewidth=2)
                elif cbarOrientation == 'horizontal':
                    cbar_ax.vlines(boldLoc, 0, 1,
                                   colors='k',
                                   linewidth=2)
            except TypeError:
                pass

    # Save figure if requested
    if save_flag:

        # Set directory for saving
        if saveDir is None:
            saveDir = os.path.dirname(os.path.realpath(__file__))

        # Set file name for saving
        tString = 'mon'
        if diff_flag:
            if all([diffIdList[j] == diffIdList[0]
                    for j in range(len(diffIdList))]):
                diffStr = 'd' + diffIdList[0][plotVar].srcid + '_'
            else:
                diffStr = ''

            saveFile = (plotVar +
                        '_latlon_comp{:d}_'.format(len(plotIdList)) +
                        diffStr + '_' +
                        tString +
                        '{:03.0f}'.format(tSteps[0]) + '-' +
                        '{:03.0f}'.format(tSteps[-1]))
        else:
            if len(plotIdList) > 3:
                caseSaveString = 'comp{:d}'.format(len(plotIdList))
            else:
                caseSaveString = '_'.join([plotIdList])
            saveFile = (
                plotVar + '_latlon_' +
                caseSaveString + '_' +
                tString +
                '{:03.0f}'.format(tSteps[0]) + '-' +
                '{:03.0f}'.format(tSteps[-1]) +
                ('_nocb' if not cbar_flag else '')
                )

        # Set saved figure size (inches)
        fx = hf.get_size_inches()[0]
        fy = hf.get_size_inches()[1]

        # Save figure
        print(saveDir + saveFile)
        mwp.savefig(saveDir + saveFile,
                    shape=np.array([fx, fy]))
        plt.close('all')


def plotmultizonregmean(dsDict,
                        plotIdList,
                        plotVar,
                        cbar_flag=True,
                        compcont_flag=False,
                        compcont=None,
                        diff_flag=False,
                        diffIdList=None,
                        diffDs=None,
                        diffVar=None,
                        fontSize=12,
                        gsEdges=None,  # list [L, R, B, T]
                        latLim=np.array([-30, 30]),
                        latlbls=None,
                        levels=None,
                        lonLim=np.array([120, 270]),
                        lonlbls=None,
                        ocnOnly_flag=False,
                        save_flag=False,
                        saveDir=None,
                        stampDate_flag=False,
                        stdUnits_flag=True,
                        subFigCountStart='a',
                        **kwargs
                        ):
    """
    Plot hovmollers of zonal means from multiple cases for comparison

    Version Date:
        2017-10-31
    """

    # Set lat/lon labels
    if latlbls is None:
        latlbls = mwp.getlatlbls(latLim)

    # Ensure diffIdList or diffDs provided if diff_flag
    if all([diff_flag, (diffIdList is None), (diffDs is None)]):
        raise ValueError('diffIdList or diffDs must be provided to plot' +
                         'differences')

    # Set variable for differencing; assumes plotVar if not provided
    if diffVar is None:
        diffVar = plotVar

    # Determine contour values if not provided
    if levels is None:
        levels = getzonmeancontlevels(plotVar,
                                      diff_flag=diff_flag)

    # Create figure for plotting
    hf = plt.figure()
    if len(plotIdList) == 9:
        # Set figure window size
        hf.set_size_inches(16.25, 7.75, forward=True)

        if gsEdges is None:
            # Edges of gridspec's outermost axes [L, R, B, T]
            gsEdges = [0.04, 0.97, 0.07, 0.97]

        # Set up subplots
        gs = gridspec.GridSpec(3, 4,
                               left=gsEdges[0],
                               right=gsEdges[1],
                               bottom=gsEdges[2],
                               top=gsEdges[3],
                               height_ratios=[1, 1, 1],
                               hspace=0.25,
                               width_ratios=[30, 30, 30, 1],
                               wspace=0.15)

        # Set gridspec colorbar location
        cbColInd = 3
        cbRowInd = 0

        # Set gridspec index order
        colInds = [0, 1, 2]*3
        rowInds = np.repeat(range(3), 3)

    # Plot hovmollers
    for jSet, plotId in enumerate(plotIdList):
        plt.subplot(gs[rowInds[jSet], colInds[jSet]])
        # Compute zonal mean
        zonMeanDa = mwfn.calcdaregzonmean(dsDict[plotId][plotVar],
                                          gwDa=(dsDict[plotId]['gw']
                                                if 'gw' in dsDict[plotId]
                                                else None),
                                          latLim=latLim,
                                          lonLim=lonLim,
                                          ocnOnly_flag=ocnOnly_flag,
                                          qc_flag=False,
                                          landFracDa=(
                                              dsDict[plotId]['LANDFRAC']
                                              if 'LANDFRAC' in dsDict[plotId]
                                              else None),
                                          stdUnits_flag=stdUnits_flag,
                                          )

        _, c1 = mwp.plotzonmean(np.concatenate((zonMeanDa.values,
                                                zonMeanDa.values[:1, :]),
                                               axis=0),
                                zonMeanDa.lat,
                                np.arange(1, 14),
                                cbar_flag=False,
                                conts=levels,
                                compcont=(compcont
                                          if compcont_flag
                                          else None),
                                dataId=plotId,
                                extend=['both', 'max'][
                                        1 if plotVar in ['PRECT', 'PRECL',
                                                         'PRECL']
                                        else 0],
                                grid_flag=True,
                                latLim=latLim,
                                varName=plotVar,
                                varUnits=zonMeanDa.units,
                                xticks=np.arange(1, 14),
                                xtickLabels=['J', 'F', 'M', 'A', 'M', 'J',
                                             'J', 'A', 'S', 'O', 'N', 'D',
                                             'J'],
                                )

        # Add longitude limits to first subplot
        if jSet == 0:
            ax = plt.gca()
            ax.annotate(r'$\theta$=[{:0d}, {:0d}]'.format(lonLim[0],
                                                          lonLim[-1]),
                        xy=(1, 1),
                        xycoords='axes fraction',
                        horizontalalignment='right',
                        verticalalignment='bottom'
                        )

        # Only label outermost axes
        if colInds[jSet] > 0:
            plt.ylabel('')
        if rowInds[jSet] < 2:
            plt.xlabel('')

    # Add colorbar
    if cbar_flag:
        # Create axis for colorbar
        cbar_ax = plt.subplot(gs[cbRowInd:, cbColInd:])

        # Create colorbar and set position
        hcb = plt.colorbar(c1,
                           cax=cbar_ax,
                           orientation='vertical'
                           )
        # Futz with colorbar position
        pcb = cbar_ax.get_position()
        cbar_ax.set_position([pcb.x0-0.01, pcb.y0 + pcb.height/6.,
                              0.015, pcb.height*2./3.])

        # Label colorbar with variable name and units
        cbar_ax.set_ylabel(
            (r'$\Delta$' if diff_flag else '') +
            mwp.getplotvarstring(dsDict[plotIdList[0]][plotVar].name) +
            ' (' +
            mwfn.getstandardunitstring(
                dsDict[plotIdList[0]][plotVar].units) +
            ')')

        # Add colorbar ticks
        # Add colorbar ticks and ensure compcont is labeled
        if (not diff_flag) and (plotVar == 'PRECT'):
            hcb.set_ticks(c1.levels[::2])
        else:
            try:
                hcb.set_ticks(
                    c1.levels[::2]
                    if np.min(np.abs(c1.levels[::2] - compcont)) < 1e-10
                    else c1.levels[1::2])
            except TypeError:
                pass

        if compcont is not None:
            boldLoc = ((compcont - c1.levels[0]) /
                       (c1.levels[-1] - c1.levels[0]))
            hcb.ax.hlines(boldLoc, 0, 1, colors='k', linewidth=1)

    # Add date of figure creation if requested
    if stampDate_flag:
        mwp.stampdate(x=1, y=0)

    # Save figure if requested
    if save_flag:

        # Set directory for saving
        if saveDir is None:
            saveDir = os.path.dirname(os.path.realpath(__file__))

        # Set file name for saving
        if diff_flag:
            if all([diffIdList[j] == diffIdList[0]
                    for j in range(len(diffIdList))]):
                diffStr = 'd' + diffIdList[0][plotVar].srcid + '_'
            else:
                diffStr = ''

            saveFile = ('d' + plotVar +
                        '_latlon_comp{:d}_'.format(len(plotIdList)) +
                        diffStr + '_' +
                        mwp.getlatlimstring(latLim, '') + '_' +
                        mwp.getlonlimstring(lonLim, '')
                        )
        else:
            if len(plotIdList) > 3:
                caseSaveString = 'comp{:d}'.format(len(plotIdList))
            else:
                caseSaveString = '_'.join([plotIdList])
            saveFile = (
                plotVar + '_zonmean_' +
                caseSaveString + '_' +
                mwp.getlatlimstring(latLim, '') + '_' +
                mwp.getlonlimstring(lonLim, '')
                )

        # Set saved figure size (inches)
        fx = hf.get_size_inches()[0]
        fy = hf.get_size_inches()[1]

        # Save figure
        # tprint(saveDir + saveFile)
        mwp.savefig(saveDir + saveFile,
                    shape=np.array([fx, fy]))
        plt.close('all')

# %%Do other stuff
