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

import mdwtools.mdwfunctions as mwfn  # import personal processing functions
import mdwtools.mdwplots as mwp       # import personal plotting functions
# import netCDF4 as nc4            # import netCDF4 as nc4
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


def calcregmeanindex(ds,
                     indexName,
                     indexType=None,
                     indexVar=None,
                     ocnOnly_flag=False,
                     qc_flag=False,
                     ):
    """
    Compute regional mean indices
    - CPacShear - central Pacific wind shear (850-200)
    - CTI - cold tongue index
    - dITCZ - double-ITCZ index
    - dSLP - Eq. Pacific SLP gradient (~Walker strength)
    - walker - Walker circulation index (based on pressure)
    """

    # Choose appropriate index to compute
    if indexName.lower() in ['cpacshear']:
        # Assign default index if none provided
        if indexType is None:
            indexType = 'testing'
        if indexVar is None:
            indexVar = 'U'

        # Compute central Pacific shear index
        indexDa = mwfn.calcdscpacshear(ds,
                                       indexType=indexType,
                                       indexVar=indexVar,
                                       )

    elif indexName.lower() in ['cti']:
        # Assign default index if none provided
        if indexType is None:
            indexType = 'Woelfleetal2017'
        if indexVar is None:
            indexVar = 'TS'

        # Compute cold tongue index
        indexDa = mwfn.calcdsctindex(ds,
                                     indexType=indexType,
                                     sstVar=indexVar,
                                     )

    elif indexName.lower() in ['ditcz']:
        # Assign default index if none provided
        if indexType is None:
            indexType = 'Bellucci2010'
        if indexVar is None:
            indexVar = 'PRECT'

        # Compute double-ITCZ index
        indexDa = mwfn.calcdsditczindex(ds,
                                        indexType='Bellucci2010',
                                        precipVar=indexVar,
                                        )

    elif indexName.lower() in ['walker', 'dslp']:
        # Assign default index if none provided
        if indexType is None:
            indexType = 'DiNezioetal2013'
        if indexVar is None:
            if indexType == 'DiNezioetal2013':
                indexVar = 'PSL'
            else:
                indexVar = 'PS'

        # Compute Walker circulation index
        indexDa = mwfn.calcdswalkerindex(ds,
                                         indexType=indexType,
                                         ocnOnly_flag=ocnOnly_flag,
                                         pressureVar=indexVar,
                                         )
    elif indexName.lower() in ['pai', 'precipasymmetry']:
        # Assigne default index if none provded
        if indexType is None:
            indexType = 'HwangFrierson2012'
        if indexVar is None:
            indexVar == 'PRECT'

        # Compute precipitation asymmetry index
        indexDa = mwfn.calcdsprecipasymindex(ds,
                                             indexType=indexType,
                                             precipVar=indexVar,
                                             qc_flag=qc_flag,
                                             )
    else:
        raise NameError('Cannot find function to compute ' +
                        'index: {:s}'.format(indexName))

    # Return index data array
    return indexDa


def getavailableyearslist(versionId):
    """
    Get list of averaging periods available for a given model version
    """
    return {'01': None,
            '28': ['2-10', '2-20', '50-74', '75-99'],
            '36': ['2-10', '2-20', '21-40', '60-60', '75-99'],
            'ga7.66': ['2-20', '20-39', '55-74'],
            '119': ['2-9', '2-21', '21-40', '30-49', '75-99'],
            '125': ['2-9', '2-21', '11-30', '21-40', '70-89', '80-99',
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
                        'U': np.array([0]),
                        'V': np.array([0]),
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
    if diff_flag:
        try:
            levels = {'FLNS': np.arange(-30., 30.1, 3),
                      # 'FNS': np.arange(-600., 600.1, 100),
                      'FNS': np.arange(-200, 200.1, 20),
                      'FSNS': np.arange(-50, 50.1, 5.),
                      'LHFLX': np.arange(-50, 50.1, 5),
                      'OMEGA': np.arange(-0.12, 0.12001, 0.01),
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
                      'U': np.arange(-5, 5.1, 0.5),
                      'V': np.arange(-2, 2.1, 0.2),
                      'U10': np.arange(-2, 2.1, 0.2),
                      'curlTau': np.arange(-1.5e-7, 1.51e-7, 1.5e-8),
                      'curlTau_y': np.arange(-4e-13, 4.01e-13, 4e-14),
                      'divTau': np.arange(-1e-7, 1.01e-7, 1e-8),
                      'ekmanx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'ekmany': np.arange(-1e4, 1.01e4, 1e3),
                      'precip': np.arange(-10, 10.1, 1),
                      'sst': np.arange(-2, 2.1, 0.2),
                      'sverdrupx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'MGx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'w': np.arange(-0.12, 0.12001, 0.01),
                      }[plotVar]
        except KeyError:
            levels = None
    else:
        try:
            levels = {'FLNS': np.arange(0., 120.1, 10),
                      'FNS': np.arange(-600., 600.1, 100),
                      'FSNS': np.arange(0, 400.1, 20.),
                      'LHFLX': np.arange(0, 200.1, 10),
                      'MGx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'OMEGA': np.arange(-0.12, 0.12001, 0.01),
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
                      'U': np.arange(-10, 10.1, 1.0),
                      'U10': np.arange(0, 10.1, 1),
                      'curlTau': np.arange(-3e-7, 3.001e-7, 3e-8),
                      'curlTau_y': np.arange(-4e-13, 4.01e-13, 4e-14),
                      'divTau': np.arange(-2e-7, 2.01e-7, 2e-8),
                      'ekmanx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'ekmany': np.arange(-3e4, 3.01e4, 3e3),
                      'precip': np.arange(0, 20.1, 2),
                      'sst': np.arange(290, 305, 1),
                      'sverdrupx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'w': np.arange(-0.12, 0.12001, 0.01),
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
                      'sp': np.arange(1004., 1013.1, 1),
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


def getyearsubdirs(versionId):
    """
    Get sudirectories for years as apparently formatting is inconsistent.
    """
    if versionId == '28':
        return ['yrs_2-10', 'yrs_2-20', 'yrs50-74', 'yrs75-99']
    else:
        yrIds = getavailableyearslist(versionId)
        return ['yrs_{:s}'.format(yid)
                for yid in yrIds]


def plotbiasrelation(ds,
                     xIndex,
                     yIndex,
                     ds_rg=None,  # For vertically regridded when needed
                     legend_flag=True,
                     makeFigure_flag=False,
                     obsDsDict=None,
                     plotObs_flag=True,
                     splitTSteps_flag=False,
                     tSteps=None,
                     tStepString=None,
                     versionIds=None,
                     xIndexType=None,
                     yIndexType=None,
                     xLim=None,
                     yLim=None,
                     xTSteps=None,
                     yTSteps=None,
                     ):
    """
    Plot scatterplot of one bias metric versus another averaged over a given
        time period
    """
    # Set version Ids to be plotted
    if versionIds is None:
        versionIds = list(ds.keys())

    # Months to include
    #   DJF - 0, 1, 11; MAM - 2, 3, 4; JJA - 5, 6, 7; SON - 8, 9, 10
    if tSteps is None:
        if tStepString is not None:
            try:
                tSteps = {'DJF': np.array([11, 0, 1]),
                          'MAM': np.array([2, 3, 4]),
                          'JJA': np.array([5, 6, 7]),
                          'SON': np.array([8, 9, 10]),
                          'Annual': np.arange(12)
                          }[tStepString]
            except KeyError:
                tSteps = np.arange(12)
        else:
            tSteps = np.arange(12)

    # Assign time steps for each index
    if not splitTSteps_flag:
        xTSteps = tSteps
        yTSteps = tSteps

    xMean = dict()
    yMean = dict()

    # Set index details
    xDs = (ds_rg if xIndex.lower() in ['cpacshear'] else ds)
    yDs = (ds_rg if yIndex.lower() in ['cpacshear'] else ds)
    indexTypes = {'cpacshear': 'testing',
                  'cti': 'Woelfleetal2017',
                  'ditcz': 'Bellucci2010',
                  'dslp': 'DiNezioetal2013',
                  'walker': 'testing'}
    if xIndexType is None:
        xIndexType = indexTypes[xIndex.lower()]
    if yIndexType is None:
        yIndexType = indexTypes[yIndex.lower()]
    indexVars = {'cpacshear': 'U',
                 'cti': 'TS',
                 'ditcz': 'PRECT',
                 'dslp': 'PSL',
                 'walker': 'PS'}
    labelDict = {'cpacshear': 'Central Pacific Wind Shear' +
                              ' (850-200 hPa; {:s})'.format(
                                  ds[versionIds[0]]['U'].units),
                 'cti': 'Cold Tongue Index (K)',
                 'ditcz': 'Double-ITCZ Index (mm/d)',
                 'dslp': 'SLP Gradient (hPa)',
                 'walker': 'Walker Circulation Index (hPa)'}
    if plotObs_flag:
        #        obsDsDict = {'cpacshear': obsDsDict['cpacshear'],
        #                   'cti': obsDsDict['cti'],
        #                   'ditcz': obsDsDict['ditcz'],
        #                   'walker': obsDsDict['walker']}
        obsVars = {'cpacshear': 'u',
                   'cti': 'sst',
                   'ditcz': 'precip',
                   'dslp': 'msl',
                   'walker': 'sp'}

    # Compute indices for various model versions
    for vid in versionIds:

        # Compute first index
        xIndexDa = calcregmeanindex(
            xDs[vid],
            xIndex,
            indexType=xIndexType,
            indexVar=indexVars[xIndex.lower()],
            ocnOnly_flag=False)
        xMean[vid] = xIndexDa[xTSteps].mean(dim='time')

        # Compute second index
        yIndexDa = calcregmeanindex(
            yDs[vid],
            yIndex,
            indexType=yIndexType,
            indexVar=indexVars[yIndex.lower()],
            ocnOnly_flag=False)
        yMean[vid] = yIndexDa[yTSteps].mean(dim='time')

    # Compute indices for observations
    #   (reference only; not in correlation)
    if plotObs_flag:
        # First index
        xObsDa = calcregmeanindex(
            obsDsDict[xIndex.lower()],
            xIndex,
            indexType=indexTypes[xIndex.lower()],
            indexVar=obsVars[xIndex.lower()],
            ocnOnly_flag=False)
        xMean['obs'] = xObsDa[xTSteps].mean(dim='time')
        # print(xMean['obs'].values)

        # Second index
        yObsDa = calcregmeanindex(
            obsDsDict[yIndex.lower()],
            yIndex,
            indexType=indexTypes[yIndex.lower()],
            indexVar=obsVars[yIndex.lower()],
            ocnOnly_flag=False)
        yMean['obs'] = yObsDa[yTSteps].mean(dim='time')
        # print(yMean['obs'].values)

    # Plot versus one another as scatter plot
    if makeFigure_flag:
        plt.figure()

    # Plot line to show version path through scatterplot
    plt.plot(np.array([xMean[vid] for vid in versionIds]),
             np.array([yMean[vid] for vid in versionIds]),
             c='k',
             label=None,
             zorder=1)

    for vid in versionIds:
        plt.scatter(xMean[vid],
                    yMean[vid],
                    marker='o',
                    s=80,
                    c=getcolordict()[vid],
                    label=vid,
                    zorder=2
                    )
    if plotObs_flag:
        plt.scatter(xMean['obs'],
                    yMean['obs'],
                    marker='^',
                    s=80,
                    c=getcolordict()['obs'],
                    label='Obs')

    # Compute correlation between cold tongue index and double-ITCZ index
    #   across model versions
    r = np.corrcoef(np.array([xMean[vid]
                              for vid in versionIds]),
                    np.array([yMean[vid]
                              for vid in versionIds]))[0, 1]

    # Add correlation to plot as annotation
    plt.gca().annotate(r'$\mathregular{r^2}$' + '={:0.3f}'.format(r),
                       xy=(1, 1),
                       xycoords='axes fraction',
                       horizontalalignment='right',
                       verticalalignment='bottom',
                       )

    # Determine string for labeling averaging time
    if splitTSteps_flag:
        tStepString = 'Split time Steps: x-XXX, y-XXX'
    else:
        if all([j in tSteps for j in np.arange(12)]):
            tStepString = 'Annual mean'
        else:
            monIds = ['J', 'F', 'M', 'A', 'M', 'J',
                      'J', 'A', 'S', 'O', 'N', 'D']
            try:
                tStepString = ''.join([monIds[tStep] for tStep in tSteps])
                # Correct JFD to DJF if necessary
                if tStepString == 'JFD':
                    tStepString = 'DJF'
                tStepString = tStepString + ' mean'
            except NameError:
                tStepString = '[Error]'

    # Add averaging time to plot as annotation
    if tStepString:
        plt.gca().annotate(tStepString,
                           xy=(0, 1),
                           xycoords='axes fraction',
                           horizontalalignment='left',
                           verticalalignment='bottom',
                           )

    # Label axes
    plt.xlabel(labelDict[xIndex.lower()])
    plt.ylabel(labelDict[yIndex.lower()])

    # Set axes limits
    if xLim is not None:
        plt.xlim(xLim)
    if yLim is not None:
        plt.ylim(yLim)

    # Add legend (if requested)
    if legend_flag:
        if len(versionIds) > 5:
            plt.legend(ncol=2)
        else:
            plt.legend()

    # Don't return anything for now.
    return


def plotlatlon(ds,
               plotVar,
               box_flag=False,
               boxLat=np.array([-20, 20]),
               boxLon=np.array([210, 260]),
               caseString=None,
               cbar_flag=True,
               cbar_dy=-0.1,
               cbar_height=0.02,
               cMap=None,
               compcont=None,
               compcont_flag=True,
               convertUnits_flag=True,
               diff_flag=False,
               diffDs=None,
               diffPlev=None,
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
               plev=None,
               qc_flag=False,
               quiver_flag=False,
               quiverDs=None,
               quiverDiffDs=None,
               quiverNorm_flag=False,
               quiverScale=0.4,
               quiverScaleVar=None,
               quiverUnits='inches',
               rmRegLatLim=None,
               rmRegLonLim=None,
               rmRegMean_flag=False,
               rmse_flag=False,
               stampDate_flag=True,
               subSamp=None,
               tSteps=None,
               tStepLabel_flag=True,
               uRef=0.1,
               uVar='TAUX',
               vVar='TAUY',
               verbose_flag=False,
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

    # Determine differencing pressure level if not provided
    if diffPlev is None:
        diffPlev = plev

    # Set caseString for plotting
    if caseString is None:
        if diff_flag:
            caseString = ds.id + '-' + diffDs.id
        else:
            caseString = ds.id

    # Ensure diffVar is defined as non-None
    if diffVar is None:
        diffVar = plotVar

    # Define quiver datasets if needed
    if quiverDs is None:
        quiverDs = ds
    if quiverDiffDs is None:
        quiverDiffDs = diffDs

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

    # Pull data for plotting contours
    if np.ndim(ds[plotVar]) == 3:
        if diff_flag:
            pData = (ds[plotVar].values[tSteps, :, :].mean(axis=0) -
                     diffDs[diffVar].values[diffTSteps, :, :].mean(axis=0))
        else:
            pData = ds[plotVar].values[tSteps, :, :].mean(axis=0)
    elif np.ndim(ds[plotVar]) == 4:
        jPlev = int(np.arange(len(ds.plev)
                              )[ds.plev.values == plev])
        if diff_flag:
            kPlev = int(np.arange(len(diffDs.plev)
                                  )[diffDs.plev.values == diffPlev])
            pData = (
                ds[plotVar].values[tSteps, jPlev, :, :].mean(axis=0) -
                diffDs[diffVar].values[diffTSteps, kPlev, :, :].mean(axis=0))
        else:
            pData = ds[plotVar].values[tSteps, jPlev, :, :].mean(axis=0)

    # Pull data for plotting vectors (if needed)
    if quiver_flag:
        if np.ndim(quiverDs[uVar]) == 3:
            if diff_flag:
                uData = (
                    quiverDs[uVar].data[tSteps, :, :].mean(axis=0) -
                    quiverDiffDs[uVar].data[diffTSteps, :, :].mean(axis=0))
                vData = (
                    quiverDs[vVar].data[tSteps, :, :].mean(axis=0) -
                    quiverDiffDs[vVar].data[diffTSteps, :, :].mean(axis=0))
                if quiverScaleVar is not None:
                    quivSc = (
                        quiverDs[quiverScaleVar].values[tSteps, :, :
                                                        ].mean(axis=0) -
                        quiverDiffDs[quiverScaleVar].values[diffTSteps, :, :
                                                            ].mean(axis=0))
            else:
                uData = quiverDs[uVar].data[tSteps, :, :].mean(axis=0)
                vData = quiverDs[vVar].data[tSteps, :, :].mean(axis=0)
                if quiverScaleVar is not None:
                    quivSc = (quiverDs[quiverScaleVar].values[tSteps, :, :
                                                              ].mean(axis=0))
        elif np.ndim(quiverDs[uVar]) == 4:
            jPlev = int(np.arange(len(quiverDs.plev)
                                  )[quiverDs.plev.values == plev])
            if diff_flag:
                kPlev = int(np.arange(len(quiverDiffDs.plev)
                                      )[quiverDiffDs.plev.values == diffPlev])
                uData = (
                    quiverDs[uVar].data[tSteps, jPlev, :, :].mean(axis=0) -
                    quiverDiffDs[uVar].data[diffTSteps, kPlev, :, :
                                            ].mean(axis=0))
                vData = (
                    quiverDs[vVar].data[tSteps, jPlev, :, :].mean(axis=0) -
                    quiverDiffDs[vVar].data[diffTSteps, kPlev, :, :
                                            ].mean(axis=0))
            else:
                uData = quiverDs[uVar].data[tSteps, jPlev, :, :].mean(axis=0)
                vData = quiverDs[vVar].data[tSteps, jPlev, :, :].mean(axis=0)
    else:
        uData = None
        vData = None

    # Normalize and/or rescale uData and vData if requested
    if quiverScaleVar is not None:
        quiverNorm_flag = False
        normMagnitude = ((uData**2 + vData**2)**0.5)
        # Normalize u and scale so that vector length is quivSc
        uData = uData/normMagnitude*np.abs(quivSc)
        # Normalize u and scale so that vector length is quivSc
        vData = vData/normMagnitude*np.abs(quivSc)
    if quiverNorm_flag:
        print('Norming quiver')
        normMagnitude = ((uData**2 + vData**2)**0.5)
        uData = uData/normMagnitude
        vData = vData/normMagnitude

    # Flip vectors if using wind stress (?)
    if all([quiver_flag,
            uVar == 'TAUX',
            vVar == 'TAUY']):
        print('flipping quivers')
        uData = -uData
        vData = -vData

    # Compute and subtract off regional mean if requested
    if rmRegMean_flag:
        if verbose_flag:
            print('Removing regional mean over ' +
                  'lat={:0d},{:0d} and '.format(rmRegLatLim[0],
                                                rmRegLatLim[-1]) +
                  'lon={:0d},{:0d}'.format(rmRegLonLim[0],
                                           rmRegLonLim[-1]))
        # Get averaging lat/lon limits if not explicity provided
        if rmRegLatLim is None:
            rmRegLatLim = latLim
        if rmRegLonLim is None:
            rmRegLonLim = lonLim

        # Compute regional mean through time
        regMeanDs = mwfn.calcdaregmean(ds[plotVar],
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
        # print(regMeanDs.values)
        # Compute time mean regional mean to be subtracted
        regMean = regMeanDs.values[tSteps].mean(axis=0)

        # Subtract off regional mean
        try:
            pData = pData - regMean
        except ValueError:
            jPlev = int(np.arange(len(ds.plev))[ds.plev.values == plev])
            regMean = regMean[jPlev]
            pData = pData - regMean

        if diff_flag:
            # Compute regional mean for subtracting off (differenced values)
            dregMeanDs = mwfn.calcdaregmean(diffDs[diffVar],
                                            gwDa=(diffDs['gw']
                                                  if 'gw' in diffDs
                                                  else None),
                                            latLim=rmRegLatLim,
                                            lonLim=rmRegLonLim,
                                            ocnOnly_flag=ocnOnly_flag,
                                            landFracDa=(diffDs['LANDFRAC']
                                                        if 'LANDFRAC' in diffDs
                                                        else None),
                                            qc_flag=qc_flag,
                                            stdUnits_flag=False,
                                            )
            dregMean = dregMeanDs.values[tSteps].mean(axis=0)

            # "Subtract off" regional mean from differenced values
            #    a' - b' = (a - a_bar) - (b - b_bar) = a - a_bar - b + b_bar
            try:
                pData = pData + dregMean
            except ValueError:
                kPlev = int(np.arange(len(diffDs.plev)
                                      )[diffDs.plev.values == diffPlev])
                dregMean = dregMean[kPlev]
                pData = pData + dregMean

        if qc_flag:
            print(regMean)

    # Set variables to only extend one end of colorbar
    maxExtendVars = ['PRECT', 'PRECC', 'PRECL', 'precip', 'U10']

    # if plotting difference, extend both ends for all variables
    if diff_flag:
        maxExtendVars = []

    # Set variable name for labeling colorbar
    varName = mwp.getplotvarstring(ds[plotVar].name)
    if np.ndim(ds[plotVar]) == 4:
        # Add level if original field is 4d
        varName = varName + str(plev)

        # Add differencing of levels if plotting differences and
        #   plev and diffPlev are not the same (for plotting shears)
        if all([diff_flag, plev != diffPlev]):
            varName = varName + '-' + str(diffPlev)

    # Plot map
    im1, ax = mwp.plotmap(ds.lon,
                          ds.lat,
                          pData,
                          box_flag=box_flag,
                          boxLat=boxLat,
                          boxLon=boxLon,
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
                          varName=varName,
                          varUnits=ds[plotVar].units,
                          quiver_flag=quiver_flag,
                          quiverKey_flag=(not quiverNorm_flag),
                          quiverScale=quiverScale,
                          quiverUnits=quiverUnits,
                          U=uData,
                          Uname=(ds[uVar].name
                                 if uVar in ds.data_vars
                                 else None),
                          Uunits=((ds[quiverScaleVar].units
                                   if quiverScaleVar is not None else
                                   ds[uVar].units)
                                  if uVar in ds.data_vars
                                  else None),
                          Uref=uRef,  # 0.1,
                          V=vData,
                          subSamp=((3 if subSamp is None else subSamp)
                                   # ds['TAUX'].shape[1]/36
                                   if uVar in ds.data_vars
                                   else None),
                          tStepLabel_flag=False,
                          **kwargs
                          )

    # Add removed regional mean to annotations
    if rmRegMean_flag:
        if diff_flag:
            rmMeanString = (
                '\nRm Reg Mean = {:0.1f} {:s}'.format(regMean - dregMean,
                                                      ds[plotVar].units))
        else:
            rmMeanString = (
                '\nRm Reg Mean = {:0.1f} {:s}'.format(regMean,
                                                      ds[plotVar].units))
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
                    diffPlev=None,
                    diffVar=None,
                    fontSize=12,
                    latLim=np.array([-30, 30]),
                    latlbls=None,
                    lonLim=np.array([119.5, 270.5]),
                    lonlbls=None,
                    plev=None,
                    quiver_flag=False,
                    quiverScale=0.4,
                    quiverUnits='inches',
                    rmse_flag=False,
                    # rmRegMean_flag=False,
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
                                   width_ratios=[30, 1],
                                   )

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
            if np.diff(latLim) >= 50:
                hf.set_size_inches(16.25, 6.75, forward=True)
            else:
                hf.set_size_inches(16.25, 5.75, forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(3, 4,
                                   height_ratios=[1, 1, 1],
                                   hspace=0.00,
                                   width_ratios=[30, 30, 30, 1],
                                   wspace=0.2,
                                   left=0.04,
                                   right=0.96,
                                   bottom=0.00,
                                   top=1.0,
                                   )

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
                                   wspace=0.5,
                                   )

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
                diff_flag=diff_flag,
                diffDs=(dsDict[diffIdList[jSet]]
                        if any([diffDs is None,
                                diffDs == dsDict])
                        else diffDs),
                diffPlev=diffPlev,
                diffVar=diffVar,
                fontSize=fontSize,
                latLim=latLim,
                latlbls=latlbls,
                lonLim=lonLim,
                lonlbls=lonlbls,
                plev=plev,
                quiver_flag=quiver_flag,
                quiverScale=quiverScale,
                quiverUnits=quiverUnits,
                # rmRegLatLim=rmRegLatLim,
                # rmRegLonLim=rmRegLonLim,
                # rmRegMean_flag=rmRegMean_flag,
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
                diff_flag=False,
                fontSize=fontSize,
                latLim=latLim,
                latlbls=latlbls,
                lonLim=lonLim,
                lonlbls=lonlbls,
                plev=plev,
                quiver_flag=quiver_flag,
                quiverScale=quiverScale,
                quiverUnits=quiverUnits,
                # rmRegMean_flag=rmRegMean_flag,
                stampDate_flag=stampDate_flag,
                tSteps=tSteps,
                tStepLabel_flag=(jSet == 0),
                **kwargs
                )

        # Add subplot label (subfigure number)
        ax.annotate('(' + chr(jSet + ord(subFigCountStart)) + ')',
                    # xy=(-0.12, 1.09),
                    xy=(-0.08, 1.07),
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

        # Get variable name for colorbar label
        varName = mwp.getplotvarstring(dsDict[plotIdList[0]][plotVar].name)
        if np.ndim(dsDict[plotIdList[0]][plotVar]) == 4:
            # Add level if original field is 4d
            varName = varName + str(plev)
        if all([diff_flag, plev != diffPlev]):
            # Add differencing of levels if plotting differences and
            #   plev and diffPlev are not the same (for plotting shears)
            varName = varName + '-' + str(diffPlev)

        # Create colorbar
        if cbarOrientation == 'vertical':
            # Place colorbar on figure
            cbar_ax.set_position([pcb.x0-0.01, pcb.y0 + pcb.height/6.,
                                  0.015, pcb.height*2./3.])

            # Label colorbar with variable name and units
            cbar_ax.set_ylabel(
                (r'$\Delta$' if diff_flag else '') +
                varName + ' (' +
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
                varName + ' (' +
                mwfn.getstandardunit(dsDict[plotIdList[0]][plotVar].units) +
                ')')

        # Add colorbar ticks and ensure compcont is labeled
        if (not diff_flag) and (plotVar == 'PRECT'):
            hcb.set_ticks(im1.levels[::2])
        else:
            try:
                hcb.set_ticks(im1.levels[::2]
                              if np.min(np.abs(im1.levels[::2] - compcont)
                                        ) < 1e-10
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

    # Expand plot(s) to fill figure window
    # gs.tight_layout(hf)

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
            # Get variable name for saving
            varName = plotVar
            if np.ndim(dsDict[plotIdList[0]][plotVar]) == 4:
                # Add level if original field is 4d
                varName = varName + str(plev)
            if plev != diffPlev:
                # Add differencing of levels if plotting differences and
                #   plev and diffPlev are not the same (for plotting shears)
                varName = varName + '-' + str(diffPlev)
            saveFile = (varName +
                        '_latlon_comp{:d}_'.format(len(plotIdList)) +
                        diffStr +
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

    elif len(plotIdList) == 4:
        # Set figure window size
        hf.set_size_inches(16.25, 7.75, forward=True)

        if gsEdges is None:
            # Edges of gridspec's outermost axes [L, R, B, T]
            gsEdges = [0.04, 0.97, 0.07, 0.97]

        # Set up subplots
        gs = gridspec.GridSpec(2, 3,
                               left=gsEdges[0],
                               right=gsEdges[1],
                               bottom=gsEdges[2],
                               top=gsEdges[3],
                               height_ratios=[1, 1],
                               hspace=0.25,
                               width_ratios=[30, 30, 1],
                               wspace=0.15)

        # Set gridspec colorbar location
        cbColInd = 2
        cbRowInd = 0

        # Set gridspec index order
        colInds = [0, 1]*3
        rowInds = np.repeat(range(2), 2)

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


def plotmultizonregmeanlines(dsDict,
                             plotIdList,
                             plotVar,
                             colorDict=None,
                             diff_flag=False,
                             diffIdList=None,
                             diffDs=None,
                             diffVar=None,
                             fontSize=12,
                             gsEdges=None,  # list [L, R, B, T]
                             latLim=np.array([-30, 30]),
                             latlbls=None,
                             legend_flag=True,
                             levels=None,
                             lonLim=np.array([120, 270]),
                             lonlbls=None,
                             lw=2,
                             obsDs=None,
                             obsVar=None,
                             ocnOnly_flag=False,
                             plotLatLim=None,
                             plotObs_flag=False,
                             save_flag=False,
                             saveDir=None,
                             stampDate_flag=False,
                             stdUnits_flag=True,
                             subFigCountStart='a',
                             **kwargs
                             ):
    """
    Plot lines of time mean, zonal means from multiple cases for comparison

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

    # Set obs properties
    if all([plotObs_flag, obsVar is None]):
        obsVar = {'PRECT': 'precip',
                  }[plotVar]

    # Set limits for x axis
    if plotLatLim is None:
        plotLatLim = latLim

    # Create figure for plotting
    hf = plt.figure()

    # Set figure window size
    hf.set_size_inches(10, 3.5, forward=True)

    if gsEdges is None:
        # Edges of gridspec's outermost axes [L, R, B, T]
        # gsEdges = [0.04, 0.97, 0.07, 0.97]
        gsEdges = [0.04, 0.97, 0.07, 0.97]

    # Set up subplots
    gs = gridspec.GridSpec(1, 2,
                           left=gsEdges[0],
                           right=gsEdges[1],
                           bottom=gsEdges[2],
                           top=gsEdges[3],
                           # height_ratios=[1, 1],
                           hspace=0.25,
                           width_ratios=[5, 1],
                           wspace=0.15,
                           )

    # Set gridspec legend location
    # legColInd = 1
    # legRowInd = 0

    # Plot lines
    plt.subplot(gs[0, 0])
    for jSet, plotId in enumerate(plotIdList):
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

        try:
            lineColor = colorDict[plotId]
        except (KeyError, TypeError):
            lineColor = None

        plt.plot(zonMeanDa.lat, zonMeanDa.mean(dim='time'),
                 color=lineColor,
                 label=plotId,
                 lw=lw)

    if plotObs_flag:
        zonMeanObsDa = mwfn.calcdaregzonmean(obsDs[obsVar],
                                             gwDa=None,
                                             latLim=latLim,
                                             lonLim=lonLim,
                                             ocnOnly_flag=ocnOnly_flag,
                                             qc_flag=False,
                                             landFracDa=None,
                                             stdUnits_flag=stdUnits_flag,
                                             )
        try:
            lineColor = colorDict['obs']
        except (KeyError, TypeError):
            lineColor = [0, 0, 0]

        plt.plot(zonMeanObsDa.lat, zonMeanObsDa.mean(dim='time'),
                 color=lineColor,
                 label='obs',
                 lw=lw)

    # Add longitude limits
    ax = plt.gca()
    ax.annotate(r'$\theta$=[{:0d}, {:0d}]'.format(lonLim[0],
                                                  lonLim[-1]),
                xy=(1, 1),
                xycoords='axes fraction',
                horizontalalignment='right',
                verticalalignment='bottom'
                )

    # Dress axes
    plt.xlim(plotLatLim)
    plt.xlabel('Latitude')
    plt.ylabel('{:s} ({:s})'.format(mwp.getplotvarstring(plotVar),
                                    zonMeanObsDa.units))

    # Add legend
    if legend_flag:

        # Create legend and set position
        plt.legend(  # hl, lName,
                   bbox_to_anchor=(1.05, 0.5), loc=6, borderaxespad=1.)

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
                plotVar + '_zonmeanlines_' +
                caseSaveString + '_' +
                mwp.getlatlimstring(latLim, '') + '_' +
                mwp.getlonlimstring(lonLim, '')
                )

        # Set saved figure size (inches)
        fx = hf.get_size_inches()[0]
        fy = hf.get_size_inches()[1]

        # Save figure
        print(saveDir + saveFile)
        mwp.savefig(saveDir + saveFile,
                    shape=np.array([fx, fy]))
        plt.close('all')


# %%Do other stuff
