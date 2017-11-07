#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:48:52 2017

@author: woelfle
"""

# %% Import modules as needed

import numpy as np  # for handling arrays
# import pandas as pd  # for handling 2d things
import xarray as xr  # for handling nd things (netcdfs)

# from scipy import interpolate    # interpolation functions

import matplotlib.pyplot as plt  # for plotting things

from socket import gethostname   # used to determine which machine we are
#                                #   running on

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

    elif gethostname()[0:6] in ['yslogi', 'geyser']:
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


# %% Main section
if __name__ == '__main__':

    # Set options/flags
    diff_flag = False
    loadErai_flag = True  # True to load ERAI fields
    loadGpcp_flag = True
    loadHadIsst_flag = True
    obs_flag = True
    ocnOnly_flag = True  # Need to implement to confirm CTindex is right.
    plotBiasRelation_flag = True
    plotOneMap_flag = False
    plotMultiMap_flag = False
    plotGpcpTest_flag = False
    plotRegMean_flag = False
    plotZonRegMeanHov_flag = False
    prect_flag = True
    reload_flag = False
    save_flag = False
    saveSubDir = 'testfigs/'
    verbose_flag = False

    # Set new variables to compute when loading
    newVars = 'PRECT'

    # Get directory of file to load
    ncDir, ncSubDir, saveDir = setfilepaths()

    # Set name(s) of file(s) to load
    versionIds = ['01',
                  '28', '36',
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
    for versionId in versionIds:
        dataSets[versionId].attrs['id'] = versionId

    if any([obs_flag, plotGpcpTest_flag, loadGpcp_flag, loadHadIsst_flag]):

        if loadGpcp_flag or plotGpcpTest_flag:
            # # Load GPCP

            # Set directories for GPCP
            gpcpDir = '/home/disk/eos9/woelfle/dataset/GPCP/climo/'
            gpcpFile = 'gpcp_197901-201012.nc'
            gpcpClimoFile = 'gpcp_197901-201012_climo.nc'

            # Load GPCP for all years and add id
            if plotGpcpTest_flag:
                gpcpDs = xr.open_dataset(gpcpDir + gpcpFile)
                gpcpDs.attrs['id'] = 'GPCP_all'

            # Load GPCP from both climo and add id
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
                                         whichHad='pd_monclimo',
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

    # Set variable of interest
    plotVars = ['PRECC']  # , 'TS', 'TAUX']

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

# %% Plot one map
    # set plotting parameters
    latLim = np.array([-30, 30])
    lonLim = np.array([119.5, 270.5])

    latLbls = np.arange(-30, 31, 10)
    lonLbls = np.arange(120, 271, 30)

    tSteps = np.arange(0, 12)

    if plotOneMap_flag:
        plotVar = 'sp'

        # Create figure for plotting
        hf = plt.figure()

        # Plot some fields for comparison
        c1to2p.plotlatlon(eraiDs,
                          plotVar,
                          box_flag=False,
                          caseString=None,
                          cbar_flag=True,
                          cbar_dy=0.001,
                          cbar_height=0.02,
                          compcont_flag=True,
                          diff_flag=False,
                          diffDs=hadIsstDs,  # gpcpClimoDs,
                          fontSize=12,
                          latLim=np.array([-20, 20]),
                          levels=np.arange(-5, 5.1, 1),
                          lonLim=np.array([119.5, 270.5]),
                          quiver_flag=False,
                          rmRegMean_flag=True,
                          stampDate_flag=False,
                          tSteps=np.arange(0, 12),
                          tStepLabel_flag=True,
                          uVar='TAUX',
                          vVar='TAUY',
                          )
        # Save figure if requested
        if save_flag:
            # Set directory for saving
            if saveDir is None:
                saveDir = setfilepaths()[2]

            # Set file name for saving
            tString = 'mon'
            saveFile = (plotVar + '_latlon_' +
                        tString +
                        '{:03.0f}'.format(tSteps[0]) + '-' +
                        '{:03.0f}'.format(tSteps[-1]))

            # Set saved figure size (inches)
            fx = hf.get_size_inches()[0]
            fy = hf.get_size_inches()[1]

            # Save figure
            print(saveDir + saveFile)
            mwp.savefig(saveDir + saveSubDir + saveFile,
                        shape=np.array([fx, fy]))
            plt.close('all')

# %% Load and plot GPCP as a test

    if plotGpcpTest_flag:
        # Create figure for plotting
        hf = plt.figure()

        # Plot some fields for comparison
        for jMon, monStart in enumerate(np.array([0, 3, 6, 9])):
            plt.subplot(2, 2, jMon + 1)
            c1to2p.plotlatlon(gpcpDs,
                              'precip',
                              box_flag=False,
                              caseString=None,
                              cbar_flag=True,
                              cbar_dy=0.001,
                              cbar_height=0.02,
                              compcont_flag=True,
                              diff_flag=True,
                              diffDs=gpcpClimoDs,
                              diffTSteps=np.arange(
                                  monStart,
                                  gpcpClimoDs['precip'].shape[0],
                                  12),
                              fontSize=12,
                              latLim=np.array([-30, 30]),
                              lonLim=np.array([119.5, 270.5]),
                              quiver_flag=False,
                              stampDate_flag=False,
                              tSteps=np.arange(monStart,
                                               gpcpDs['precip'].shape[0],
                                               12),
                              tStepLabel_flag=True,
                              uVar='TAUX',
                              vVar='TAUY',
                              )

# %% Plot multiple maps
    if plotMultiMap_flag:
        plotVars = ['PS']
        for plotVar in plotVars:
            c1to2p.plotmultilatlon(dataSets,
                                   versionIds,
                                   plotVar,
                                   box_flag=False,
                                   cbar_flag=True,
                                   cbarOrientation='vertical',
                                   compcont_flag=True,
                                   diff_flag=diff_flag,
                                   diffIdList=None,  # ['01']*9,
                                   diffDs=hadIsstDs,
                                   diffVar='sst',
                                   fontSize=24,
                                   latLim=np.array([-20.1, 20.1]),
                                   latlbls=None,
                                   levels=np.arange(-5, 5.1, 1),
                                   lonLim=np.array([119.5, 270.5]),
                                   lonlbls=None,
                                   ocnOnly_flag=True,
                                   quiver_flag=False,
                                   quiverScale=0.4,
                                   quiverUnits='inches',
                                   rmRegLatLim=np.array([-20, 20]),
                                   rmRegLonLim=np.array([119.5, 270.5]),
                                   rmRegMean_flag=True,
                                   rmse_flag=False,
                                   save_flag=save_flag,
                                   saveDir=setfilepaths()[2] + saveSubDir,
                                   stampDate_flag=False,
                                   subFigCountStart='a',
                                   tSteps=np.arange(0, 12),
                                   )

# %% Plot regional means

    if plotRegMean_flag:

        # Set variable for plotting
        plotVar = 'PS'

        if plotVar in ['PRECT', 'PRECL', 'PRECC']:
            yLim = np.array([1, 2.7])
            rmRefRegMean_flag = False
            latLim = np.array([-20, 0])
            lonLim = np.array([210, 260])
            obsDs = gpcpClimoDs
            obsVar = 'precip'
            ocnOnly_flag = False
            title = '2xITCZ Index'
        elif plotVar in ['PS']:
            rmRefRegMean_flag = True
            latLim = np.array([-5, 5])
            lonLim = np.array([240, 270])
            refLatLim = np.array([-5, 5])
            refLonLim = np.array([150, 180])
            obsDs = eraiDs
            obsVar = 'sp'
            ocnOnly_flag = False
            title = 'Pressure gradient for Walker (E-W)'
            yLim = None
        elif plotVar in ['TS']:
            # Set flags
            rmRefRegMean_flag = True
            # Set lat/lon limits
            latLim = np.array([-3, 3])
            lonLim = np.array([180, 220])
            refLatLim = np.array([-20, 20])
            refLonLim = np.array([150, 250])
            if rmRefRegMean_flag:
                yLim = np.array([-2.0, 1.0])
            else:
                yLim = np.array([297, 301])
            obsDs = hadIsstDs
            obsVar = 'sst'
            ocnOnly_flag = True
            title = 'Cold Tongue Index'
        else:
            yLim = None

        # Create dictionary to hold annual mean value (and colors)
        annMean = dict()
        colorDict = dict()

        # Create figure for plotting
        plt.figure()

        # Compute regional mean over double-ITCZ region as defined in
        #   Bellucci et al. (2010, J Clim)
        # latLim = np.array([-20, 0])
        # lonLim = np.array([210, 260])
        # latLim = np.array([-3, 3])
        # lonLim = np.array([180, 220])
        # latLim = np.array([-20, 20])
        # lonLim = np.array([150, 250])

        for vid in versionIds:
            # Compute regional mean through time
            regMeanDs = mwfn.calcdaregmean(dataSets[vid][plotVar],
                                           gwDa=dataSets[vid]['gw'],
                                           latLim=latLim,
                                           lonLim=lonLim,
                                           ocnOnly_flag=ocnOnly_flag,
                                           qc_flag=False,
                                           landFracDa=(
                                               dataSets[vid]['LANDFRAC']),
                                           stdUnits_flag=True,
                                           )

            # Compute reference regional mean if needed
            if rmRefRegMean_flag:
                refRegMeanDs = mwfn.calcdaregmean(
                    dataSets[vid][plotVar],
                    gwDa=dataSets[vid]['gw'],
                    latLim=refLatLim,
                    lonLim=refLonLim,
                    ocnOnly_flag=ocnOnly_flag,
                    landFracDa=dataSets[vid]['LANDFRAC'],
                    qc_flag=False,
                    stdUnits_flag=True,
                    )
                regMeanDs = regMeanDs - refRegMeanDs

            # Plot regional mean through time
            hl, = plt.plot(np.arange(1, 13),
                           regMeanDs.values,
                           label=vid,
                           marker='o',
                           )
            annMean[vid] = regMeanDs.mean(dim='time')
            colorDict[vid] = hl.get_color()

        # Repeat above for obs

        obsRegMeanDs = mwfn.calcdaregmean(obsDs[obsVar],
                                          gwDa=None,
                                          latLim=latLim,
                                          lonLim=lonLim,
                                          stdUnits_flag=True,
                                          )
        if rmRefRegMean_flag:
            obsRefRegMeanDs = mwfn.calcdaregmean(obsDs[obsVar],
                                                 gwDa=None,
                                                 latLim=refLatLim,
                                                 lonLim=refLonLim,
                                                 stdUnits_flag=True,
                                                 )

            obsRegMeanDs = obsRegMeanDs - obsRefRegMeanDs

        hl, = plt.plot(np.arange(1, 13),
                       obsRegMeanDs.values,
                       lw=2,
                       c=[0, 0, 0],
                       label=obsDs.id,
                       marker='^'
                       )

        annMean['obs'] = obsRegMeanDs.mean(dim='time')
        colorDict['obs'] = hl.get_color()

        plt.xticks(np.arange(1, 13))
        plt.xlabel('Month')

        plt.ylabel(plotVar + ' (' +
                   mwp.getlatlimstring(latLim) + ', ' +
                   mwp.getlonlimstring(lonLim, lonFormat='EW') + ')'
                   )
        plt.ylim(yLim)

        plt.legend(title='Version')

        # plt.title('Seasonal cycle of 2xITCZ index')
        plt.title('Seasonal cycle of {:s}'.format(title))

        # Plot annual mean values
        plt.figure()

        plt.scatter(np.arange(1, len(annMean)),
                    np.array([annMean[j] for j in versionIds]),
                    marker='o',
                    c=[colorDict[j] for j in versionIds],
                    s=80,
                    )
        plt.scatter([len(annMean)],
                    annMean['obs'],
                    marker='^',
                    c=colorDict['obs'],
                    s=80,
                    )

        plt.xticks(np.arange(1, len(annMean) + 1),
                   versionIds + ['obs'])
        plt.xlabel('Version')

        plt.ylabel(plotVar + ' (' +
                   mwp.getlatlimstring(latLim) + ', ' +
                   mwp.getlonlimstring(lonLim, lonFormat='EW') + ')'
                   )
        plt.ylim(yLim)

        plt.grid(ls='--')
        plt.gca().set_axisbelow(True)

        plt.title('Annual mean {:s}'.format(title))

    # Plot index through model versions
    # Use scatter plot with version on x-axis and index on y-axis

    # Change x-axis labels to be model version rather than generic number

# %% Correlate bias indices (CTI, dTICZ, Walker)
    if plotBiasRelation_flag:

        # Options: 'CTI', 'dITCZ', 'Walker'
        index1 = 'CTI'  # x axis
        index2 = 'dITCZ'  # y axis

        annMean1 = dict()
        annMean2 = dict()

        # Set index details
        indexTypes = {'cti': 'Woelfleetal2017',
                      'ditcz': 'Bellucci2010',
                      'walker': 'testing'}
        indexVars = {'cti': 'TS',
                     'ditcz': 'PRECT',
                     'walker': 'PS'}
        labelDict = {'cti': 'Cold Tongue Index (K)',
                     'ditcz': 'Double-ITCZ Index (mm/d)',
                     'walker': 'Walker Circulation Index (hPa)'}
        obsDict = {'cti': hadIsstDs,
                   'ditcz': gpcpClimoDs,
                   'walker': eraiDs}
        obsVars = {'cti': 'sst',
                   'ditcz': 'precip',
                   'walker': 'sp'}

        # Compute indices for various model versions
        for vid in versionIds:

            # Compute first index
            index1Da = c1to2p.calcregmeanindex(
                dataSets[vid],
                index1,
                indexType=indexTypes[index1.lower()],
                indexVar=indexVars[index1.lower()],
                ocnOnly_flag=False)
            annMean1[vid] = index1Da.mean(dim='time')

            # Compute second index
            index2Da = c1to2p.calcregmeanindex(
                dataSets[vid],
                index2,
                indexType=indexTypes[index2.lower()],
                indexVar=indexVars[index2.lower()],
                ocnOnly_flag=False)
            annMean2[vid] = index2Da.mean(dim='time')

        # Compute indices for observations
        #   (reference only; not in correlation)
        obs1Da = c1to2p.calcregmeanindex(
            obsDict[index1.lower()],
            index1,
            indexType=indexTypes[index1.lower()],
            indexVar=obsVars[index1.lower()],
            ocnOnly_flag=False)
        annMean1['obs'] = obs1Da.mean(dim='time')

        obs2Da = c1to2p.calcregmeanindex(
            obsDict[index2.lower()],
            index2,
            indexType=indexTypes[index2.lower()],
            indexVar=obsVars[index2.lower()],
            ocnOnly_flag=False)
        annMean2['obs'] = obs2Da.mean(dim='time')

        # Plot versus one another as scatter plot
        plt.figure()

        # Plot line to show version path through scatterplot
        plt.plot(np.array([annMean1[vid] for vid in versionIds]),
                 np.array([annMean2[vid] for vid in versionIds]),
                 c='k',
                 label=None,
                 zorder=1)

        for vid in versionIds:
            plt.scatter(annMean1[vid],
                        annMean2[vid],
                        marker='o',
                        s=80,
                        c=getcolordict()[vid],
                        label=vid,
                        zorder=2
                        )
        plt.scatter(annMean1['obs'],
                    annMean2['obs'],
                    marker='^',
                    s=80,
                    c=getcolordict()['obs'],
                    label='Obs')

        # Compute correlation between cold tongue index and double-ITCZ index
        #   across model versions
        r = np.corrcoef(np.array([annMean1[vid]
                                  for vid in versionIds]),
                        np.array([annMean2[vid]
                                  for vid in versionIds]))[0, 1]

        # Add correlation to plot as annotation
        plt.gca().annotate(r'$\mathregular{r^2}$' + '={:0.3f}'.format(r),
                           xy=(1, 1),
                           xycoords='axes fraction',
                           horizontalalignment='right',
                           verticalalignment='bottom'
                           )

        # Dress plot
        plt.xlabel(labelDict[index1.lower()])
        plt.ylabel(labelDict[index2.lower()])
        plt.legend()

# %% Plot zonal mean hovmoller
    if plotZonRegMeanHov_flag:

        # Set flags and options
        latLim = np.array([-20, 20])
        lonLim = np.array([180, 220])

        plotVar = 'PS'

        # Plot versions
        c1to2p.plotmultizonregmean(dataSets,
                                   versionIds,
                                   plotVar,
                                   cbar_flag=True,
                                   compcont_flag=True,
                                   compcont=np.array([300.]),
                                   diff_flag=False,
                                   diffIdList=None,
                                   diffDs=None,
                                   diffVar=None,
                                   fontSize=12,
                                   latLim=latLim,
                                   latlbls=None,
                                   lonLim=lonLim,
                                   ocnOnly_flag=False,
                                   save_flag=save_flag,
                                   saveDir=setfilepaths()[2] + saveSubDir,
                                   stdUnits_flag=True,
                                   subFigCountStart='a',
                                   )

        # Plot obs for reference
        if obs_flag:
            if plotVar in ['PRECT', 'PRECC', 'PRECL']:
                obsDs = gpcpClimoDs
                obsVar = 'precip'
            elif plotVar in ['TS']:
                obsDs = hadIsstDs
                obsVar = 'sst'
            elif plotVar in ['PS']:
                obsDs = eraiDs
                obsVar = 'sp'

            hf = plt.figure()
            hf.set_size_inches(7.05, 2.58,
                               forward=True)

            zonMeanObsDa = mwfn.calcdaregzonmean(obsDs[obsVar],
                                                 gwDa=None,
                                                 latLim=latLim,
                                                 lonLim=lonLim,
                                                 ocnOnly_flag=False,
                                                 qc_flag=False,
                                                 landFracDa=None,
                                                 stdUnits_flag=True,
                                                 )

            mwp.plotzonmean(np.concatenate((zonMeanObsDa.values,
                                            zonMeanObsDa.values[:1, :]),
                                           axis=0),
                            zonMeanObsDa.lat,
                            np.arange(1, 14),
                            cbar_flag=True,
                            conts=c1to2p.getzonmeancontlevels(obsVar),
                            compcont=np.array([300]),
                            dataId=obsDs.id,
                            extend=['both', 'max'][
                                1 if plotVar in ['PRECT', 'PRECL', 'PRECL']
                                else 0],
                            grid_flag=True,
                            latLim=latLim,
                            varName=plotVar,
                            varUnits=zonMeanObsDa.units,
                            xticks=np.arange(1, 14),
                            xtickLabels=['J', 'F', 'M', 'A', 'M', 'J',
                                         'J', 'A', 'S', 'O', 'N', 'D',
                                         'J'],
                            )

            ax = plt.gca()
            ax.annotate(r'$\theta$=[{:0d}, {:0d}]'.format(lonLim[0],
                                                          lonLim[-1]),
                        xy=(1, 1),
                        xycoords='axes fraction',
                        horizontalalignment='right',
                        verticalalignment='bottom'
                        )

            if save_flag:
                mwp.savefig(setfilepaths()[2] +
                            saveSubDir + plotVar + '_zonmean_' +
                            mwp.getlatlimstring(latLim, '') + '_' +
                            mwp.getlonlimstring(lonLim, '') +
                            'obs')
