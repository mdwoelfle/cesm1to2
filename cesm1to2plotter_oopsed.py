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
import matplotlib.gridspec as gridspec  # for subplot management

from socket import gethostname   # used to determine which machine we are
#                                #   running on

import multiprocessing as mp  # Allow use of multiple cores
import datetime  # For keeping track of run times

from mdwtools import mdwfunctions as mwfn  # For averaging things
from mdwtools import mdwplots as mwp  # For plotting things

import cesm1to2plotter as c1to2p

# import matplotlib.cm as cm
# from scipy.stats import linregress

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


def getquiverprops(uVar,
                   vVar,
                   plev=None,
                   diff_flag=None,
                   ):
    """
    Return properties for quiver plots based on which fields are being plotted
    """
    if diff_flag:
        if all([uVar == 'TAUX',
                vVar == 'TAUY']):
            quiverScale = 0.1
            uRef = 0.03
        elif all([uVar == 'U',
                  vVar == 'V']):
            if plev == 850:
                quiverScale = 10
                uRef = 1
            elif plev == 200:
                quiverScale = 10
                uRef = 1
            else:
                quiverScale = 10
                uRef = 1
    else:
        if all([uVar == 'TAUX',
                vVar == 'TAUY']):
            quiverScale = 0.3
            uRef = 0.05
        elif all([uVar == 'U',
                  vVar == 'V']):
            if plev == 850:
                quiverScale = 40
                uRef = 5
            elif plev == 200:
                quiverScale = 40
                uRef = 5
            else:
                quiverScale = 40
                uRef = 5

    return {'quiverScale': quiverScale,
            'uRef': uRef,
            'Uref': uRef}


# %% Main section
if __name__ == '__main__':

    # Set options/flags
    diff_flag = False
    loadErai_flag = False  # True to load ERAI fields
    loadGpcp_flag = False
    loadHadIsst_flag = False
    mp_flag = True  # True to use multiprocessing when regridding
    obs_flag = False
    ocnOnly_flag = True  # Need to implement to confirm CTindex is right.
    plotBiasRelation_flag = False
    plotOneMap_flag = False
    plotMultiMap_flag = False
    plotGpcpTest_flag = False
    plotRegMean_flag = False
    plotSeasonalBiasRelation_flag = False
    plotZonRegMeanHov_flag = False
    prect_flag = True
    regridVertical_flag = True
    reload_flag = False
    save_flag = False
    saveSubDir = 'testfigs/'
    testPlot_flag = True
    verbose_flag = False

    # Set new variables to compute when loading
    newVars = 'PRECT'

    # Get directory of file to load
    ncDir, ncSubDir, saveDir = setfilepaths()

    # Set name(s) of file(s) to load
    versionIds = [  # '01',
                  # '28',  '36',
                  # 'ga7.66',
                  '119', '125',
                  # '161', '194', '195'
                  ]
    fileBases = [  # 'b.e15.B1850G.f09_g16.pi_control.01',
                 # 'b.e15.B1850G.f09_g16.pi_control.28',
                 # 'b.e15.B1850.f09_g16.pi_control.36',
                 # 'b.e15.B1850.f09_g16.pi_control.all_ga7.66',
                 'b.e15.B1850.f09_g16.pi_control.all.119',
                 'b.e20.B1850.f09_g16.pi_control.all.125',
                 # 'b.e20.BHIST.f09_g17.20thC.161_01',
                 # 'b.e20.B1850.f09_g17.pi_control.all.194',
                 # 'b.e20.B1850.f09_g17.pi_control.all.195',
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
            hadIsstYrs = [1979, 2010]
            # Attempt to look at other averaging periods for HadISST
            hadIsstDs = mwfn.loadhadisst(climoType='monthly',
                                         daNewGrid=None,
                                         kind='linear',
                                         newGridFile=None,
                                         newGridName='0.9x1.25',
                                         newLat=None,
                                         newLon=None,
                                         qc_flag=False,
                                         regrid_flag=True,
                                         whichHad='all',  # 'pd_monclimo'
                                         years=hadIsstYrs,
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
    plotVars = ['TS']  # , 'TS', 'TAUX']

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

# %% Regrid 3D fields

    # Regridding with multiprocessing
    if all([mp_flag,
            regridVertical_flag,
            any([reload_flag, load_flag])]):

        print('>> Regridding vertical levels <<')

        # Set new levels for regridding
        #   > Timingk works out to about 30s per level per case (seems long...)
        newLevs = np.array([200, 300, 400, 500, 600, 700, 775, 850, 900, 950])
        regridVars = ['U', 'V', 'OMEGA']
        regridStartTime = datetime.datetime.now()
        print(regridStartTime.strftime('--> Regrid start time: %X'))

        for jVar, regridVar in enumerate(regridVars):  # = 'U'
            print('--> Regridding {:s}'.format(regridVar))
            startTime = datetime.datetime.now()

            # Regrid 3D variables using multiprocessing
            mpPool = mp.Pool(8)

            # Get list of cases for unpacking later
            versionIds = versionIds

            # Load all datasets to memory to enable multiprocessing
            for vid in versionIds:
                dataSets[vid][regridVar].load()
                dataSets[vid]['PS'].load()
                dataSets[vid]['hyam'].load()
                dataSets[vid]['hybm'].load()

            # Create input tuple for regridding to pressure levels
            mpInList = [(dataSets[vid],
                         regridVar,
                         newLevs,
                         'cesm')
                        for vid in versionIds]

            # Call multiprocessing of regridding
            # regriddedVars = mpPool.map(mwfn.regriddssigmatopres_mp,
            #                            mpInList)
            regriddedVars = mpPool.map_async(mwfn.regriddssigmatopres_mp,
                                             mpInList)

            # Close multiprocessing pool
            regriddedVars = regriddedVars.get()
            mpPool.close()
            mpPool.terminate()  # Not proper,
            #                   #    but may be needed to work properly
            mpPool.join()

            print('\n##------------------------------##')
            print('Time to regrid with mp:')
            print(datetime.datetime.now() - startTime)
            print('##------------------------------##\n')

            # Unpack regriddedVars
            if jVar == 0:
                dataSets_rg = dict()
                for jVid, vid in enumerate(versionIds):
                    dataSets_rg[vid] = \
                        regriddedVars[jVid].to_dataset(name=regridVar)
                    dataSets_rg[vid].attrs['id'] = regriddedVars[jVid].id
            else:
                for jVid, vid in enumerate(versionIds):
                    dataSets_rg[vid] = xr.merge(
                        [dataSets_rg[vid],
                         regriddedVars[jVid].to_dataset(name=regridVar)])
                    dataSets_rg[vid].attrs['id'] = regriddedVars[jVid].id

        # Write out total time required to do regridding
        print('\n-->Total regrid time:')
        print(datetime.datetime.now() - regridStartTime)
        print('\n')

    elif all([regridVertical_flag,
              any([reload_flag, load_flag])]):
        """
        DEPRECATED
        """
        startTime = datetime.datetime.now()
        # Set levels for regridding
        newLevs = np.array([800, 900])

        # Set variable to regrid
        regridVar = 'U'

        # Regrid dataarray
        # Benchmark for timing
        newDas = dict()
        for vid in versionIds:
            print('---Processing {:s}---'.format(vid))
            newDas[vid] = mwfn.regriddssigmatopres(dataSets[vid],
                                                   regridVar,
                                                   newLevs,
                                                   modelId='cesm',
                                                   verbose_flag=True)

        print('\n##------------------------------##')
        print('Time to regrid with loop:')
        print(datetime.datetime.now() - startTime)
        print('##------------------------------##\n')


# %% Plot one map
    # set plotting parameters
    latLim = np.array([-30, 30])
    lonLim = np.array([119.5, 270.5])

    latLbls = np.arange(-30, 31, 10)
    lonLbls = np.arange(120, 271, 30)

    tSteps = np.arange(0, 12)

    if plotOneMap_flag:

        plotVar = 'PRECT'
        uVar = 'U'
        vVar = 'V'
        plev = 950
        diffPlev = 950  # plev
        diff_flag = True  # False
        plotCase = '125'  # '125'
        diffCase = '119'  # '119'

        # Create figure for plotting
        hf = plt.figure()

        # Get quiver properties
        quiverProps = getquiverprops(uVar, vVar, plev,
                                     diff_flag=diff_flag)

        # Plot some fields for comparison
        c1to2p.plotlatlon((dataSets_rg[plotCase]
                           if plotVar in dataSets_rg[plotCase]
                           else dataSets[plotCase]),  # hadIsstDs
                          plotVar,
                          box_flag=False,
                          boxLat=np.array([-3, 3]),
                          boxLon=np.array([180, 220]),
                          caseString=None,
                          cbar_flag=True,
                          # cbar_dy=0.001,
                          cbar_height=0.02,
                          cMap=None,  # 'RdBu_r',
                          compcont_flag=True,
                          diff_flag=diff_flag,
                          diffDs=(dataSets_rg[diffCase]
                                  if plotVar in dataSets_rg[diffCase]
                                  else dataSets[diffCase]),  # gpcpClimoDs,
                          diffPlev=diffPlev,
                          fontSize=12,
                          latLim=np.array([-20, 20]),
                          levels=None,  # np.arange(-15, 15.1, 1.5),
                          lonLim=np.array([119.5, 270.5]),
                          plev=plev,
                          quiver_flag=True,  # True,
                          quiverDs=(dataSets_rg[plotCase]
                                    if uVar in dataSets_rg[plotCase]
                                    else dataSets[plotCase]),
                          quiverDiffDs=(dataSets_rg[diffCase]
                                        if uVar in dataSets_rg[diffCase]
                                        else dataSets[diffCase]),
                          quiverNorm_flag=False,
                          quiverScale=quiverProps['quiverScale'],
                          quiverScaleVar=None,
                          rmRegMean_flag=False,
                          stampDate_flag=False,
                          tSteps=tSteps,
                          tStepLabel_flag=True,
                          uRef=quiverProps['uRef'],
                          uVar=uVar,
                          vVar=vVar,
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
            fx, fy = hf.get_size_inches()

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
        plotVars = ['U10']
        for plotVar in plotVars:
            c1to2p.plotmultilatlon(dataSets,
                                   versionIds,
                                   plotVar,
                                   box_flag=False,
                                   boxLat=np.array([-20, 0]),
                                   boxLon=np.array([210, 260]),
                                   cbar_flag=True,
                                   cbarOrientation='vertical',
                                   compcont_flag=True,
                                   diff_flag=False,
                                   diffIdList=versionIds,
                                   diffDs=dataSets,
                                   diffPlev=200,
                                   diffVar='U',
                                   fontSize=24,
                                   latLim=np.array([-30.1, 30.1]),
                                   latlbls=None,
                                   levels=None,  # np.arange(-20, 20.1, 2),
                                   lonLim=np.array([119.5, 270.5]),
                                   lonlbls=None,
                                   ocnOnly_flag=False,
                                   plev=850,
                                   quiver_flag=True,
                                   quiverNorm_flag=True,
                                   quiverScale=5,
                                   quiverUnits='inches',
                                   rmRegLatLim=np.array([-20, 20]),
                                   rmRegLonLim=np.array([119.5, 270.5]),
                                   rmRegMean_flag=False,
                                   rmse_flag=False,
                                   save_flag=save_flag,
                                   saveDir=setfilepaths()[2] + saveSubDir,
                                   stampDate_flag=False,
                                   subFigCountStart='a',
                                   subSamp=7,
                                   tSteps=np.arange(0, 12),
                                   uVar='TAUX',
                                   vVar='TAUY',
                                   )

# %% Plot regional means (biases)

    if plotRegMean_flag:

        # Set variable for plotting
        plotVar = 'TS'

        # Set plotting flags and specifications
        rmAnnMean_flag = True
        plotAnnMean_flag = False
        plotPeriodMean_flag = False
        # tSteps = np.arange(1, 5)
        tSteps = np.append(np.arange(5, 12), 0)
        divideByTropMean_flag = False
        tropLatLim = np.array([-30, 30])
        tropLonLim = np.array([0, 360])

        # Set default plot values
        title = mwp.getplotvarstring(plotVar)
        yLim = None

        if plotVar in ['PRECT', 'PRECL', 'PRECC']:
            ds = dataSets
            yLim = np.array(([0, 2] if divideByTropMean_flag
                             else [0.5, 4.7]))
            rmRefRegMean_flag = False
            plotObs_flag = True
            latLim = np.array([-20, 0])
            lonLim = np.array([210, 260])
            obsDs = gpcpClimoDs
            obsVar = 'precip'
            ocnOnly_flag = False
            title = '2xITCZ Index'
        elif plotVar in ['PS']:
            ds = dataSets
            rmRefRegMean_flag = True
            plotObs_flag = True
            latLim = np.array([-5, 5])
            lonLim = np.array([240, 270])
            refLatLim = np.array([-5, 5])
            refLonLim = np.array([150, 180])
            obsDs = eraiDs
            obsVar = 'sp'
            ocnOnly_flag = False
            title = 'Pressure gradient for Walker (E-W)'
            yLim = None
        elif plotVar in ['PSL']:
            ds = dataSets
            rmRefRegMean_flag = True
            plotObs_flag = True
            latLim = np.array([0, 10])  # -5, 5])
            lonLim = np.array([210, 260])  # 200, 280])
            refLatLim = np.array([-10, 0])  # -5, 5])
            refLonLim = np.array([210, 260])  # 100, 180])
            obsDs = eraiDs
            obsVar = 'msl'
            ocnOnly_flag = False
            title = 'Pressure gradient for dITCZ (N-S)'  # Walker (E-W)'
            yLim = None
        elif plotVar in ['TS']:
            ds = dataSets
            # Set flags
            rmRefRegMean_flag = True  # True
            plotObs_flag = True
            # Set lat/lon limits
            latLim = np.array([-3, 3])  # 0, 10])
            lonLim = np.array([180, 220])  # 210, 260])
            refLatLim = np.array([-20, 20])  # -10, 0])
            refLonLim = np.array([150, 250])  # 210, 260])
            if rmRefRegMean_flag or rmAnnMean_flag:
                yLim = np.array([-2.5, 2.5])
            else:
                yLim = np.array([297, 301])
            obsDs = hadIsstDs
            obsVar = 'sst'
            ocnOnly_flag = True
            title = 'dITCZ Region'  # 'Cold Tongue Index'
        elif plotVar in ['U']:
            ds = dataSets_rg
            plotObs_flag = True
            obsDs = eraiDs
            obsVar = 'u'
            plev = 850
            diffPlev = 200
            latLim = np.array([-5, 5])
            lonLim = np.array([180, 220])
            rmRefRegMean_flag = True
            refLatLim = latLim
            refLonLim = lonLim
            ocnOnly_flag = False
            title = 'U'

        # Create dictionary to hold annual mean value (and colors)
        annMean = dict()
        timeMean = dict()
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
            regMeanDa = mwfn.calcdaregmean(ds[vid][plotVar],
                                           gwDa=dataSets[vid]['gw'],
                                           latLim=latLim,
                                           lonLim=lonLim,
                                           ocnOnly_flag=ocnOnly_flag,
                                           qc_flag=False,
                                           landFracDa=(
                                               dataSets[vid]['LANDFRAC']),
                                           stdUnits_flag=True,
                                           )

            if divideByTropMean_flag:
                tropMeanDa = mwfn.calcdaregmean(ds[vid][plotVar],
                                                gwDa=dataSets[vid]['gw'],
                                                latLim=tropLatLim,
                                                lonLim=tropLonLim,
                                                ocnOnly_flag=ocnOnly_flag,
                                                qc_flag=False,
                                                landFracDa=(
                                                    dataSets[vid]['LANDFRAC']),
                                                stdUnits_flag=True,
                                                )

            if np.ndim(regMeanDa) == 2:
                # Subset to level(s) of interest
                regMeanDa = regMeanDa.loc[dict(plev=slice(plev, plev))]

                try:
                    tropMeanDa = tropMeanDa.loc[dict(plev=slice(plev, plev))]
                except NameError:
                    pass
                # Update title on first go 'round
                title = (title + '{:d}'.format(plev)
                         if '{:d}'.format(plev) not in title
                         else title)

            # Compute reference regional mean if needed
            if rmRefRegMean_flag:
                refRegMeanDa = mwfn.calcdaregmean(
                    ds[vid][plotVar],
                    gwDa=dataSets[vid]['gw'],
                    latLim=refLatLim,
                    lonLim=refLonLim,
                    ocnOnly_flag=ocnOnly_flag,
                    landFracDa=dataSets[vid]['LANDFRAC'],
                    qc_flag=False,
                    stdUnits_flag=True,
                    )
                # Subtract off reference regional mean, but allow for this to
                #   be a difference between levels as opposed to a difference
                #   of regions
                if np.ndim(refRegMeanDa) == 2:
                    refRegMeanDa = refRegMeanDa.loc[
                        dict(plev=slice(diffPlev, diffPlev))]
                    refRegMeanDa['plev'].values = np.array([plev])
                    regMeanDa = regMeanDa - refRegMeanDa
                    if all([vid == versionIds[0],
                            plev != diffPlev]):
                        title = title + '-{:d}'.format(diffPlev)
                else:
                    regMeanDa = regMeanDa - refRegMeanDa

            # Pull regional mean through time and plot
            if divideByTropMean_flag:
                regMeanDa = regMeanDa/tropMeanDa
            pData = (regMeanDa.values - regMeanDa.mean(dim='time').values
                     if rmAnnMean_flag
                     else regMeanDa.values)
            hl, = plt.plot(np.arange(1, 13),
                           pData,
                           label=vid,
                           marker='o',
                           )
            annMean[vid] = regMeanDa.mean(dim='time')
            timeMean[vid] = regMeanDa.values[tSteps].mean()
            colorDict[vid] = hl.get_color()

        # Repeat above for obs
        if plotObs_flag:
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
            try:
                pData = (obsRegMeanDa.values -
                         obsRegMeanDa.mean(dim='time').values
                         if rmAnnMean_flag
                         else obsRegMeanDa.values)
            except ValueError:
                pData = (obsRegMeanDa.values -
                         obsRegMeanDa.mean(dim='month').values
                         if rmAnnMean_flag
                         else obsRegMeanDa.values)

            hl, = plt.plot(np.arange(1, 13),
                           pData,
                           lw=2,
                           c=[0, 0, 0],
                           label=obsDs.id,
                           marker='^'
                           )
            try:
                annMean['obs'] = obsRegMeanDa.mean(dim='time')
            except ValueError:
                annMean['obs'] = obsRegMeanDa.mean(dim='month')
            timeMean['obs'] = obsRegMeanDa.values[tSteps].mean()
            colorDict['obs'] = hl.get_color()

        plt.xticks(np.arange(1, 13))
        plt.xlabel('Month')

        plt.ylabel(plotVar + ' (' +
                   mwp.getlatlimstring(latLim) + ', ' +
                   mwp.getlonlimstring(lonLim, lonFormat='EW') +
                   ((' minus \n' +
                     mwp.getlatlimstring(refLatLim) + ', ' +
                     mwp.getlonlimstring(refLonLim, lonFormat='EW')
                     ) if rmRefRegMean_flag else '') +
                   ')' +
                   ('\n[Annual mean removed]' if rmAnnMean_flag else '')
                   )
        plt.ylim(yLim)

        plt.legend(title='Version', ncol=2)

        # plt.title('Seasonal cycle of 2xITCZ index')
        plt.title('Seasonal cycle of {:s}'.format(title) +
                  ('\n(divided by Tropical Mean)' if divideByTropMean_flag
                   else '') +
                  ('\n[Annual mean removed]' if rmAnnMean_flag else '')
                  )
        # Add annotation of years used to compute HadISST climatology
        if all([plotVar == 'TS', plotObs_flag]):
            plt.annotate('(obs: {:d}-{:d})'.format(hadIsstYrs[0],
                                                   hadIsstYrs[1]),
                         xy=(1, 1),
                         xycoords='axes fraction',
                         horizontalalignment='right',
                         verticalalignment='bottom',
                         )

        # Add grid
        plt.grid()

        plt.tight_layout()

        # Plot annual mean values
        if plotAnnMean_flag:
            plt.figure()
            # print([annMean[j].values for j in versionIds])
            plt.scatter(np.arange(1, len(annMean) +
                                  (0 if plotObs_flag else 1)),
                        np.array([annMean[j] for j in versionIds]),
                        marker='o',
                        c=[colorDict[j] for j in versionIds],
                        s=80,
                        )
            if 'obs' in annMean.keys():
                # print(annMean['obs'])
                plt.scatter([len(annMean)],
                            annMean['obs'],
                            marker='^',
                            c=colorDict['obs'],
                            s=80,
                            )

            plt.xticks(np.arange(1, len(annMean) + 1),
                       ((versionIds + ['obs'])
                        if 'obs' in annMean.keys()
                        else versionIds))
            plt.xlabel('Version')

            plt.ylabel(plotVar + ' (' +
                       mwp.getlatlimstring(latLim) + ', ' +
                       mwp.getlonlimstring(lonLim, lonFormat='EW') +
                       ((' minus \n' +
                         mwp.getlatlimstring(refLatLim) + ', ' +
                         mwp.getlonlimstring(lonLim, lonFormat='EW')
                         ) if rmRefRegMean_flag else '') +
                       ')'
                       )
            plt.ylim(yLim)

            plt.grid(ls='--')
            plt.gca().set_axisbelow(True)

            plt.title('Annual mean {:s}'.format(title) +
                      ('\n(divided by Tropical Mean)' if divideByTropMean_flag
                       else '')
                      )

        # Plot time mean values
        if plotPeriodMean_flag:
            plt.figure()

            plt.scatter(np.arange(1, len(timeMean) +
                                  (0 if plotObs_flag else 1)),
                        np.array([timeMean[j] for j in versionIds]),
                        marker='o',
                        c=[colorDict[j] for j in versionIds],
                        s=80,
                        )
            if 'obs' in timeMean.keys():
                plt.scatter([len(timeMean)],
                            timeMean['obs'],
                            marker='^',
                            c=colorDict['obs'],
                            s=80,
                            )

            plt.xticks(np.arange(1, len(timeMean) + 1),
                       ((versionIds + ['obs'])
                        if 'obs' in timeMean.keys()
                        else versionIds))
            plt.xlabel('Version')

            plt.ylabel(plotVar + ' (' +
                       mwp.getlatlimstring(latLim) + ', ' +
                       mwp.getlonlimstring(lonLim, lonFormat='EW') +
                       ((' minus \n' +
                         mwp.getlatlimstring(refLatLim) + ', ' +
                         mwp.getlonlimstring(lonLim, lonFormat='EW')
                         ) if rmRefRegMean_flag else '') +
                       ')'
                       )
            plt.ylim(yLim)

            plt.grid(ls='--')
            plt.gca().set_axisbelow(True)

            monIds = ['J', 'F', 'M', 'A', 'M', 'J',
                      'J', 'A', 'S', 'O', 'N', 'D']
            tStepString = ''.join([monIds[tStep] for tStep in tSteps])
            if tStepString == 'JFD':
                tStepString = 'DJF'
            plt.title('{:s} mean {:s}'.format(tStepString, title) +
                      ('\n(divided by Tropical Mean)' if divideByTropMean_flag
                       else '')
                      )

    # Plot index through model versions
    # Use scatter plot with version on x-axis and index on y-axis

    # Change x-axis labels to be model version rather than generic number

# %% Correlate bias indices (CTI, dTICZ, Walker)
    if plotBiasRelation_flag:

        # True to use different time steps for x and y axes
        splitTSteps_flag = False

        c1to2p.plotbiasrelation(dataSets,
                                'dSLP',
                                'dITCZ',
                                ds_rg=None,  # dataSets_rg
                                legend_flag=True,
                                makeFigure_flag=True,
                                obsDsDict={'cpacshear': erai3dDs,
                                           'cti': hadIsstDs,
                                           'ditcz': gpcpClimoDs,
                                           'dslp': eraiDs,
                                           'walker': eraiDs},
                                plotObs_flag=True,
                                splitTSteps_flag=splitTSteps_flag,
                                # tSteps=np.arange(8, 11),
                                tSteps=np.arange(12),
                                tStepString=None,  # 'Annual',
                                xIndexType=None,
                                yIndexType=None,
                                xTSteps=np.roll(np.arange(12), -1),
                                yTSteps=np.arange(12),
                                )

    if plotSeasonalBiasRelation_flag:

        xIndex = 'CTI'
        yIndex = 'dITCZ'

        matchLimits_flag = False
        axisLimitDict = {'cpacshear': np.array([-5, -25]),
                         'cti': np.array([-2, 0.5]),
                         'ditcz': np.array([0.5, 5]),
                         'walker': np.array([1, 4.5]),
                         }

        hf = plt.figure()
        hf.set_size_inches(10, 7)
        gs = gridspec.GridSpec(2, 2,
                               hspace=(0.17 if matchLimits_flag
                                       else 0.27),
                               wspace=(0.15 if matchLimits_flag
                                       else 0.22)
                               )
        rowInds = [0, 0, 1, 1]
        colInds = [0, 1, 0, 1]
        tStepStrings = ['DJF', 'MAM', 'JJA', 'SON']

        for j in np.arange(len(tStepStrings)):
            plt.subplot(gs[rowInds[j], colInds[j]])
            c1to2p.plotbiasrelation(dataSets,
                                    xIndex,
                                    yIndex,
                                    ds_rg=dataSets_rg,
                                    legend_flag=(j == 3),
                                    makeFigure_flag=False,
                                    obsDsDict={'cpacshear': erai3dDs,
                                               'cti': hadIsstDs,
                                               'ditcz': gpcpClimoDs,
                                               'walker': eraiDs},
                                    plotObs_flag=True,
                                    tSteps=None,
                                    tStepString=tStepStrings[j],
                                    xIndexType=None,
                                    yIndexType=None,
                                    xLim=(axisLimitDict[xIndex.lower()]
                                          if matchLimits_flag else None),
                                    yLim=(axisLimitDict[yIndex.lower()]
                                          if matchLimits_flag else None),
                                    )
            if matchLimits_flag:
                if rowInds[j] != max(rowInds):
                    plt.xlabel('')
                if colInds[j] != 0:
                    plt.ylabel('')
        gs.tight_layout(hf)

        if save_flag:
            # Set directory for saving
            if saveDir is None:
                saveDir = setfilepaths()[2]

            # Set file name for saving
            tString = 'mon'
            saveFile = ('seasonal_{:s}_vs_{:s}'.format(xIndex, yIndex))

            # Set saved figure size (inches)
            fx, fy = hf.get_size_inches()

            # Save figure
            print(saveDir + saveFile)
            mwp.savefig(saveDir + saveSubDir + saveFile,
                        shape=np.array([fx, fy])
                        )
            plt.close(hf)



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

# %% Plot pressure-latitude figure with vectors (potentially)

    if testPlot_flag:
        # Set variable to plot
        plotVar = 'OMEGA'
        plotCase = '119'

        # Set plotting limits
        latLim = np.array([-30, 30])
        lonLim = np.array([210, 260])
        pLim = np.array([1000, 200])
        tLim = np.array([0, 12])
        dt = 1

        # Set other parameters
        quiver_flag = True

        # Compute meridional mean over requested longitudes
        a = dataSets_rg[plotCase].loc[
            dict(lon=slice(lonLim[0], lonLim[-1]),
                 lat=slice(latLim[0]-2, latLim[-1]+2))
            ].mean(dim='lon')

        # Mean data over requested plotting time period
        a = a.isel(time=slice(tLim[0], tLim[-1], dt)).mean(dim='time')

        # Get contours for plotting
        try:
            conts = {'OMEGA': np.arange(-0.1, 0.101, 0.01),
                     'V': np.arange(-4, 4.1, 0.5),
                     }[plotVar]
        except KeyError:
            conts = None

        # Create figure for plotting
        hf = plt.figure()

        # Plot meridional mean slice
        cset1 = plt.contourf(a['lat'],
                             a['plev'],
                             a[plotVar],
                             conts,
                             cmap='RdBu_r',
                             extend='both')

        # Add vectors
        if quiver_flag:
            plt.quiver(a['lat'],
                       a['plev'],
                       a['V'],
                       -a['OMEGA']*10,
                       pivot='tail',
                       # units=quiverUnits,
                       # scale=quiverScale
                       )

        # Dress plot
        ax = plt.gca()
        # Flip y direction
        ax.invert_yaxis()

        # Set x and y limits
        plt.xlim(latLim)
        plt.ylim(pLim)

        # Label axes
        plt.xlabel('Latitude')
        plt.ylabel('Pressure ({:s})'.format(a['plev'].units))

        # Add colorbar
        hcb = plt.colorbar(cset1,
                           label='{:s} ({:s})'.format(
                               plotVar,
                               dataSets_rg[plotCase][plotVar].units))

        # Add case number
        ax.annotate(plotCase,
                    xy=[0, 1],
                    xycoords='axes fraction',
                    horizontalalignment='left',
                    verticalalignment='bottom')

        # Add time range
        tStepString = 't = [{:0d}, {:0d}]'.format(tLim[0], tLim[-1])
        ax.annotate(tStepString,
                    xy=[1, 1],
                    xycoords='axes fraction',
                    horizontalalignment='right',
                    verticalalignment='bottom')

        # Add vectors?
