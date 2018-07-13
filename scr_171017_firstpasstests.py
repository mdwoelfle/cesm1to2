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

import multiprocessing as mp  # Allow use of multiple cores
import datetime  # For keeping track of run times

from mdwtools import mdwfunctions as mwfn  # For averaging things
from mdwtools import mdwplots as mwp  # For plotting things

import cesm1to2plotter as c1to2p

import os  # operating system things.
# import matplotlib.cm as cm
# from scipy.stats import linregress

# %% Define funcitons as needed


def getcolordict():
    return c1to2p.getcolordict()


def getmarkerdict():
    return c1to2p.getmarkerdict()


def getquiverprops(uVar, vVar, **kwargs):
    return(c1to2p.getquiverprops(uVar, vVar, **kwargs))


# %% Main section
if __name__ == '__main__':

    # Set options/flags
    diff_flag = False
    loadErai_flag = False  # True to load ERAI fields
    loadGpcp_flag = True
    loadHadIsst_flag = False
    mp_flag = False  # True to use multiprocessing when regridding
    newRuns_flag = False
    obs_flag = False
    ocnOnly_flag = True  # Need to implement to confirm CTindex is right.
    regridVertical_flag = True
    regrid2file_flag = True
    regridOverwrite_flag = False
    reload_flag = False
    save_flag = False
    saveDir = c1to2p.setfilepaths()[2]
    saveSubDir = ''  # 'testfigs/66to125/'
    saveThenClose_flag = True
    verbose_flag = False

    fns_flag = True
    fnt_flag = True
    prect_flag = True

    plotBiasRelation_flag = False
    plotIndices_flag = False
    plotLonVCentroid_flag = False
    plotObsMap_flag = False
    plotOneMap_flag = False
    plotMultiMap_flag = False
    plotMultiPressureLat_flag = True
    plotGpcpTest_flag = False
    plotPressureLat_flag = False
    plotPressureLon_flag = False
    plotRegMean_flag = False
    plotSeasonalBiasRelation_flag = False
    plotZonRegMeanHov_flag = False
    plotZonRegMeanLines_flag = False
    testPlotErai_flag = False

    # Set name(s) of file(s) to load
    versionIds = ['01',
                  '28',
                  '36',
                  'ga7.66',
                  # '100',
                  # '113',
                  # '114',
                  # '116',
                  # '118',
                  '119',
                  # '119f',
                  # '119f_gamma',
                  # '119f_microp',
                  # '119f_liqss',
                  '125',
                  # '125f',
                  '161',
                  '194',
                  '195'
                  '297'
                  ]

    # Set levels for vertically regridding
    newLevs = np.array([100, 200, 275, 350, 425,
                        500, 550, 600, 650, 700,
                        750, 800, 850, 900, 950,
                        975, 1000])

    # Set variables to regrid vertically
    regridVars = ['V', 'OMEGA', 'RELHUM', 'CLOUD', 'T', 'U',
                  'AREI', 'AREL', 'AWNC', 'AWNI',
                  'CLDICE', 'CLDLIQ',
                  'ICIMR', 'ICWMR',
                  ]

    # Determine if need to load new datasets
    try:
        if not all([vid in dataSets.keys() for vid in versionIds]):
            load_flag = True
        else:
            if regridVertical_flag:
                # Ensure that vertical regrid dataset exists
                try:
                    # Esnure all requested regridded variables are present
                    if not all([regridVar in dataSets_rg[versionIds[0]]
                                for regridVar in regridVars]):
                        load_flag = True
                    else:
                        load_flag = False
                except NameError:
                    load_flag = True
            else:
                load_flag = False
    except NameError:
        load_flag = True

    # Get directory of file to load
    ncDir, ncSubDir, saveDir2 = c1to2p.setfilepaths(newRuns_flag)
    if saveDir is None:
        saveDir = saveDir2

    # Load (or reload) datasets from file and regrid if requested/needed
    if load_flag or reload_flag:
        dataSets, dataSets_rg = c1to2p.loadmodelruns(
            versionIds,
            mp_flag=mp_flag,
            ncDir=ncDir,
            ncSubDir=ncSubDir,
            newLevs=newLevs,
            regrid2file_flag=regrid2file_flag,
            regridOverwrite_flag=regridOverwrite_flag,
            regridVars=regridVars,
            regridVertical_flag=regridVertical_flag,
            )

    # Load observational datasets to be used for comparison
    obsDsDict = c1to2p.loadobsdatasets(
        obsList=None,
        gpcp_flag=(loadGpcp_flag or plotGpcpTest_flag),
        erai_flag=loadErai_flag,
        hadIsst_flag=loadHadIsst_flag,
        hadIsstYrs=[1979, 2010],
        )

    # Compatability code for quickness. May update later.
    if loadGpcp_flag or plotGpcpTest_flag:
        gpcpDs = obsDsDict['gpcp']
        # Load GPCP from both climo and add id
        gpcpClimoDs = obsDsDict['gpcpClimo']

    if loadHadIsst_flag:
        hadIsstDs = obsDsDict['hadIsst']

    if loadErai_flag:
        eraiDs = obsDsDict['erai']
        erai3dDs = obsDsDict['erai3d']

# %% Plot one map

    # save_flag = False
    # set plotting parameters
    latLim = np.array([-30.1, 30.1])
    # lonLim = np.array([119.5, 290.5])
    lonLim = np.array([90, 300])

    latLbls = np.arange(-30, 31, 10)
    # lonLbls = np.arange(120, 271, 30)
    lonLbls = np.arange(0, 361, 30)

    tSteps = np.arange(0, 12)

    rmRegMean_flag = False

    if plotOneMap_flag:

        levels = None  # np.arange(-0.5, 0.501, 0.05)

        for plotVar in ['OMEGA']:  # 'CLDTOT', 'PRECT', 'FNS']:
            plev = 850
            diffPlev = plev
            diff_flag = True  # False
            for plotCase in ['125']:
                diffCase = {'125': '119',
                            '125f': '119f',
                            '119f_gamma': '119f',
                            '119f_microp': '119f',
                            '119f_liqss': '119f',
                            '119': '119f'
                            }[plotCase]
                # plotCase, diffCase = [['01', '01'],
                #                      ['ga7.66', '36'],
                #                      ['119', '36'],
                #                      ['118', 'ga7.66'],
                #                      ['119', '118'],
                #                      ['125f', '119f'],
                #                      ['125', '36']][5]
                ocnOnly_flag = False
                quiver_flag = True
                uVar = 'TAUX'
                vVar = 'TAUY'

                # Ensure dataSets_rg exists
                try:
                    dataSets_rg[plotCase]
                except NameError:
                    dataSets_rg = {jCase: ['foo', 'bar']
                                   for jCase in list(dataSets.keys())}
                except KeyError:
                    dataSets_rg = {jCase: ['foo', 'bar']
                                   for jCase in list(dataSets.keys())}

                # Get quiver properties
                quiverProps = getquiverprops(uVar, vVar, plev,
                                             diff_flag=diff_flag)

                # Plot some fields for comparison
                (a, ax, c, m) = c1to2p.plotlatlon(
                    (dataSets_rg[plotCase]
                     if plotVar in dataSets_rg[plotCase]
                     else dataSets[plotCase]),  # hadIsstDs
                    plotVar,
                    box_flag=False,
                    boxLat=np.array([-3, 3]),
                    boxLon=np.array([180, 220]),
                    caseString=None,
                    cbar_flag=True,
                    cbar_dy=-0.05,
                    cbar_height=0.02,
                    cMap=None,  # 'RdBu_r',
                    compcont_flag=True,
                    diff_flag=diff_flag,
                    diffDs=(dataSets_rg[diffCase]
                            if plotVar in dataSets_rg[diffCase]
                            else dataSets[diffCase]),  # gpcpClimoDs,
                    diffPlev=diffPlev,
                    figDims=[12.25, 4.5],  # [17, 4.5] for full tropics
                    fontSize=12,
                    latLim=latLim,
                    latlbls=latLbls,
                    levels=levels,
                    lonLim=lonLim,
                    lonlbls=lonLbls,
                    ocnOnly_flag=ocnOnly_flag,
                    plev=plev,
                    quiver_flag=quiver_flag,
                    quiverDs=(dataSets_rg[plotCase]
                              if uVar in dataSets_rg[plotCase]
                              else dataSets[plotCase]),
                    quiverDiffDs=(dataSets_rg[diffCase]
                                  if uVar in dataSets_rg[diffCase]
                                  else dataSets[diffCase]),
                    quiverNorm_flag=False,
                    quiverScale=0.05,  # quiverProps['quiverScale'],
                    quiverScaleVar=None,
                    rmRegMean_flag=rmRegMean_flag,
                    subSamp=5,
                    stampDate_flag=False,
                    tSteps=tSteps,
                    tStepLabel_flag=True,
                    uRef=quiverProps['uRef'],
                    uVar=uVar,
                    vVar=vVar,
                    )

                # plt.tight_layout()

                # Save figure if requested
                if save_flag:
                    # Set directory for saving
                    if saveDir is None:
                        saveDir = c1to2p.setfilepaths()[2]

                    # Set file name for saving
                    tString = 'mon'
                    saveFile = (('d' if diff_flag else '') +
                                plotVar +
                                ('{:d}'.format(plev)
                                 if plev else '') +
                                '_latlon_' +
                                plotCase +
                                (('-' + diffCase + '_')
                                 if diff_flag else '_') +
                                tString +
                                '{:03.0f}'.format(tSteps[0]) + '-' +
                                '{:03.0f}'.format(tSteps[-1]))

                    # Set saved figure size (inches)
                    try:
                        fx, fy = hf.get_size_inches()
                    except NameError:
                        hf = plt.gcf()
                        fx, fy = hf.get_size_inches()

                    # Save figure
                    mapSaveSubDir = 'atm/maps/'
                    print(saveDir + mapSaveSubDir + saveFile)
                    # mwp.savefig(saveDir + saveSubDir + saveFile,
                    #            shape=np.array([fx, fy]))
                    mwp.savefig(saveDir + mapSaveSubDir + saveFile,
                                shape=np.array([fx, fy]))
                    if saveThenClose_flag:
                        plt.close('all')

# %% Plot map of obs

    if plotObsMap_flag:
        for plotVar in ['PRECT']:
            uVar = 'TAUX'
            vVar = 'TAUY'
            plev = 200
            diffPlev = plev
            tSteps = np.arange(0, 12)

            # Get quiver properties
            quiverProps = getquiverprops(uVar, vVar, plev,
                                         diff_flag=diff_flag)

            obsDs = {'OMEGA': erai3dDs,
                     'PRECT': gpcpClimoDs,
                     'TAUX': eraiDs,
                     'TS': hadIsstDs,
                     }[plotVar]
            obsVar = {'OMEGA': 'w',
                      'PRECT': 'precip',
                      'TAUX': 'iews',
                      'TS': 'sst',
                      }[plotVar]
            obsQuivDs = {'TAUX': eraiDs,
                         'U': erai3dDs}[uVar]
            obsUVar = {'TAUX': 'iews',
                       'U': 'u',
                       }[uVar]
            obsVVar = {'TAUY': 'inss',
                       'V': 'v',
                       }[vVar]

            # Plot some fields for comparison
            (a, ax, c, m) = c1to2p.plotlatlon(
                obsDs,  # hadIsstDs
                obsVar,
                box_flag=False,
                boxLat=np.array([-3, 3]),
                boxLon=np.array([180, 220]),
                caseString=None,
                cbar_flag=True,
                # cbar_dy=0.001,
                cbar_height=0.02,
                cMap=None,  # 'RdBu_r',
                compcont_flag=True,
                diff_flag=False,
                figDims=[6, 3.5],
                fontSize=12,
                latLim=latLim,  # np.array([-20, 20]),
                levels=None,  # np.arange(-15, 15.1, 1.5),
                lonLim=lonLim,  # np.array([119.5, 270.5]),
                makeFigure_flag=True,
                plev=plev,
                quiver_flag=False,  # True,
                quiverDs=obsQuivDs,
                quiverLat=obsQuivDs['lat'],
                quiverLon=obsQuivDs['lon'],
                quiverNorm_flag=False,
                quiverScale=quiverProps['quiverScale'],
                quiverScaleVar=None,
                rmRegMean_flag=False,
                save_flag=save_flag,
                saveDir=(c1to2p.setfilepaths()[2] + saveSubDir +
                         'obs_'),
                stampDate_flag=False,
                tSteps=tSteps,
                tStepLabel_flag=True,
                uRef=quiverProps['uRef'],
                uVar=obsUVar,
                vVar=obsVVar,
                )

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
        # save_flag = True
        plotVars = ['PRECT']  # , 'T', 'RELHUM', 'CLOUD']
        #        'CDNUMC']  # , 'SWCF', 'PBLH']
        plevs = [850, 500, 200]
        box_flag = False
        boxLat = np.array([-30, 30])
        boxLon = np.array([240, 270])
        diff_flag = False
        diffDs = None  # dataSets_rg
        diffIdList = ['119', '119f', None,
                      '119f', '119f', '119f']
        diffVar = None
        diffPlevs = plevs
        plotIdList = ['125', '125f', (None if diff_flag else '119f'),
                      '119f_microp', '119f_gamma', '119f_liqss']
        #        '01', '28', '36',
        #              'ga7.66', '119', '125',
        #              '161', '194', '195']
        quiver_flag = False
        uVar = 'U'
        vVar = 'V'
        levels = None  # np.arange(-0.25, 0.251, 0.025)

        tSteps = np.arange(0, 12)

        for plotVar in plotVars:
            for jlev, plev in enumerate(plevs):
                if diffVar is None:
                    diffVar = plotVar
                c1to2p.plotmultilatlon((dataSets_rg
                                        if plotVar
                                        in dataSets_rg[plotIdList[0]]
                                        else dataSets),
                                       plotIdList,
                                       plotVar,
                                       box_flag=box_flag,
                                       boxLat=boxLat,
                                       boxLon=boxLon,
                                       cbar_flag=True,
                                       cbarOrientation='vertical',
                                       compcont_flag=False,
                                       diff_flag=diff_flag,
                                       diffIdList=diffIdList,
                                       diffDs=diffDs,
                                       diffPlev=diffPlevs[jlev],
                                       diffVar=diffVar,
                                       figSize=[18, 6],
                                       fontSize=24,
                                       latLim=np.array([-30.1, 30.1]),
                                       latlbls=np.arange(-30, 30.1, 10),
                                       levels=levels,
                                       lonLim=np.array([99.5, 290.5]),
                                       lonlbls=np.arange(120, 270.1, 30),
                                       obsDs=gpcpClimoDs,
                                       ocnOnly_flag=False,
                                       plev=plev,
                                       quiver_flag=quiver_flag,
                                       quiverNorm_flag=False,
                                       quiverScale=(
                                           getquiverprops(
                                                uVar, vVar,
                                                diff_flag=diff_flag)['quiverScale']
                                           ),
                                       quiverUnits='inches',
                                       rmRegLatLim=np.array([-20, 20]),
                                       rmRegLonLim=np.array([119.5, 270.5]),
                                       rmRegMean_flag=False,
                                       rmse_flag=False,
                                       save_flag=save_flag,
                                       saveDir=(saveDir +
                                                saveSubDir +
                                                'atm/maps/'
                                                ),
                                       stampDate_flag=False,
                                       subFigCountStart='a',
                                       subSamp=7,
                                       tSteps=tSteps,
                                       uRef=getquiverprops(
                                           uVar, vVar,
                                           diff_flag=diff_flag)['uRef'],
                                       uVar=uVar,
                                       vVar=vVar,
                                       verbose_flag=False,
                                       )
                if diffVar == plotVar:
                    diffVar = None

# %% Plot regional means (biases)

    if plotRegMean_flag:

        # Set variable for plotting
        plotVar = 'TS'

        # Set plotting flags and specifications
        rmAnnMean_flag = False
        plotAnnMean_flag = False
        plotPeriodMean_flag = False
        # tSteps = np.arange(1, 5)
        tSteps = np.append(np.arange(0, 12), 0)
        divideByTropMean_flag = False
        tropLatLim = np.array([-20, 20])
        tropLonLim = np.array([0, 360])

        # Set default plot values
        title = mwp.getplotvarstring(plotVar)
        yLim = None
        yLim_annMean = None
        rmRefRegMean_flag = False
        colorDict = getcolordict()

        if plotVar in ['FNT']:
            ds = dataSets
            plotObs_flag = False
            latLim = np.array([-90, 90])
            lonLim = np.array([0, 360])
            obsDs = None
            obsVar = None
            ocnOnly_flag = False
            rmRefRegMean_flag = False
            title = plotVar
            yLim_annMean = None  # np.array([-0.1, 0.1])
        elif plotVar in ['PRECT', 'PRECL', 'PRECC']:
            ds = dataSets
            yLim = np.array([0, 2]
                            if divideByTropMean_flag
                            else [0, 6])
            plotObs_flag = True
            latLim = np.array([-20, 0])
            lonLim = np.array([210, 260])
            obsDs = gpcpClimoDs
            obsVar = 'precip'
            ocnOnly_flag = True
            rmRefRegMean_flag = False
            refLatLim = np.array([-20, 0])
            refLonLim = np.array([0, 360])
            title = 'PAI'  # '2xITCZ Index'
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
            latLim = np.array([0, 10])
            lonLim = np.array([210, 260])  # 210, 260])
            refLatLim = np.array([-10, 0])  # -10, 0])
            refLonLim = np.array([210, 260])  # 210, 260])
            if rmRefRegMean_flag or rmAnnMean_flag:
                yLim = np.array([-2.5, 2.5])
            else:
                yLim = np.array([297, 301])
            obsDs = hadIsstDs
            obsVar = 'sst'
            ocnOnly_flag = True
            title = 'd/dy(SST) in dITCZ Region'  # 'Cold Tongue Index'
            varSaveString = 'ddySSTinDITCZ'
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
        # colorDict = dict()

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
                           c=colorDict[vid],
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

        if save_flag:
            saveString = ('seascyc_' +
                          varSaveString +
                          ('_annMeanRemoved' if rmAnnMean_flag else '') +
                          ('divTropMean' if divideByTropMean_flag else '')
                          )
            print(saveDir + saveSubDir + saveString)
            mwp.savefig(saveDir + saveSubDir + saveString)

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
            if yLim_annMean is not None:
                plt.ylim(yLim_annMean)

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

# %% Plot predefined indices through time or as annual mean
    if plotIndices_flag:

        # Set name of index to plot
        #   available: 'dITCZ', 'PAI', 'pcent', 'dsstdy_epac', 'fnsasym'
        indexName = 'dITCZ'
        plotVar = None

        # Set plotting flags and specifications
        rmAnnMean_flag = False
        ocnOnly_flag = True
        plotAnnMean_flag = True
        plotPeriodMean_flag = True
        tSteps = np.arange(0, 12)
        # tSteps = np.append(np.arange(0, 12), 0)

        # Set default plot values
        title = indexName
        yLim = None

        if indexName in ['dITCZ']:
            ds = dataSets
            plotObs_flag = True
            plotVar = 'PRECT'
            obsDs = gpcpDs
            obsVar = 'precip'
            ocnOnly_flag = False
            title = 'Double-ITCZ Index'
            yLim = np.array([0, 6])
            yLim_annMean = np.array([1, 3])
        elif indexName.lower() in ['dpdy_epac']:
            ds = dataSets
            plotObs_flag = True
            plotVar = 'PS'
            obsDs = eraiDs
            obsVar = 'sp'
            ocnOnly_flag = True
            title = 'dP/dy (E Pac)'
            yLim = np.array([-1.5, 1.5])
            yLim_annMean = np.array([-1, 0])
            yLim_period = np.array([-1, 1])
        elif indexName.lower() in ['dsstdy_epac']:
            ds = dataSets
            plotObs_flag = True
            plotVar = 'TS'
            obsDs = hadIsstDs
            obsVar = 'sst'
            ocnOnly_flag = True
            title = 'dSST/dy (E. Pac)'
            yLim = np.array([-3, 3])
            yLim_annMean = np.array([0, 2])
            yLim_period = np.array([-1.2, 1])
        elif indexName.lower() in ['fnsasym']:
            ds = dataSets
            plotObs_flag = False
            plotVar = 'FNS'
            obsDs = None
            obsVar = None
            ocnOnly_flag = True
            title = 'FNS Asymmetry'
            yLim = None
            yLim_annMean = None
        elif indexName in ['PAI']:
            ds = dataSets
            plotObs_flag = True
            plotVar = 'PRECT'
            # obsDs = gpcpClimoDs
            obsDs = gpcpDs
            obsVar = 'precip'
            ocnOnly_flag = False
            title = 'Precipitation Asymmetry Index'
            yLim = np.array([-1.5, 1.5])
            yLim_annMean = np.array([0, 0.5])
        elif indexName.lower() in ['pcent']:
            ds = dataSets
            plotObs_flag = True
            plotVar = 'PRECT'
            obsDs = gpcpDs
            obsVar = 'precip'
            ocnOnly_flag = False
            title = 'Precipitation Centroid'
            yLim = np.array([-10, 10])
            yLim_annMean = np.array([0, 2])

        # Create dictionary to hold annual mean value (and colors)
        annMean = dict()
        timeMean = dict()
        colorDict = getcolordict()
        markerDict = getmarkerdict()

        # Create figure for plotting
        hf = plt.figure()
        hf.set_size_inches(6, 4.5,
                           forward=True)

        for vid in versionIds:
            # Compute given index through time
            indexDa = c1to2p.calcregmeanindex(ds[vid],
                                              indexName,
                                              indexType=None,
                                              indexVar=plotVar,
                                              ocnOnly_flag=ocnOnly_flag,
                                              )

            # Pull regional mean through time and plot
            pData = (indexDa.values - indexDa.mean(dim='time').values
                     if rmAnnMean_flag
                     else indexDa.values)
            hl, = plt.plot(np.arange(1, 13),
                           pData,
                           color=colorDict[vid],
                           label=vid,
                           marker=markerDict[vid],
                           )
            annMean[vid] = indexDa.mean(dim='time')
            timeMean[vid] = indexDa.values[tSteps].mean()

        # Repeat above for obs
        if plotObs_flag:
            # Compute given index through time
            obsIndexDa = c1to2p.calcregmeanindex(obsDs,
                                                 indexName,
                                                 indexType=None,
                                                 indexVar=obsVar,
                                                 ocnOnly_flag=False,
                                                 qc_flag=False,
                                                 )

            # Ensure plotting on correct figure
            plt.figure(hf.number)

            # Get data for plotting
            #   also remove annual mean if requested
            try:
                pData = (obsIndexDa.values -
                         obsIndexDa.mean(dim='time').values
                         if rmAnnMean_flag
                         else obsIndexDa.values)
            except ValueError:
                pData = (obsIndexDa.values -
                         obsIndexDa.mean(dim='month').values
                         if rmAnnMean_flag
                         else obsIndexDa.values)

            # Plot time series
            try:
                hl, = plt.plot(np.arange(1, 13),
                               pData,
                               lw=2,
                               c=colorDict['obs'],
                               label=obsDs.id,
                               marker=markerDict['obs']
                               )
            except ValueError:
                # Compute monthly climatologies and plot
                pData = np.reshape(pData,
                                   [int(pData.size/12), 12]).mean(axis=0)
                hl, = plt.plot(np.arange(1, 13),
                               pData,
                               lw=2,
                               c=colorDict['obs'],
                               label=obsDs.id,
                               marker=markerDict['obs']
                               )

            # Compute annual means
            try:
                annMean['obs'] = obsIndexDa.mean(dim='time')
            except ValueError:
                annMean['obs'] = obsIndexDa.mean(dim='month')

            # Compute mean over given timesteps
            timeMean['obs'] = pData[tSteps].mean()

        plt.xticks(np.arange(1, 13))
        plt.xlabel('Month')

        plt.ylabel('{:s}'.format(title) +
                   (' ({:s})'.format(indexDa.units)
                    if indexDa.units is not None
                    else '') +
                   ('\n[Annual mean removed]' if rmAnnMean_flag else '')
                   )
        try:
            plt.ylim(yLim)
        except NameError:
            pass

        plt.legend(title='Version', ncol=2)

        plt.title('Seasonal cycle of {:s}'.format(title) +
                  ('\n[Annual mean removed]' if rmAnnMean_flag else '')
                  )

        # Add annotation of years used to compute climatology
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

        # Clean up figure
        plt.tight_layout()

        # Save figure if requested
        if save_flag:
            # Set directory for saving
            if saveDir is None:
                saveDir = c1to2p.setfilepaths()[2]

            # Set file name for saving
            tString = 'mon'
            saveFile = ('seascyc_' + indexName.lower())

            # Set saved figure size (inches)
            fx, fy = hf.get_size_inches()

            # Save figure
            print(saveDir + saveFile)
            mwp.savefig(saveDir + saveSubDir + saveFile,
                        shape=np.array([fx, fy]))
            plt.close('all')

        # Plot annual mean values
        if plotAnnMean_flag:
            hf = plt.figure()
            hf.set_size_inches(6, 4.5, forward=True)
            # print([annMean[j].values for j in versionIds])
            for idx, vid in enumerate(versionIds):
                plt.scatter(idx + 1,
                            np.array(annMean[vid]),
                            marker=markerDict[vid],
                            c=colorDict[vid],
                            s=80,
                            )
            if 'obs' in annMean.keys():
                # print(annMean['obs'])
                plt.scatter([len(annMean)],
                            annMean['obs'],
                            marker=markerDict['obs'],
                            c=colorDict['obs'],
                            s=80,
                            )

            plt.xticks(np.arange(1, len(annMean) + 1),
                       ((versionIds + ['obs'])
                        if 'obs' in annMean.keys()
                        else versionIds))
            plt.xlabel('Version')

            plt.ylabel('{:s}'.format(title) +
                       (' ({:s})'.format(indexDa.units)
                       if indexDa.units is not None
                       else '')
                       )
            try:
                plt.ylim(yLim_annMean)
            except NameError:
                pass

            plt.grid(ls='--')
            plt.gca().set_axisbelow(True)

            plt.title('Annual mean {:s}'.format(title))
            plt.tight_layout()

            # Save figure if requested
            if save_flag:
                # Set directory for saving
                if saveDir is None:
                    saveDir = c1to2p.setfilepaths()[2]

                # Set file name for saving
                tString = 'mon'
                saveFile = ('annmean_' + indexName.lower())

                # Set saved figure size (inches)
                fx, fy = hf.get_size_inches()

                # Save figure
                print(saveDir + saveSubDir + saveFile)
                mwp.savefig(saveDir + saveSubDir + saveFile,
                            shape=np.array([fx, fy]))
                plt.close('all')

        # Plot time mean values
        if plotPeriodMean_flag:
            plt.figure()
            for indx, vid in enumerate(versionIds):
                plt.scatter(indx + 1,
                            np.array(timeMean[vid]),
                            marker=markerDict[vid],
                            c=colorDict[vid],
                            s=80,
                            )
            if 'obs' in timeMean.keys():
                plt.scatter([len(timeMean)],
                            timeMean['obs'],
                            marker=markerDict['obs'],
                            c=colorDict['obs'],
                            s=80,
                            )

            plt.xticks(np.arange(1, len(timeMean) + 1),
                       ((versionIds + ['obs'])
                        if 'obs' in timeMean.keys()
                        else versionIds))
            plt.xlabel('Version')

            plt.ylabel('{:s}'.format(title) +
                       (' ({:s})'.format(indexDa.units)
                       if indexDa.units is not None
                       else '')
                       )
            try:
                plt.ylim(yLim_period)
            except NameError:
                pass

            plt.grid(ls='--')
            plt.gca().set_axisbelow(True)

            monIds = ['J', 'F', 'M', 'A', 'M', 'J',
                      'J', 'A', 'S', 'O', 'N', 'D']
            tStepString = ''.join([monIds[tStep] for tStep in tSteps])
            if tStepString == 'JFD':
                tStepString = 'DJF'
            plt.title('{:s} mean {:s}'.format(tStepString, title))
            plt.tight_layout()


# %% Correlate bias indices (CTI, dTICZ, Walker)

    # Available biases:
    #   'cpacshear', 'cti', 'ditcz', 'dpdy_epac', 'dslp', 'dsstdy_epac',
    #   'walker'
    xIndex = 'dpdy_epac'
    yIndex = 'dITCZ'

    if plotBiasRelation_flag:

        # True to use different time steps for x and y axes
        splitTSteps_flag = False

        c1to2p.plotbiasrelation(dataSets,
                                xIndex,
                                yIndex,
                                ds_rg=None,  # dataSets_rg
                                legend_flag=True,
                                makeFigure_flag=True,
                                obsDsDict={'cpacshear': erai3dDs,
                                           'cti': hadIsstDs,
                                           'ditcz': gpcpClimoDs,
                                           'dpdy_epac': eraiDs,
                                           'dslp': eraiDs,
                                           'dsstdy_epac': hadIsstDs,
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

        matchLimits_flag = False
        axisLimitDict = {'cpacshear': np.array([-5, -25]),
                         'cti': np.array([-2, 0.5]),
                         'ditcz': np.array([0.5, 5]),
                         'dpdy_epac': np.array([-1.5, 1.5]),
                         'dsstdy_epac': np.array([-1.2, 1]),
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
                                    ds_rg=None,  # dataSets_rg,
                                    legend_flag=(j == 3),
                                    makeFigure_flag=False,
                                    obsDsDict={'cpacshear': erai3dDs,
                                               'cti': hadIsstDs,
                                               'ditcz': gpcpClimoDs,
                                               'dpdy_epac': eraiDs,
                                               'dsstdy_epac': hadIsstDs,
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
                saveDir = c1to2p.setfilepaths()[2]

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
                                   saveDir=c1to2p.setfilepaths()[2] + saveSubDir,
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
                mwp.savefig(c1to2p.setfilepaths()[2] +
                            saveSubDir + plotVar + '_zonmean_' +
                            mwp.getlatlimstring(latLim, '') + '_' +
                            mwp.getlonlimstring(lonLim, '') +
                            'obs')

# %% Plot time mean, zonal mean lines
    if plotZonRegMeanLines_flag:

        # Set flags and options
        latLim = np.array([-25, 25])
        lonLim = np.array([210, 260])
        plotLatLim = np.array([-20, 20])

        plotVar = 'PRECT'

        # Plot versions
        c1to2p.plotmultizonregmeanlines(dataSets,
                                        versionIds,
                                        plotVar,
                                        colorDict=getcolordict(),
                                        diff_flag=False,
                                        diffIdList=None,
                                        diffDs=None,
                                        diffVar=None,
                                        fontSize=12,
                                        gsEdges=[0.1, 1.0, 0.15, 0.95],
                                        latLim=latLim,
                                        latlbls=None,
                                        lonLim=lonLim,
                                        legend_flag=True,
                                        lw=2,
                                        obsDs=gpcpClimoDs,
                                        ocnOnly_flag=False,
                                        plotObs_flag=True,
                                        plotLatLim=plotLatLim,
                                        save_flag=save_flag,
                                        saveDir=c1to2p.setfilepaths()[2] + saveSubDir,
                                        stdUnits_flag=True,
                                        subFigCountStart='a',
                                        )


# %% Plot pressure-latitude figure with vectors (potentially)
    # plotPressureLat_flag = True
    if plotPressureLat_flag:
        # Set variable to plot with colored contours
        colorVar = 'OMEGA'
        plotCases = ['125', '125f',
                     '119f_microp', '119f_gamma', '119f_liqss']
        diff_flag = True
        lineCont_flag = False
        lineContDiff_flag = False

        for plotCase in plotCases:
            diffCase = {'119f_gamma': '119f',
                        '119f_liqss': '119f',
                        '119f_microp': '119f',
                        '125': '119',
                        '125f': '119f',
                        }[plotCase]

            # Set variable to plot with black contours
            contVar = None  # 'CLOUD'

            quiver_flag = True
            # save_flag = False

            # Set plotting limits
            latLim = np.array([-30, 30])
            lonLim = np.array([240, 270])
            pLim = np.array([1000, 200])
            tLim = np.array([0, 12])  # exclusive of end pt.
            dt = 1

            c1to2p.plotpressurelat(
                dataSets_rg[plotCase],
                colorVar,
                # caseString=None,
                cbar_flag=True,
                # cbar_dy=-0.1,
                # cbar_height=0.02,
                # cMap=None,
                colorConts=None,
                # dCont_flag=False,
                # dContCase=None,
                diff_flag=diff_flag,
                diffDs=dataSets_rg[diffCase],
                dt=1,
                latLim=latLim,
                latSubSamp=3,
                lonLim=lonLim,
                lineCont_flag=lineCont_flag,
                lineContDiff_flag=lineContDiff_flag,
                lineConts=None,
                lineContVar=colorVar,
                lineContDs=dataSets_rg[(plotCase
                                        if lineContDiff_flag
                                        else diffCase)],
                lineContDiffDs=dataSets_rg[diffCase],
                makeFigure_flag=True,
                pLim=pLim,
                quiver_flag=quiver_flag,
                # quiverScale=3,
                # quiverUnits='inches',
                save_flag=False,
                saveDir=None,
                saveSubDir=None,
                tLim=tLim,
                wScale=100,
                )

# %% Plot multiple pressure-latitude figure with vectors (potentially)

    if plotMultiPressureLat_flag:
        # save_flag = True
        # Set variable to plot with colored contours
        colorVars = ['AREI']  # , 'AREL', 'CLDLIQ', 'CLDICE',
        #             'T', 'RELHUM', 'CLOUD', 'OMEGA']
        plotCases = ['125', '125f', None,
                     '119f_microp', '119f_gamma', '119f_liqss']
        diff_flag = False
        lineCont_flag = False
        lineContDiff_flag = False

        diffCase = {'119f_gamma': '119f',
                    '119f_liqss': '119f',
                    '119f_microp': '119f',
                    '125': '119',
                    '125f': '119f',
                    }

        # Set variable to plot with black contours
        contVar = None  # 'CLOUD'

        quiver_flag = True
        # save_flag = False

        # Set plotting limits
        latLim = np.array([-40, 40])
        lonLim = np.array([240, 270])
        pLim = np.array([1000, 200])
        tLim = np.array([0, 12])  # exclusive of end pt.
        dt = 1

        # Loop through variables and plot them.
        for colorVar in colorVars:
            c1to2p.plotmultipressurelat(
                dataSets_rg,
                plotCases,
                colorVar,
                # caseString=None,
                cbar_flag=True,
                # cbar_dy=-0.1,
                # cbar_height=0.02,
                # cMap=None,
                colorConts=None,
                # dCont_flag=False,
                # dContCase=None,
                diff_flag=diff_flag,
                diffIdList=[diffCase[vid] if vid is not None else None
                            for vid in plotCases],
                dt=1,
                latLbls=np.arange(-30, 30.1, 10),
                latLim=latLim,
                latSubSamp=3,
                lonLim=lonLim,
                lineCont_flag=lineCont_flag,
                # lineContDiff_flag=lineContDiff_flag,
                # lineConts=None,
                # lineContVar=colorVar,
                # lineContDs=dataSets_rg,
                # lineContDiffIdList=None,
                pLim=pLim,
                quiver_flag=quiver_flag,
                # quiverScale=3,
                # quiverUnits='inches',
                save_flag=save_flag,
                saveDir=saveDir,
                saveSubDir='atm/meridslices/',
                saveThenClose_flag=saveThenClose_flag,
                tLim=tLim,
                wScale=100,
                )

# %% Plot pressure-longitude figure with vectors (potentially)

    if plotPressureLon_flag:
        # Set variable to plot with colored contours
        plotVar = 'CLOUD'
        plotCases = ['125', '125f']
        # , '119f_gamma', '119f_liqss', '119f_microp']
        contVar = None  # 'CLOUD'
        contCases = plotCases
        diff_flag = True

        for idx, plotCase in enumerate(plotCases):
            if diff_flag:
                diffCase = {'119f_gamma': '119f',
                            '119f_liqss': '119f',
                            '119f_microp': '119f',
                            '125': '119',
                            '125f': '119f',
                            }[plotCase]
            else:
                diffCase = None

            contCase = contCases[idx]
            dcontCase = diffCase
            dcont_flag = diff_flag

            quiver_flag = True
            # save_flag = False

            # Set plotting limits
            latLim = np.array([-5, 5])
            lonLim = np.array([100, 300])
            pLim = np.array([1000, 200])
            tLim = np.array([0, 12])  # exclusive of end pt.
            dt = 1

            # Compute meridional mean over requested latitudes
            #   **currently not area weighting***
            a = dataSets_rg[plotCase].loc[
                dict(lon=slice(lonLim[0], lonLim[-1]),
                     lat=slice(latLim[0]-2, latLim[-1]+2))
                ].mean(dim='lat')
            if diff_flag:
                b = dataSets_rg[diffCase].loc[
                    dict(lon=slice(lonLim[0], lonLim[-1]),
                         lat=slice(latLim[0]-2, latLim[-1]+2))
                    ].mean(dim='lat')

            # Mean data over requested plotting time period
            a = a.isel(time=slice(tLim[0], tLim[-1], dt)).mean(dim='time')
            if diff_flag:
                b = b.isel(time=slice(tLim[0], tLim[-1], dt)).mean(dim='time')

            # Get contours for plotting filled contours
            try:
                if diff_flag:
                    colorConts = {'CLOUD': np.arange(-0.2, 0.201, 0.02),
                                  'OMEGA': (np.arange(-0.02, 0.0201, 0.002) *
                                            (1 if 'f' in plotCase else 5)),
                                  'RELHUM': np.arange(-20, 20.1, 2),
                                  'T': np.arange(-2, 2.1, 0.2),
                                  'V': np.arange(-0.3, 0.31, 0.03),
                                  }[plotVar]
                else:
                    colorConts = {'CLOUD': np.arange(0, 0.301, 0.02),
                                  'OMEGA': np.arange(-0.1, 0.101, 0.01),
                                  'RELHUM': np.arange(0, 100.1, 5),
                                  'T': np.arange(225, 295.1, 5),
                                  'V': np.arange(-4, 4.1, 0.5),
                                  'Z3': np.arange(0, 15001, 1000),
                                  }[plotVar]
            except KeyError:
                conts = None

            # Get contours for plotting lined contours
            try:
                if diff_flag:
                    lineConts = {'CLOUD': np.arange(-0.2, 0.201, 0.02),
                                 'RELHUM': np.arange(-20, 20.1, 2),
                                 'T': np.arange(-2, 2.1, 0.2),
                                 'V': np.arange(-3, 3.1, 0.3),
                                 }[contVar]
                else:
                    lineConts = {'CLOUD': np.arange(0, 0.301, 0.05),
                                 'RELHUM': np.arange(0, 100.1, 10),
                                 'T': np.arange(225, 295.1, 10),
                                 'V': np.arange(-4, 4.1, 0.5),
                                 'Z3': np.arange(0, 15001, 1500),
                                 }[contVar]
            except KeyError:
                lineConts = None

            # Create figure for plotting
            hf = plt.figure()

            # Plot meridional mean slice with filled contours
            cset1 = plt.contourf(a['lon'],
                                 a['plev'],
                                 a[plotVar] -
                                 (b[plotVar] if diff_flag
                                  else 0),
                                 colorConts,
                                 cmap=mwp.getcmap(plotVar,
                                                  diff_flag=diff_flag),
                                 extend='both')

            # Plot meridional mean slice with black contours
            if contVar is not None:
                cset2 = plt.contour(a['lon'],
                                    a['plev'],
                                    a[contVar] -
                                    (b[contVar] if diff_flag
                                     else 0),
                                    lineConts,
                                    colors='k')
                plt.clabel(cset2)  # , fontsize=9)

            # Compute w
            if quiver_flag:
                R = 287.058  # [J/kg/K]
                g = 9.80662  # [m/s^2]
                aw = -a['OMEGA']*R*a['T']/(a['plev']*100*g)  # *100 converts to Pa
                bw = -b['OMEGA']*R*b['T']/(b['plev']*100*g)  # *100 converts to Pa
                wScale = (1000 if diff_flag else 100)

                lonSubSamp = 6
                quiverUnits = 'inches'
                quiverScale = (1 if diff_flag else 3)  # 5
                q1 = plt.quiver(a['lon'][::lonSubSamp],
                                a['plev'],
                                a['V'][:, ::lonSubSamp] -
                                (b['V'][:, ::lonSubSamp] if diff_flag
                                 else 0),
                                wScale*(aw[:, ::lonSubSamp] -
                                        (bw[:, ::lonSubSamp] if diff_flag
                                         else 0)
                                        ),
                                units=quiverUnits,
                                scale=quiverScale
                                )
                plt.quiverkey(q1, 0.3, 1.05,
                              1,
                              '[v ({:d} {:s}), '.format(
                                  1,
                                  mwfn.getstandardunitstring('m/s')) +
                              'w ({:0.0e} {:s})]'.format(
                                  1/wScale,
                                  mwfn.getstandardunitstring('m/s')),
                              coordinates='axes',
                              labelpos='E')

            # Dress plot
            ax = plt.gca()
            # Flip y direction
            ax.invert_yaxis()

            # Set x and y limits
            plt.xlim(lonLim)
            plt.ylim(pLim)
            ax.set_yscale('log')

            # Label axes
            plt.xlabel('Longitude')
            plt.ylabel('Pressure ({:s})'.format(a['plev'].units))

            # Add colorbar
            hcb = plt.colorbar(cset1,
                               label='{:s} ({:s})'.format(
                                   plotVar,
                                   dataSets_rg[plotCase][plotVar].units))
            if plotVar == 'OMEGA':
                plt.annotate('(up)',
                             xy=(0.85, 0.1),
                             xycoords='figure fraction',
                             horizontalalignment='right',
                             verticalalignment='bottom')

            # Add case number
            ax.annotate(plotCase +
                        ('-{:s}'.format(diffCase) if diff_flag
                         else ''),
                        xy=[0, 1],
                        xycoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom')

            # Add time range
            tStepString = 't = [{:0d}, {:0d}]'.format(tLim[0], tLim[-1]-1)
            ax.annotate(tStepString,
                        xy=[1, 1],
                        xycoords='axes fraction',
                        horizontalalignment='right',
                        verticalalignment='bottom')

            if save_flag:
                # Set directory for saving
                if saveDir is None:
                    saveDir = c1to2p.setfilepaths()[2] + saveSubDir
                saveDir = c1to2p.setfilepaths()[2] + 'atm/meridslices/'

                # Set filename for saving
                saveFile = (('d' if diff_flag else '') +
                            plotVar +
                            ('_VW' if quiver_flag else '') +
                            '_' + plotCase +
                            ('-{:s}'.format(diffCase) if diff_flag else '') +
                            '_' + mwp.getlatlimstring(latLim, '') +
                            '_' + mwp.getlonlimstring(lonLim, '') +
                            '_mon{:02d}-{:02d}'.format(tLim[0], tLim[-1]-1)
                            )

                # Set saved figure size (inches)
                fx = hf.get_size_inches()[0]
                fy = hf.get_size_inches()[1]

                # Save figure
                print(saveDir + saveFile)
                mwp.savefig(saveDir + saveFile,
                            shape=np.array([fx, fy]))
                plt.close('all')

    # %% Plot meridional slices of obs
    # Plot observed meridional slice
    if testPlotErai_flag:  # all([testPlot_flag,
        #    loadErai_flag]):
        # Set variable to plot
        plotVar = 'r'
        quiver_flag = True
        save_flag = True

        # Set plotting limits
        latLim = np.array([-20, 20])
        lonLim = np.array([210, 260])
        pLim = np.array([1000, 200])
        tLim = np.array([7, 12])
        dt = 1

        # Compute meridional mean over requested longitudes
        a = erai3dDs.loc[
            dict(lon=slice(lonLim[0], lonLim[-1]),
                 lat=slice(latLim.max()+2, latLim.min()-2))
            ].mean(dim='lon')

        # Mean data over requested plotting time period
        a = a.isel(time=slice(tLim[0], tLim[-1], dt)).mean(dim='time')

        # Get contours for plotting
        try:
            conts = {'r': np.arange(0, 100.1, 5),
                     't': np.arange(225, 295.1, 5),
                     'u': np.arange(-24, 24.1, 2),
                     'V': np.arange(-4, 4.1, 0.5),
                     'Z3': np.arange(0, 15001, 1000),
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
                             cmap=mwp.getcmap(plotVar),
                             extend='both')

        # Compute w
        if quiver_flag:
            R = 287.058  # [J/kg/K]
            g = 9.80662  # [m/s^2]
            w = -a['w']*R*a['t']/(a['plev']*100*g)  # *100 to covert to Pa
            wScale = 100

            latSubSamp = 2
            quiverUnits = 'inches'
            quiverScale = 5
            q1 = plt.quiver(a['lat'][::latSubSamp],
                            a['plev'],
                            a['v'][:, ::latSubSamp],
                            wScale*w[:, ::latSubSamp],
                            units=quiverUnits,
                            scale=quiverScale
                            )
            plt.quiverkey(q1, 0.3, 1.05,
                          1,
                          '[v ({:d} {:s}), '.format(
                              1,
                              mwfn.getstandardunitstring(
                                  erai3dDs['v'].units)) +
                          'w ({:0.0e} {:s})]'.format(
                              1/wScale,
                              mwfn.getstandardunitstring(
                                  'm/s')),
                          coordinates='axes',
                          labelpos='E')

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
                               erai3dDs[plotVar].units))

        # Add case number
        ax.annotate('ERAI',
                    xy=[0, 1],
                    xycoords='axes fraction',
                    horizontalalignment='left',
                    verticalalignment='bottom')

        # Add time range
        tStepString = 't = [{:0d}, {:0d}]'.format(tLim[0], tLim[-1]-1)
        ax.annotate(tStepString,
                    xy=[1, 1],
                    xycoords='axes fraction',
                    horizontalalignment='right',
                    verticalalignment='bottom')

        if save_flag:
            # Set directory for saving
            if saveDir is None:
                saveDir = c1to2p.setfilepaths()[2] + saveSubDir
            saveDir = c1to2p.setfilepaths()[2] + 'atm/meridslices/'

            # Set filename for saving
            saveFile = (('d' if diff_flag else '') +
                        plotVar +
                        ('_VW' if quiver_flag else '') +
                        '_' + 'erai' +
                        ('-{:s}'.format(diffCase) if diff_flag else '') +
                        '_' + mwp.getlatlimstring(latLim, '') +
                        '_' + mwp.getlonlimstring(lonLim, '') +
                        '_mon{:02d}-{:02d}'.format(tLim[0], tLim[-1]-1)
                        )

            # Set saved figure size (inches)
            fx = hf.get_size_inches()[0]
            fy = hf.get_size_inches()[1]

            # Save figure
            print(saveDir + saveFile)
            mwp.savefig(saveDir + saveFile,
                        shape=np.array([fx, fy]))
            # plt.close('all')

# %% Plot precipitation centroid as function of longitude
    if plotLonVCentroid_flag:
        plotCaseList = ['125f']
        refDs = dataSets['119f']  # gpcpClimoDs
        if 'PRECT' in refDs:
            refVar = 'PRECT'
        elif 'precip' in refDs:
            refVar = 'precip'
        c1to2p.plotprecipcentroidvlon([dataSets[jCase]
                                       for jCase in plotCaseList],
                                      ['PRECT']*len(plotCaseList),
                                      closeOnSaving_flag=False,
                                      contCmap='RdYlGn',
                                      diff_flag=True,
                                      makeFigure_flag=True,
                                      refDs=refDs,
                                      refVar=refVar,
                                      yLim=None,
                                      save_flag=save_flag,
                                      saveDir=saveDir + 'atm/centroidstuff/',
                                      )
