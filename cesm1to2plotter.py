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
import matplotlib.colors as colors  # for getting colors
import matplotlib.cm as cmx  # for making colors
import matplotlib.gridspec as gridspec  # pretty subplots

import calendar


import mdwtools.mdwfunctions as mwfn  # import personal processing functions
import mdwtools.mdwplots as mwp       # import personal plotting functions
# import netCDF4 as nc4            # import netCDF4 as nc4
import numpy as np               # for computational things
import xarray as xr              # for handling netcdfs and datasets

from socket import gethostname   # used to determine which machine we are
#                                #   running on
import datetime    # for getting dates and tracking run time
import multiprocessing as mp     # Allow use of multiple cores
# import time

# Functions for plotlatloncontsovertime
# from mpl_toolkits.basemap import Basemap  # import tool for lat/lon plotting
# from matplotlib import cm  # import access to colormaps

# from scipy import interpolate    # import interpolation functions from scipy

# import multiprocessing as mp  # Allow use of multiple cores

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
    - dpdy_epac - East Pacific meridonal PS gradient
    - dSLP - Eq. Pacific SLP gradient (~Walker strength)
    - dSSTdy_epac - East Pacific meridional SST gradient
    - fnsasym - Asymmetry in net surface flux over ocean only
    - pai - precipitation asymmetry index
    - pcent - precipitation centroid
    - sepsst - southeast Pacific SST metric (still in flux)
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
                                        indexType=indexType,
                                        precipVar=indexVar,
                                        )

    elif indexName.lower() in ['dsstdy_epac', 'dpdy_epac']:
        # Assign default index if none provided
        if indexType is None:
            indexType = 'epac'
        if indexVar is None:
            indexVar = 'TS'

        # Compute dsstdy_epac
        indexDa = mwfn.calcdsddyindex(ds,
                                      indexType=indexType,
                                      indexVar=indexVar)

    elif indexName.lower() in ['fnsasym']:
        # Assign default index if none provided
        if indexType is None:
            indexType = 'Xiangetal2017'
        if indexVar is None:
            indexVar = 'FNS'

        # Compute net surface flux asymmetry
        indexDa = mwfn.calcdsfnsasymindex(ds,
                                          indexType=indexType,
                                          fnsVar=indexVar,
                                          qc_flag=qc_flag
                                          )

    elif indexName.lower() in ['pai', 'precipasymmetry']:
        # Assign default index if none provded
        if indexType is None:
            indexType = 'HwangFrierson2012'
        if indexVar is None:
            indexVar = 'PRECT'

        # Compute precipitation asymmetry index
        indexDa = mwfn.calcdsprecipasymindex(ds,
                                             indexType=indexType,
                                             precipVar=indexVar,
                                             qc_flag=qc_flag,
                                             )

    elif indexName.lower() in ['precipcentroid', 'precipitationcentroid',
                               'pcent']:
        # Assign default index if none provided
        if indexType is None:
            indexType = 'areaweight'
        if indexVar is None:
            indexVar = 'PRECT'

        # Compute centroid of tropical precipitation
        indexDa = mwfn.calcdsprecipcentroid(ds,
                                            indexType=indexType,
                                            precipVar=indexVar,
                                            qc_flag=qc_flag)

    elif indexName.lower() in ['sepacsst', 'sepsst', 'sepsst_raw']:
        # Assign default index if none provided
        if indexType is None:
            indexType = 'tropicalRelative'
        if indexVar is None:
            indexVar = 'TS'
        
        # Compute southeast Pacific SST metric
        indexDa = mwfn.calcdssepsstindex(ds,
                                         indexType=('raw' if 'raw' in indexName.lower() else indexType),
                                         indexVar=indexVar,
                                         ocnOnly_flag=ocnOnly_flag,
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
    else:
        raise NameError('Cannot find function to compute ' +
                        'index: {:s}'.format(indexName))

    # Return index data array
    return indexDa


def getavailableyearslist(versionId):
    """
    Get list of averaging periods available for a given model version
    """
    return {'01': ['0.9x1.25'],
            '28': ['2-10', '2-20', '50-74', '75-99'],
            '36': ['2-10', '2-20', '21-40', '60-60', '75-99'],
            'ga7.66': ['2-20', '20-39', '55-74'],
            '100': ['2-5', '2-7', '2-10', '2-20', '2-21', '10-29'],
            '113': ['0.9x1.25', '2-21'],
            '114': ['2-3', '2-11', '2-21'],
            '116': ['2-3'],
            '118': ['2-11', '2-21'],
            '119': ['2-9', '2-21', '21-40', '30-49', '75-99'],
            '125': ['2-9', '2-21', '11-30', '21-40', '70-89', '80-99',
                    '100-109', '100-119'],
            '161': ['1850-1869', '1920-1939', '1980-1999'],
            '194': ['14-33', '15-29', '50-69', '100-119'],
            '195': ['15-29', '50-69', '80-99', '100-119', '122-141'],
            }[versionId]


def getcasebase(versionId=None,
                dict_flag=False
                ):
    """
    Get long form nave for a given version ID for cesm1to2 cases

    Args:
        versionId - id for the version of interest
    Kwargs:
        dict_flag - True to return full dictionary rather than one case's value
    """
    # Define case bases
    casebaseDict = {'01': 'b.e15.B1850G.f09_g16.pi_control.01',
                    '28': 'b.e15.B1850G.f09_g16.pi_control.28',
                    '36': 'b.e15.B1850.f09_g16.pi_control.36',
                    'ga7.66': 'b.e15.B1850.f09_g16.pi_control.all_ga7.66',
                    '100': 'b.e15.B1850.f09_g16.pi_control.all.100',
                    '113': 'b.e15.B1850.f09_g16.pi_control.all.113',
                    '114': 'b.e15.B1850.f09_g16.pi_control.all.114',
                    '116': 'b.e15.B1850.f09_g16.pi_control.all.116',
                    '118': 'b.e15.B1850.f09_g16.pi_control.all.118',
                    '119': 'b.e15.B1850.f09_g16.pi_control.all.119',
                    '119f': 'f.2000_DEV.f09_f09.pd_control.119',
                    '119f_gamma': 'f.2000_DEV.f09_f09.pd_gamma.119',
                    '119f_ice': 'f.2000_DEV.f09_f09.pd_ice.119',
                    '119f_liqss': 'f.2000_DEV.f09_f09.pd_liqss.119',
                    '119f_microp': 'f.2000_DEV.f09_f09.pd_microp.119',
                    '119f_nocwv': 'f.2000_DEV.f09_f09.pd_pra.119',
                    '119f_pra': 'f.2000_DEV.f09_f09.pd_pra.119',
                    '125': 'b.e20.B1850.f09_g16.pi_control.all.125',
                    '125f': 'f.2000_DEV.f09_f09.pd_control.125',
                    '161': 'b.e20.BHIST.f09_g17.20thC.161_01',
                    '194': 'b.e20.B1850.f09_g17.pi_control.all.194',
                    '195': 'b.e20.B1850.f09_g17.pi_control.all.195',
                    '297': 'b.e20.B1850.f09_g17.pi_control.all.297',
                    '297_nocwv': 'b.1850.f09_g17.pi_unimicro.297',
                    '297f': 'f.2000.f09_f09.pd_control.cesm20',
                    '297f_microp': 'f.2000.f09_f09.pd_microp.cesm20',
                    '297f_pra': 'f.2000.f09_f09.pd_pra.cesm20',
                    '297f_sp': 'f.2000.f09_f09.pd_spcontrol.cesm20',
                    'cesm20f': 'f.2000.f09_f09.pd_control.cesm20',
                    'cesm20f_microp': 'f.2000.f09_f09.pd_microp.cesm20',
                    'cesm20f_pra': 'f.2000.f09_f09.pd_pra.cesm20',
                    'cesm20f_sp': 'f.2000.f09_f09.pd_spcontrol.cesm20',
                    }

    # Return as requested
    if any([dict_flag, versionId is None]):
        return casebaseDict
    else:
        return casebaseDict[versionId]


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
            '28': '#ff7f0e',
            '36': '#2ca02c',
            'ga7.66': '#d62728',
            '100': '#42e5f4',
            '113': '#aff441',
            '114': '#f441a6',
            '116': '#41f4b2',
            '118': '#f4b541',
            '119': '#9467bd',
            '119f': '#9467bd',
            '119f_gamma': '#9467bd',
            '119f_ice': '#9467bd',
            '119f_liqss': '#9467bd',
            '119f_microp': '#9467bd',
            '119f_nocwv': '#9467bd',
            '119f_pra': '#9467bd',
            '125': '#8c564b',
            '125f': '#8c564b',
            '161': '#e377c2',
            '194': '#7f7f7f',
            '195': '#bcbd22',
            '297': '#42d1f4',
            '297_nocwv': '#42d1f4',
            '297f': '#42d1f4',
            '297f_microp': '#42d1f4',
            '297f_pra': '#42d1f4',
            '297f_sp': '#42d1f4',
            'cesm20f': '#42d1f4',
            'cesm20f_microp': '#42d1f4',
            'cesm20f_pra': '#42d1f4',
            'cesm20f_sp': '#42d1f4',
            'obs': [0, 0, 0],
            }

def getloadfilelists(versionIds,
                     loadSuffixes,
                     climo_flag=True,
                     fileBaseDict=None,
                     nyrs=1,
                     yr0=2,
                     ):
    """
    get list of files to load for each model version/case
    - will be machine specific as NCAR systems differ from UW machines
    """
    
    # Ensure have full case names
    if fileBaseDict is None:
        fileBaseDict = getcasebase()
    
    loadFileLists = dict()
    if gethostname() in getuwmachlist():
        for vid in versionIds:
            loadFileLists[vid] = [ncDir + fileBaseDict[vid] +
                                  '/' +
                                  ('atm/hist/'
                                   if 'f' in vid
                                   else ncSubDir) +
                                  fileBaseDict[vid] +
                                  loadSuffix
                                  for loadSuffix in loadSuffixes[vid]
                                 ]
    elif gethostname()[0:6] in getncarmachlist(6):

        # Directories on NCAR systems may be all screwy due to changes to file
        #   system on 208-07-11 

        # Set info for "runs of convenience" from Celice at NCAR
        cecileCases = ['01', '28', '36', 'ga7.66', '100', '113', '114',
                       '116', '118','119', '125', '161', '194', '195']
        cecileDir = '/glade/p_old/cgd/amp/people/hannay/amwg/climo/'
        cecileSubDir = '0.9x1.25/'

        # Set info for release runs from NCAR
        releaseCases = ['297']
        releaseDir = '/glade/p_old/cesm0005/archive/'
        releaseSubDir = 'atm/hist/'
        releaseClimoDir = '/gpfs/fs1/work/woelfle/cesm1to2/climos/'
        releaseClimoSubDir = '131-139/'
        
        # Set info for runs by me
        woelfleCases = ['119f', '119f_gamma', '119f_ice', '119f_liqss',
                        '119f_microp', '119f_nocwv', '119f_pra', '125f',
                        '297_nocwv',
                        '297f', '297f_microp', '297f_pra', '297f_sp',
                        'cesm20f', 'cesm20f_microp', 'cesm20f_pra', 'cesm20f_sp']
        woelfleClimoDir = '/gpfs/fs1/work/woelfle/cesm1to2/climos/'
        woelfleClimoSubDir = ''
        woelfleRawDir = '/gpfs/fs1/scratch/woelfle/archive/'
        woelfleRawSubDir = 'atm/hist/'

        for vid in versionIds:
            if vid in cecileCases:
                loadFileLists[vid] = [cecileDir + 
                                      fileBaseDict[vid] + '/' +
                                      cecileSubDir +
                                      fileBaseDict[vid] +
                                      loadSuffix
                                      for loadSuffix in getloadsuffix(vid,
                                                                      climo_flag=True)]
            elif vid in releaseCases:
                loadFileLists[vid] = [(releaseClimoDir if climo_flag else releaseDir) + 
                                      fileBaseDict[vid] + '/' +
                                      (releaseClimoSubDir if climo_flag else releaseSubDir) +
                                      fileBaseDict[vid] +
                                      loadSuffix
                                      for loadSuffix in getloadsuffix(vid,
                                                                      climo_flag=climo_flag)]
            else:
                loadFileLists[vid] = [(woelfleClimoDir if climo_flag else woelfleRawDir) + 
                                      fileBaseDict[vid] + '/' +
                                      (woelfleClimoSubDir if climo_flag else woelfleRawSubDir) +
                                      fileBaseDict[vid] +
                                      loadSuffix
                                      for loadSuffix in getloadsuffix(vid,
                                                                      climo_flag=climo_flag,
                                                                      yr0=yr0,
                                                                      nyrs=nyrs)]

                # Ensure climos exist
                if all([not os.path.isfile(loadFileLists[vid][0]), climo_flag]):
                    print('Cannot find climo for {:s} loading raw output files instead.'.format(vid))
                    print(loadFileLists[vid][0])
                    loadFileLists[vid] = [
                        woelfleRawDir + 
                        fileBaseDict[vid] + '/' +
                        woelfleRawSubDir +
                        fileBaseDict[vid] +
                        loadSuffix
                        for loadSuffix in getloadsuffix(vid,
                                                        climo_flag=False,
                                                        yr0=yr0,
                                                        nyrs=nyrs)]

    return loadFileLists


def getloadsuffix(vid,
                  climo_flag=True,
                  nyrs=1,
                  yr0=2):
    """
    Get file endings for a single case for constructing file names when loading
    """

    loadSuffix = (['_' + '{:02d}'.format(mon + 1) + '_climo.nc'
                   for mon in range(12)]
                  if climo_flag
                  else ['.cam.h0.' + '{:04d}'.format(yr + yr0) +
                        '-{:02d}'.format(mon+1) + '.nc'
                        for mon in range(12)
                        for yr in range(nyrs)
                        ]
                  )
    
    return loadSuffix


def getmarkerdict():
    return {'01': 'o',
            '28': 'o',
            '36': 'o',
            'ga7.66': 'o',
            '100': 'o',
            '113': 'o',
            '114': 'o',
            '116': 'o',
            '118': 'o',
            '119': 'o',
            '119f': 's',
            '119f_gamma': '^',
            '119f_ice': '*',
            '119f_liqss': 'd',
            '119f_microp': 'v',
            '119f_nocwv': '<',
            '119f_pra': '<',
            '125': 'o',
            '125f': 's',
            '161': 'o',
            '194': 'o',
            '195': 'o',
            '297': 'o',
            '297_nocwv': '<',
            '297f': 's',
            '297f_microp': 'v',
            '297f_pra': '<',
            '297f_sp': '+',
            'cesm20f': 's',
            'cesm20f_microp': 'v',
            'cesm20f_pra': '<',
            'cesm20f_sp': '+',
            'obs': 'o',
            }


def getmapcontlevels(plotVar,
                     diff_flag=False):
    """
    Determine contour values for given plotVar

    Author:
        Matthew Woelfle

    Version Date:
        2018-06-20

    Args:
        plotVar - name of variable (in CESM parlance) for which contours are to
            be retrieved

    Kwargs:
        diff_flag - true if plotting difference in variable
    """
    if diff_flag:
        try:
            levels = {'CLDHGH': np.arange(-0.2, 0.21, 0.02),
                      'CLDLOW': np.arange(-0.5, 0.51, 0.05),
                      'CLDMED': np.arange(-0.5, 0.51, 0.05),
                      'CLDTOT': np.arange(-0.5, 0.51, 0.05),
                      'CLOUD': np.arange(-0.25, 0.251, 0.025),
                      'FLNS': np.arange(-30., 30.1, 3),
                      'FLUT': np.arange(-15, 15.1, 1.5),
                      # 'FNS': np.arange(-600., 600.1, 100),
                      'FNS': np.arange(-50, 50.1, 5),
                      'FSNS': np.arange(-50, 50.1, 5.),
                      'LHFLX': np.arange(-50, 50.1, 5),
                      'LWCF': np.arange(-20, 20.1, 2),
                      # 'OMEGA': np.arange(-0.12, 0.12001, 0.01),
                      'OMEGA': np.arange(-0.01, 0.01001, 0.001),
                      'OMEGA500': np.arange(-0.125, 0.1251, 0.0125),
                      'OMEGA850': np.arange(-0.125, 0.1251, 0.0125),
                      'PBLH': np.arange(-150, 150.1, 15),
                      'PRECC': np.arange(-10, 10.1, 1),
                      'PRECL': np.arange(-10, 10.1, 1),
                      'PRECT': np.arange(-5, 5.01, 0.5),
                      'PS': np.arange(-4., 4.01, 0.5),
                      'PSL': np.arange(-4, 4.01, 0.5),
                      'RELHUM': np.arange(-10, 10.1, 1),
                      'SHFLX': np.arange(-10, 10.1, 1.),
                      'SWCF': np.arange(-50, 50.1, 5),
                      'T': np.arange(-2, 2.1, 0.2),
                      'TAUX': np.arange(-0.1, 0.101, 0.01),
                      'TAUY': np.arange(-0.1, 0.101, 0.01),
                      'TGCLDIWP': np.arange(-0.03, 0.0301, 0.003),
                      'TGCLDLWP': np.arange(-0.03, 0.0301, 0.003),
                      'TMQ': np.arange(-10, 10.1, 1),
                      'TS': np.arange(-2, 2.1, 0.2),
                      'U': np.arange(-5, 5.1, 0.5),
                      'V': np.arange(-2, 2.1, 0.2),
                      'U10': np.arange(-2, 2.1, 0.2),
                      'curlTau': np.arange(-1.5e-7, 1.51e-7, 1.5e-8),
                      'curlTau_y': np.arange(-4e-13, 4.01e-13, 4e-14),
                      'divTau': np.arange(-1e-7, 1.01e-7, 1e-8),
                      'ekmanx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'ekmany': np.arange(-1e4, 1.01e4, 1e3),
                      'iews': np.arange(-0.1, 0.101, 0.01),
                      'inss': np.arange(-0.1, 0.101, 0.01),
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
            levels = {'CLDHGH': np.arange(0, 0.51, 0.025),
                      'CLDLOW': np.arange(0, 0.51, 0.025),
                      'CLDMED': np.arange(0, 0.51, 0.025),
                      'CLDTOT': np.arange(0, 0.51, 0.025),
                      'CLOUD': np.arange(0, 0.51, 0.025),
                      'FLNS': np.arange(0., 120.1, 10),
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
                      'PSL': np.arange(1004, 1013.1, 1),
                      'RELHUM': np.arange(0., 101., 10.),
                      'SHFLX': np.arange(0, 20., 1.),
                      'T': np.arange(290, 305.1, 1),
                      'TAUX': np.arange(-0.2, 0.201, 0.02),
                      'TAUY': np.arange(-0.1, 0.101, 0.01),
                      'TS': np.arange(290, 305.1, 1),
                      'U': np.arange(-10, 10.1, 1.0),
                      'U10': np.arange(0, 10.1, 1),
                      'curlTau': np.arange(-3e-7, 3.001e-7, 3e-8),
                      'curlTau_y': np.arange(-4e-13, 4.01e-13, 4e-14),
                      'divTau': np.arange(-2e-7, 2.01e-7, 2e-8),
                      'ekmanx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'ekmany': np.arange(-3e4, 3.01e4, 3e3),
                      'iews': np.arange(-0.2, 0.201, 0.02),
                      'inss': np.arange(-0.1, 0.101, 0.01),
                      'precip': np.arange(0, 20.1, 2),
                      'sst': np.arange(290, 305, 1),
                      'sverdrupx': np.arange(-1.5e5, 1.501e5, 1.5e4),
                      'w': np.arange(-0.12, 0.12001, 0.01),
                      }[plotVar]
        except KeyError:
            levels = None

    return levels


def getncarmachlist(nchars=6):
    """
    Return list of prefixes for NCAR/UCAR machines
    - Only returns first nchars characters for each machine
    """

    ncarMachList = ['caldera', 'cheyenne', 'geyser', 'pronghorn', 'yslogin']
    
    return [machName[0:nchars] for machName in ncarMachList]


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


def getregriddedfilename(versionIds,
                         loadSuffixes,
                         climo_flag=True,
                         fileBaseDict=None,
                         nyrs=1,
                         yr0=2,
                         ):
    
    raise NotImplementedError('Not working yet. Too many things ' +
                              'to track for the way my code is ' +
                              'currently structured.')


def getuwmachlist(nchars=30):
    """
    Return list of machines at UW
    - only returns first nchars characters of the machine name
    """
    
    uwMachList = ['stable', 'challenger', 'p', 'fog']
    
    return [machName[0:nchars] for machName in uwMachList]


def getzonmeancontlevels(plotVar,
                         diff_flag=False):
    """
    Determine contour values for given plotVar

    Author:
        Matthew Woelfle

    Version Date:
        2018-06-18

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
        return [('{:s}'.format(yid)
                 if 'x' in yid else
                 'yrs_{:s}'.format(yid))
                for yid in yrIds]


def loadmodelruns(versionIds,
                  climoCases=None,
                  computeAll2DVars_flag=True,
                  ncDir=None,
                  ncSubDir=None,
                  newRuns_flag=False,
                  prect_flag=False,
                  fns_flag=False,
                  fnt_flag=False,
                  loadClimo_flag=True,
                  lts_flag=True,
                  regridVertical_flag=False,
                  verbose_flag=False,
                  **kwargs
                  ):
    """
    Load cases for the cesm1to2 ITCZ study
    """

    # Set cases to load from climatologies
    if climoCases is None:
        climoCases = [
            '01', '28', '36', 'ga7.66', '100', '113', '114', '116',
            '118', '119', '119f', '119f_gamma', '119f_microp',
            '125', '125f', '161', '194', '195'
            ]

    # Set file paths if not provided
    if ncDir is None:
        ncDir, ncSubDir, _ = setfilepaths()

    # Set which variables to compute in addition to history variables
    if computeAll2DVars_flag:
        prect_flag = True
        fns_flag = True
        fnt_flag = True

    # Get full case name for each versionId
    fileBaseDict = getcasebase()

    # Get full output file names for each version
    loadSuffixes = {
        versionIds[j]: (['_' + '{:02d}'.format(mon + 1) + '_climo.nc'
                         for mon in range(12)]
                        if all([versionIds[j] in climoCases, loadClimo_flag])
                        else ['.cam.h0.' + '{:04d}'.format(yr + 2) +
                              '-{:02d}'.format(mon+1) + '.nc'
                              for mon in range(12)
                              for yr in range(1)
                              ])
        for j in range(len(versionIds))
        }

    # Create list of files to load
    if newRuns_flag:
        loadFileLists = {versionIds[j]: [ncDir + fileBaseDict[versionIds[j]] +
                                         '/' +
                                         ncSubDir +
                                         fileBaseDict[versionIds[j]] +
                                         loadSuffix
                                         for loadSuffix in loadSuffixes]
                         for j in range(len(versionIds))}
    else:
        loadFileLists = getloadfilelists(versionIds,
                                         loadSuffixes,
                                         climo_flag=loadClimo_flag)

    # Open netcdf file(s)
    dataSets = {versionId: xr.open_mfdataset(loadFileLists[versionId],
                                             decode_times=False)
                for versionId in versionIds}

    # Add version id to dataSets for easy access and bookkeeping
    for vid in versionIds:
        dataSets[vid].attrs['id'] = vid

    # Obtain vertically regridded datasets from file or regrid (if requested)
    if regridVertical_flag:
        dataSets_rg = regriddatasets(
            dataSets,
            fileBaseDict=fileBaseDict,
            ncDir=ncDir,
            ncSubDir=ncSubDir,
            versionIds=versionIds,
            **kwargs
            )
    else:
        dataSets_rg = {vid: [] for vid in versionIds}

    # Compute extra variable fields as requested
    for vid in versionIds:
        if verbose_flag:
            print('Computing extra variables for {:s}'.format(vid))

        # Compute PRECT
        if prect_flag:
            dataSets[vid]['PRECT'] = mwfn.calcprectda(dataSets[vid])

        # Compute FNS
        if fns_flag:
            dataSets[vid]['FNS'] = mwfn.calcfnsda(dataSets[vid])

        # Compute FNT
        if fnt_flag:
            dataSets[vid]['FNT'] = mwfn.calcfntda(dataSets[vid])

        # Compute LTS
        if lts_flag:
            try:
                dataSets[vid]['LTS'] = mwfn.calcltsda(dataSets_rg[vid])
            except TypeError:
                print('Cannot compute LTS for {:s}'.format(vid))
        
    return dataSets, dataSets_rg


def loadobsdatasets(obsList=None,
                    ceresEbaf_flag=False,
                    erai_flag=False,
                    gpcp_flag=False,
                    hadIsst_flag=False,
                    hadIsstYrs=[1979, 2010],
                    whichHad='all',
                    ):
    """
    Load observational datasets for comparison with simulations
    """
    # Parse list of requested sources if provided
    if obsList is not None:
        obsList = [j.lower() for j in obsList]
        if 'ceresebaf' in obsList:
            ceresEbaf_flag = True
        if 'erai' in obsList:
            erai_flag = True
        if 'gpcp' in obsList:
            gpcp_flag = True
        if 'hadisst' in obsList:
            hadIsst_flag = True

    # Create dictionary for holding observed datasets
    obsDsDict = dict()

    # Load CERES-EBAF
    if ceresEbaf_flag:
        if gethostname() in getuwmachlist():
            raise NotImplementedError('CERES-EBAF not set to load yet.')
        elif gethostname()[0:6] in getncarmachlist(6):
            ceresDir = '/gpfs/p/cesm/amwg/amwg_data/obs_data/'
            ceresClimoFiles = [ceresDir + 
                               'CERES-EBAF_{:02d}_climo.nc'.format(mon + 1)
                               for mon in range(12)]

            # Load CERES for climo only (as this is all I can find easily)
            obsDsDict['ceresClimo'] = xr.open_mfdataset(ceresClimoFiles,
                                                        decode_times=False)
            obsDsDict['ceresClimo'].attrs['id'] = 'CERES_climo'
            obsDsDict['ceresClimo'].attrs['climo_yrs'] = '2000-2013'

    # Load ERA-I
    if erai_flag:
        if gethostname() in getuwmachlist():
            obsDsDict['erai'] = mwfn.loaderai(
                daNewGrid=None,
                kind='linear',
                loadClimo_flag=True,
                newGridFile=None,
                newGridName='0.9x1.25',
                newLat=None,
                newLon=None,
                regrid_flag=False,
                whichErai='monmean',
                )
            obsDsDict['erai3d'] = mwfn.loaderai(
                daNewGrid=None,
                kind='linear',
                loadClimo_flag=True,
                newGridFile=None,
                newGridName='0.9x1.25',
                newLat=None,
                newLon=None,
                regrid_flag=False,
                whichErai='monmean.3d',
                )
        elif gethostname()[0:6] in getncarmachlist(6):
            eraiDir = '/gpfs/p/cesm/amwg/amwg_data/obs_data/'
            eraiClimoFiles = [eraiDir + 
                              'ERAI_{:02d}_climo.nc'.format(mon + 1)
                              for mon in range(12)]

            # Load ERAI for climo only (as this is all I can find easily)
            obsDsDict['eraiClimo'] = xr.open_mfdataset(eraiClimoFiles,
                                                       decode_times=False)
            obsDsDict['eraiClimo'].attrs['id'] = 'ERAI_climo'
            obsDsDict['eraiClimo'].attrs['climo_yrs'] = '????'

    # Load GPCP
    if gpcp_flag:
        # Set load differently depending on system (UW vs UCAR)
        if gethostname() in getuwmachlist():
            # Set directories for GPCP
            gpcpDir = '/home/disk/eos9/woelfle/dataset/GPCP/climo/'
            gpcpFile = 'gpcp_197901-201012.nc'
            gpcpClimoFile = 'gpcp_197901-201012_climo.nc'

            # Load GPCP for all years and add id
            obsDsDict['gpcp'] = xr.open_dataset(gpcpDir + gpcpFile)
            obsDsDict['gpcp'].attrs['id'] = 'GPCP_all'

            # Load GPCP from both climo and add id
            obsDsDict['gpcpClimo'] = xr.open_dataset(gpcpDir + gpcpClimoFile)
            obsDsDict['gpcpClimo'].attrs['id'] = 'GPCP_climo'
        elif gethostname()[0:6] in getncarmachlist(6):
            gpcpDir = '/gpfs/p/cesm/amwg/amwg_data/obs_data/'
            gpcpClimoFiles = [gpcpDir + 
                              'GPCP_{:02d}_climo.nc'.format(mon + 1)
                              for mon in range(12)]
            
            # Load GPCP for climo only (as this is all I can find easily)
            obsDsDict['gpcpClimo'] = xr.open_mfdataset(gpcpClimoFiles,
                                                       decode_times=False)
            obsDsDict['gpcpClimo'].attrs['id'] = 'GPCP_climo'
            obsDsDict['gpcpClimo'].attrs['climo_yrs'] = '1979-2009'

    # Load HadISST
    if hadIsst_flag:
        # Set load differently depending on system (UW vs UCAR)
        if gethostname() in getuwmachlist():
            # Attempt to look at other averaging periods for HadISST
            obsDsDict['hadIsst'] = mwfn.loadhadisst(
                climoType='monthly',
                daNewGrid=None,
                kind='linear',
                newGridFile=None,
                newGridName='0.9x1.25',
                newLat=None,
                newLon=None,
                qc_flag=False,
                regrid_flag=True,
                whichHad=whichHad,
                years=hadIsstYrs,
                )
        elif gethostname()[0:6] in getncarmachlist(6):
            # Options for :
            #   _CL_ - 1992-2001 (climo)
            #   _PD_ - 1999-2008 (climo)
            #   _PI_ - 1870-1900 (climo)
            #   all - 1870-2005 (all)
            if whichHad in ['CL', 'PD', 'PI']:
                hadIsstDir = '/gpfs/p/cesm/amwg/amwg_data/obs_data/'
                hadIsstClimoFiles = [
                    hadIsstDir + 
                    'HadISST_{:s}_'.format(whichHad) +
                    '{:02d}_climo.nc'.format(mon + 1)
                    for mon in range(12)]

                # Load HadISST for climos
                obsDsDict['hadIsstClimo'] = xr.open_mfdataset(hadIsstClimoFiles,
                                                              decode_times=False)
                obsDsDict['hadIsstClimo'].attrs['id'] = 'HadISST_climo'
                obsDsDict['hadIsstClimo'].attrs['climo_yrs'] = {
                    'CL': '1992-2001',
                    'PD': '1999-2008',
                    'PI': '1870-1900'}[whichHad]
            elif whichHad == 'all':
                # Load full HadISST field
                hadISSTfile = ('/gpfs/p/cesm/amwg/amwg_data/obs_data/' + 
                               'sst.hadley.187001-200512.nc')
                obsDsDict['hadIsst'] = xr.open_dataset(hadISSTfile)
                obsDsDict['hadIsst'].attrs['id'] = 'HadISST'
                obsDsDict['hadIsst'].attrs['yrs'] = '1870-2005'
                
                if obsDsDict['hadIsst'].lon.values.min() < 0:
                    # Determine lenght of roll required
                    rollDist = np.sum(obsDsDict['hadIsst'].lon.values < 0)

                    # Roll entire dataset
                    hadIsstDs = obsDsDict['hadIsst'].roll(lon=rollDist)

                    # Update longitudes to be positive definite
                    hadIsstDs['lon'].values = np.mod(
                        hadIsstDs['lon'] + 360, 360)

                # Subset to requested years
                hadIsstDs_sub = hadIsstDs.loc[
                        dict(time=slice('{:4d}01'.format(hadIsstYrs[0]),
                                        '{:4d}31'.format(hadIsstYrs[1])))
                        ]
                
                # Update attributes
                hadIsstDs_sub.attrs['yrs'] = (
                    '{:4d}-{:4d}'.format(hadIsstYrs[0], hadIsstYrs[1]))
                hadIsstDs_sub.attrs['Comment'] = (
                    'Subsetted output for given years')
                
                # Compute monthly means (i.e. seasonal cycle)
                hadClimo = np.zeros([12,
                                     hadIsstDs_sub['SST'].values.shape[1],
                                     hadIsstDs_sub['SST'].values.shape[2]])
                nmon = hadIsstDs_sub['SST'].values.shape[0]
                for mon in range(12):
                    hadClimo[mon, :, :] = hadIsstDs_sub['SST'].values[
                        np.arange(mon, nmon, 12)].mean(axis=0)

                # Construct climatological data array
                obsDa = xr.DataArray(
                    hadClimo,
                    attrs=hadIsstDs_sub['SST'].attrs,
                    coords={'time': hadIsstDs_sub['SST'].coords['time'][0:12],
                            'lat': hadIsstDs_sub['SST'].coords['lat'],
                            'lon': hadIsstDs_sub['SST'].coords['lon']},
                    dims=hadIsstDs_sub['SST'].dims,
                    )
                # Transform to dataset for consistency
                obsDsDict['hadIsstClimo'] = xr.Dataset(
                    data_vars={'SST': obsDa})
                obsDsDict['hadIsstClimo'].attrs['climo_yrs'] = (
                    '{:4d}-{:4d}'.format(hadIsstYrs[0], hadIsstYrs[1]))
                obsDsDict['hadIsstClimo'].attrs['id'] = 'HadISST_climo'

    # Return datasets
    return obsDsDict


def plotbiasrelation(ds,
                     xIndex,
                     yIndex,
                     ds_rg=None,  # For vertically regridded when needed
                     legend_flag=True,
                     makeFigure_flag=False,
                     obsDsDict=None,
                     obsVarDict=None,
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
    # Set default index types
    indexTypes = {'cpacshear': 'testing',
                  'cti': 'Woelfleetal2017',
                  'ditcz': 'Bellucci2010',
                  'dpdy_epac': 'epac',
                  'dslp': 'DiNezioetal2013',
                  'dsstdy_epac': 'epac',
                  'sepsst': None,
                  'sepsst_raw': 'raw',
                  'walker': 'testing'}
    if xIndexType is None:
        xIndexType = indexTypes[xIndex.lower()]
    if yIndexType is None:
        yIndexType = indexTypes[yIndex.lower()]
    indexVars = {'cpacshear': 'U',
                 'cti': 'TS',
                 'ditcz': 'PRECT',
                 'dpdy_epac': 'PS',
                 'dslp': 'PSL',
                 'dsstdy_epac': 'TS',
                 'sepsst': 'TS',
                 'sepsst_raw': 'TS',
                 'walker': 'PS'}
    labelDict = {'cpacshear': 'Central Pacific Wind Shear' +
                              ' (850-200 hPa; {:s})'.format(
                                  ds[versionIds[0]]['U'].units),
                 'cti': 'Cold Tongue Index (K)',
                 'ditcz': 'Double-ITCZ Index (mm/d)',
                 'dpdy_epac': 'E. Pac. Meridional SST gradient (hPa)',
                 'dslp': 'SLP Gradient (hPa)',
                 'dsstdy_epac': 'E. Pac. Meridional SST gradient (K)',
                 'sepsst': 'SEP SST Index (K)',
                 'sepsst_raw': 'SEP SST (K)',
                 'walker': 'Walker Circulation Index (hPa)'}
    if plotObs_flag:
        #        obsDsDict = {'cpacshear': obsDsDict['cpacshear'],
        #                   'cti': obsDsDict['cti'],
        #                   'ditcz': obsDsDict['ditcz'],
        #                   'walker': obsDsDict['walker']}
        if obsVarDict is None:
            obsVarDict = {'cpacshear': 'u',
                          'cti': 'sst',
                          'ditcz': 'precip',
                          'dpdy_epac': 'sp',
                          'dslp': 'msl',
                          'dsstdy_epac': 'sst',
                          'sepsst': 'sst',
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
            indexVar=obsVarDict[xIndex.lower()],
            ocnOnly_flag=False)
        try:
            xMean['obs'] = xObsDa[xTSteps].mean(dim='time')
        except ValueError:
            xMean['obs'] = xObsDa[xTSteps].mean(dim='month')
        # print(xMean['obs'].values)

        # Second index
        yObsDa = calcregmeanindex(
            obsDsDict[yIndex.lower()],
            yIndex,
            indexType=indexTypes[yIndex.lower()],
            indexVar=obsVarDict[yIndex.lower()],
            ocnOnly_flag=False)
        try:
            yMean['obs'] = yObsDa[yTSteps].mean(dim='time')
        except ValueError:
            xMean['obs'] = yObsDa[yTSteps].mean(dim='month')
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


def plotmetricvsversion(indexName,
                        ds,
                        indexType=None,
                        legend_flag=True,
                        makeFigure_flag=True,
                        obsDs=None,
                        obsVar=None,
                        ocnOnly_flag=True,
                        plotAnnMean_flag=True,
                        plotPeriodMean_flag=True,
                        plotSeasCyc_flag=True,
                        plotObs_flag=True,
                        plotVar=None,
                        rmAnnMean_flag=False,
                        save_flag=False,
                        saveDir=None,
                        tSteps=np.arange(0, 12),
                        versionIds=None,
                        yLim_annMean=None,
                        yLim_periodMean=None,
                        yLim_seasCyc=None,
                        ):
    """
    Plot some predefined bias metric versus model version
    
    Available indices:
        'dITCZ', 'PAI', 'pcent', 'dpdy_epac', dsstdy_epac', 'fnsasym',
        'sepsst', 'sepsst_raw', 'walker'
    """

    # Set default plot values
    title = indexName
    
    # Set versions to plot
    if versionIds is None:
        versionIds = list(ds.keys())
    
    if indexName.lower() in ['cti', 'coldtongueindex']:
        if plotVar is None:
            plotVar = 'TS'
        if obsVar is None:
            obsVar = ('sst' if 'sst' in obsDs else 'SST')
        ocnOnly_flag = True
        title = 'CTI'
        if yLim_annMean is None:
            yLim_annMean = np.array([-1, 0.5])
        if yLim_periodMean is None:
            yLim_periodMean = np.array([-1, 0.5])
        if yLim_seasCyc is None:
            yLim_seasCyc = np.array([-1.75, 0.75])
    if indexName.lower() in ['ditcz']:
        if plotVar is None:
            plotVar = 'PRECT'
        if obsVar is None:
            obsVar = ('precip' if 'precip' in obsDs else 'PRECT')
        ocnOnly_flag = False
        title = 'Double-ITCZ Index'
        if yLim_annMean is None:
            yLim_annMean = np.array([1, 3])
        if yLim_periodMean is None:
            yLim_periodMean = np.array([1, 3])
        if yLim_seasCyc is None:
            yLim_seasCyc = np.array([0, 6])
    elif indexName.lower() in ['dpdy_epac']:
        if plotVar is None:
            plotVar = 'PS'
        if obsVar is None:
            obsVar = ('sp' if 'sp' in obsDs else 'PS')
        ocnOnly_flag = True
        title = 'dP/dy (E Pac)'
        if yLim_annMean is None:
            yLim_annMean = np.array([-1, 0])
        if yLim_periodMean is None:
            yLim_periodMean = np.array([-1, 1])
        if yLim_seasCyc is None:
            yLim = np.array([-1.5, 1.5])
    elif indexName.lower() in ['dsstdy_epac']:
        if plotVar is None:
            plotVar = 'TS'
        if obsVar is None:
            obsVar = ('sst' if 'sst' in obsDs else 'SST')
        ocnOnly_flag = True
        title = 'dSST/dy (E. Pac)'
        if yLim_annMean is None:
            yLim_annMean = np.array([0, 2])
        if yLim_periodMean is None:
            yLim_periodMean = np.array([-1.2, 1])
        if yLim_seasCyc is None:
            yLim_seasCyc = np.array([-3, 3])
    elif indexName.lower() in ['fnsasym']:
        if plotVar is None:
            plotVar = 'FNS'
        if obsVar is None:
            obsVar = None
        ocnOnly_flag = True
        title = 'FNS Asymmetry'
        if yLim_annMean is None:
            yLim_annMean = None
        if yLim_periodMean is None:
            yLim_periodMean = None
        if yLim_seasCyc is None:
            yLim_seasCyc = None
    elif indexName in ['PAI']:
        if plotVar is None:
            plotVar = 'PRECT'
        if obsVar is None:
            obsVar = ('precip' if 'precip' in obsDs else 'PRECT')
        ocnOnly_flag = False
        title = 'PAI'
        if yLim_annMean is None:
            yLim_annMean = np.array([0, 0.5])
        if yLim_periodMean is None:
            yLim_periodMean = np.array([-1.5, 1.5])
        if yLim_seasCyc is None:
            yLim_seasCyc = np.array([-1.5, 1.5])
    elif indexName.lower() in ['pcent']:
        if plotVar is None:
            plotVar = 'PRECT'
        if obsVar is None:
            obsVar = ('precip' if 'precip' in obsDs else 'PRECT')
        ocnOnly_flag = False
        title = 'Precipitation Centroid'
        if yLim_annMean is None:
            yLim_annMean = np.array([0, 2])
        if yLim_periodMean is None:
            yLim_periodMean = np.array([-10, 10])
        if yLim_seasCyc is None:
            yLim_seasCyc = np.array([-10, 10])
    elif indexName.lower() in ['sepacsst', 'sepsst', 'sepsst_raw']:
        if plotVar is None:
            plotVar = 'TS'
        if obsVar is None:
            obsVar = ('sst' if 'sst' in obsDs else 'SST')
        ocnOnly_flag = True
        if '_raw' in indexName.lower():
            indexName = indexName[:-4]
            indexType = 'raw'
        if indexType == 'raw':
            title = 'SE Pac. SST'
        else:
            title = 'SE Pac. SST index'
        if yLim_annMean is None:
            yLim_annMean = None
        if yLim_periodMean is None:
            yLim_periodMean = None
        if yLim_seasCyc is None:
            yLim_seasCyc = None
    elif indexName.lower() in ['walker']:
        if plotVar is None:
            plotVar = 'PSL'
        if obsVar is None:
            obsVar = ('sp' if 'sp' in obsDs else 'PSL')
        ocnOnly_flag = True
        title = 'dP/dx (Walker)'
        if yLim_annMean is None:
            yLim_annMean = None  # np.array([-1, 0])
        if yLim_periodMean is None:
            yLim_periodMean = None  # np.array([-1, 1])
        if yLim_seasCyc is None:
            yLim = None  # np.array([-1.5, 1.5])

    # Create dictionary to hold mean values
    annMean = dict()
    timeMean = dict()
    
    # Get line properties
    colorDict = getcolordict()
    markerDict = getmarkerdict()

    # Create figure for plotting
    if makeFigure_flag and plotSeasCyc_flag:
        hf = plt.figure()
        hf.set_size_inches(6, 4.5,
                           forward=True)

    # Get line handles for returning
    hlSeas = []

    for vid in versionIds:
        # Compute given index through time
        indexDa = calcregmeanindex(ds[vid],
                                   indexName,
                                   indexType=indexType,
                                   indexVar=plotVar,
                                   ocnOnly_flag=ocnOnly_flag,
                                   )

        # Get index values and remove annual mean if requested
        pData = (indexDa.values - indexDa.mean(dim='time').values
                 if rmAnnMean_flag
                 else indexDa.values)
        if plotSeasCyc_flag:
            hl, = plt.plot(np.arange(1, 13),
                           pData,
                           color=colorDict[vid],
                           label=vid,
                           marker=markerDict[vid],
                           )
            hlSeas.append(hl)
        annMean[vid] = indexDa.mean(dim='time')
        timeMean[vid] = indexDa.values[tSteps].mean()

    # Repeat above for obs
    if plotObs_flag:
        # Compute given index through time
        obsIndexDa = calcregmeanindex(obsDs,
                                      indexName,
                                      indexType=indexType,
                                      indexVar=obsVar,
                                      ocnOnly_flag=False,
                                      qc_flag=False,
                                      )
        
        # Get data for index
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

        if plotSeasCyc_flag:
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
            hlSeas.append(hl)

        # Compute annual means
        try:
            annMean['obs'] = obsIndexDa.mean(dim='time')
        except ValueError:
            annMean['obs'] = obsIndexDa.mean(dim='month')

        # Compute mean over given timesteps
        timeMean['obs'] = pData[tSteps].mean()

    if plotSeasCyc_flag:
        # Dress plot
        plt.xticks(np.arange(1, 13))
        plt.xlabel('Month')

        try:
            plt.ylabel('{:s}'.format(title) +
                       (' ({:s})'.format(indexDa.units)
                        if indexDa.units is not None
                        else '') +
                       ('\n[Annual mean removed]' if rmAnnMean_flag else '')
                       )
        except AttributeError as err:
            print(indexDa)
            raise AttributeError(err)
        try:
            plt.ylim(yLim_seasCyc)
        except NameError:
            pass
        if legend_flag:
            plt.legend(title='Version', ncol=2)

        plt.title('Seasonal cycle of {:s}'.format(title) +
                  ('\n[Annual mean removed]' if rmAnnMean_flag else '')
                  )

        # Add annotation of years used to compute climatology
        if False:  # all([plotVar == 'TS', plotObs_flag]):
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
        if makeFigure_flag:
            plt.tight_layout()

        # Save figure if requested
        if save_flag:
            # Set directory for saving
            if saveDir is None:
                saveDir = setfilepaths()[2]

            # Set file name for saving
            tString = 'mon'
            saveFile = ('seascyc_' + indexName.lower())

            # Set saved figure size (inches)
            fx, fy = hf.get_size_inches()

            # Save figure
            print(saveDir + saveFile)
            mwp.savefig(saveDir + saveFile,
                        shape=np.array([fx, fy]))
            plt.close('all')

    # Plot annual mean values
    if plotAnnMean_flag:

        # Make figure for plotting if needed
        if makeFigure_flag:
            hf = plt.figure()
            hf.set_size_inches(6, 4.5, forward=True)
        
        # Plot annual mean index vs model version
        for idx, vid in enumerate(versionIds):
            plt.scatter(idx + 1,
                        np.array(annMean[vid]),
                        marker=markerDict[vid],
                        c=colorDict[vid],
                        s=80,
                        )
        if plotObs_flag:
            plt.scatter([len(annMean)],
                        annMean['obs'],
                        marker=markerDict['obs'],
                        c=colorDict['obs'],
                        s=80,
                        )

        # Dress plot
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
        
        if makeFigure_flag:
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
        
        # Create figure for plotting
        if makeFigure_flag:
            plt.figure()
            hf.set_size_inches(6, 4.5, forward=True)

        for indx, vid in enumerate(versionIds):
            plt.scatter(indx + 1,
                        np.array(timeMean[vid]),
                        marker=markerDict[vid],
                        c=colorDict[vid],
                        s=80,
                        )
        if plotObs_flag:
            plt.scatter([len(timeMean)],
                        timeMean['obs'],
                        marker=markerDict['obs'],
                        c=colorDict['obs'],
                        s=80,
                        )

        # Dress plot
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
        
        if makeFigure_flag:
            plt.tight_layout()
        
        # Save figure if requested
        if save_flag:
            # Set directory for saving
            if saveDir is None:
                saveDir = setfilepaths()[2]

            # Set file name for saving
            tString = 'mon'
            saveFile = ('pdmean_' + tStepString +
                        indexName.lower())

            # Set saved figure size (inches)
            fx, fy = hf.get_size_inches()

            # Save figure
            print(saveDir + saveFile)
            mwp.savefig(saveDir + saveFile,
                        shape=np.array([fx, fy]))
            plt.close('all')

    return hlSeas


def plotmultimetricvsversion(
    indexNames,
    ds,
    figSize=None,
    obsDsDict=None,
    save_flag=False,
    saveDir=None,
    **kwargs
    ):
    """
    Plot multiple metrics on one figure.
    Assumes only one of plotAnnMean_flag, plotPeriodMean_flag,
        and plotSeasCyc_flag will be True
    """
    
    if len(indexName) == 2:
        hf = plt.figure()
        if figSize is None:
            hf.set_size_inches(6, 4.5, forward=True)
        else:
            hf.set_size_inches(figSize[0],
                               figSize[1],
                               forward=True)

    for indexName in indexNames:
        try:
            obsDs = obsDsDict[indexName]
        except TypeError:
            obsDs = None
        plotmetricvsversion(indexName,
            ds,
            makeFigure_flag=True,
            obsDs=obsDsDict[indexName],
            obsVar=None,
            ocnOnly_flag=True,
            plotAnnMean_flag=True,
            plotPeriodMean_flag=True,
            plotSeasCyc_flag=True,
            plotObs_flag=True,
            plotVar=None,
            rmAnnMean_flag=False,
            save_flag=False,
            saveDir=None,
            tSteps=np.arange(0, 12),
            versionIds=None,
            yLim_annMean=None,
            yLim_periodMean=None,
            yLim_seasCyc=None,
            )


def plotprecipcentroidvlon(dsList,
                           varList,
                           closeOnSaving_flag=True,
                           contCmap='RdYlGn',
                           diff_flag=True,
                           makeFigure_flag=True,
                           refDs=None,
                           refVar=None,
                           saveDir=None,
                           save_flag=False,
                           yLim=None,
                           ):
    """
    Create plot of precipitation centroid at each longitude
    """

    # Create figure if needed
    if makeFigure_flag:
        hf = plt.figure()
        hf.set_size_inches(10, 3*len(dsList) + 1, forward=True)
    else:
        hf = plt.gcf()

    # Create suplot axes
    gs = gridspec.GridSpec(len(dsList), 2,
                           width_ratios=[10, 1],
                           wspace=0,
                           )

    # Ensure varList is a list
    # if isinstance(varList, str):
    #     varList = [varList]*len(dsList)

    for (jDs, ds) in enumerate(dsList):
        # Compute centroid as fn(longitude) data arrays
        try:
            centDa = mwfn.calcdslonprecipcentroid(ds,
                                                  indexType='areaweight',
                                                  precipVar=varList[jDs],
                                                  qc_flag=False,
                                                  )
        except KeyError:
            centDa = mwfn.calcdslonprecipcentroid(ds,
                                                  indexType='areaweight',
                                                  precipVar=varList,
                                                  qc_flag=False,
                                                  )

        centVals = centDa.values.copy()

        if diff_flag:
            try:
                refCentDa = mwfn.calcdslonprecipcentroid(
                    refDs,
                    indexType='areaweight',
                    precipVar=refVar,
                    qc_flag=False,
                    )
                refCentVals = refCentDa.values.copy()
            except NameError:
                raise NameError('Must provide refDS and refVar to difference')

        # Subset centVals if needed to match spacing for differencing
        #   ONLY WORKS WITH GPCP <--> CESM 1 DEGREE!!
        if diff_flag:
            if centVals.shape[1] > refCentVals.shape[1]:
                centVals = centVals[:, np.arange(1, centVals.shape[1], 2)]

        # Get colors for plotting
        colorIdx = range(centVals.shape[0])
        cm = plt.get_cmap(contCmap)
        cNorm = colors.Normalize(vmin=0, vmax=colorIdx[-1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

        # Pull longitudes for plotting
        if diff_flag:
            lon = refDs.lon
        else:
            lon = ds.lon

        # Plot lon vs dlat plot of centroids for each month
        plt.subplot(gs[jDs, 0])
        for mon in range(12):
            colorVal = scalarMap.to_rgba(colorIdx[mon])
            plt.plot(lon.values,
                     centVals[mon, :] -
                     (refCentVals[mon, :] if diff_flag else 0),
                     color=colorVal,
                     label=calendar.month_abbr[colorIdx[mon] + 1],
                     )

        # Plot lon vs dlat plot of centroids for annual mean
        plt.plot(lon.values,
                 centVals.mean(axis=0) -
                 (refCentVals.mean(axis=0) if diff_flag else 0),
                 '--k',
                 label='Ann. Mean')

        # Dress plot
        plt.xlim([0, 360])
        plt.xlabel('Longitude')
        plt.xticks(np.arange(0, 361, 30))
        plt.ylabel(ds.id +
                   (('-' + refDs.id) if diff_flag else '') +
                   ' (deg. latitude)')
        if yLim is None:
            yLim = plt.ylim()
            yLim = [-np.max(np.abs(yLim)), np.max(np.abs(yLim))]
        plt.ylim(yLim)
        plt.title(('Bias in p' if diff_flag else 'P') +
                  'recipitation centroid as function of longitude\n' +
                  '(' + ds.id +
                  (('-' + refDs.id) if diff_flag else '') +
                  ')')
        plt.grid()

        # Get yticks for second subplot
        yTicks = plt.yticks()

        # Add legend to last row only
        if jDs == (len(dsList) - 1):
            plt.legend(ncol=7, loc=4)

        # Plot zonal mean reference lines of dlat for centroids
        ax2 = plt.subplot(gs[jDs, 1])

        for mon in range(12):
            colorVal = scalarMap.to_rgba(colorIdx[mon])

            plt.plot([0, 1],
                     [centVals[mon, :].mean() -  # *2 to plot line
                      (refCentVals[mon, :].mean() if diff_flag else 0)]*2,
                     color=colorVal,
                     label=None,
                     )
        plt.plot([0, 1],
                 [centVals.mean(axis=0).mean() -
                  (refCentVals.mean(axis=0).mean() if diff_flag else 0)]*2,
                 '--k',
                 label=None)
        plt.xlim([-1, 2])
        plt.xticks([])
        plt.yticks(yTicks[0])

        plt.ylim(yLim)
        ax2.yaxis.set_ticklabels([])

        plt.grid()
        for tic in ax2.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
        plt.title('Zonal\nMean')

    # Force everything to fit on the plot
    plt.tight_layout()

    # Save figure if requested
    if save_flag:
        # Set directory for saving
        if saveDir is None:
            saveDir = os.path.dirname(os.path.realpath(__file__))

        # Set file name for saving
        saveFile = (varList[0] +
                    '_centrVlon_' +
                    '_'.join([dsList[j].id for j in range(len(dsList))])
                    )
        if diff_flag:
            saveFile = ('d' + saveFile +
                        '_minus' + refDs.id
                        )

        # Set saved figure size (inches)
        fx = hf.get_size_inches()[0]
        fy = hf.get_size_inches()[1]

        # Save figure
        print(saveDir + saveFile)
        mwp.savefig(saveDir + saveFile,
                    shape=np.array([fx, fy]))
        if closeOnSaving_flag:
            plt.close('all')

    return hf


def getlatlonpdata(ds,
                   plotVar,
                   tSteps,
                   diff_flag=False,
                   diffAsPct_flag=False,
                   diffDs=None,
                   diffPlev=None,
                   diffTSteps=None,
                   diffVar=None,
                   ocnOnly_flag=False,
                   plev=None,
                   qc_flag=False,
                   quiver_flag=False,
                   quiverDs=None,
                   quiverDiffDs=None,
                   quiverNorm_flag=False,
                   quiverScaleVar=None,
                   rmRegMean_flag=False,
                   rmRegLatLim=None,
                   rmRegLonLim=None,
                   uVar=None,
                   vVar=None,
                   ):
    """
    Get array for plotitng a map
    ** Started, but not implemented **
    """
    
    if diffPlev is None:
        diffPlev = plev
    if diffTSteps is None:
        diffTSteps = tSteps
    if diffVar is None:
        diffVar = plotVar
    if quiverDs is None:
        quiverDs = ds
    if quiverDiffDs is None:
        quiverDiffDs = diffDs

    # Pull data for plotting color contours
    if np.ndim(ds[plotVar]) == 3:
        if diff_flag:
            pData = (ds[plotVar].values[tSteps, :, :].mean(axis=0) -
                     diffDs[diffVar].values[diffTSteps, :, :].mean(axis=0))
            if diffAsPct_flag:
                pData = (
                    pData /
                    diffDs[diffVar].values[diffTSteps, :, :].mean(axis=0)
                    )
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
            if diffAsPct_flag:
                pData = (
                    pData /
                    diffDs[diffVar].values[diffTSteps, kPlev, :, :].mean(axis=0)
                    )
        else:
            pData = ds[plotVar].values[tSteps, jPlev, :, :].mean(axis=0)

    # Filter to only values over ocean

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
    
    return (pData, uData, vData)


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
               diffAsPct_flag=False,
               diffDs=None,
               diffPlev=None,
               diffTSteps=None,
               diffVar=None,
               fontSize=12,
               figDims=None,
               latLim=np.array([-30, 30]),
               latlbls=None,
               levels=None,
               lineCont_flag=False,
               lineContDiff_flag=False,
               lineContDiffAsPct_flag=False,
               lineContLevels=None,
               lineContVar=None,
               lineContDiffVar=None,
               lineContUseDiffDs_flag=False,
               lonLim=np.array([119.5, 270.5]),
               lonlbls=None,
               makeFigure_flag=True,
               newUnits=None,
               ocnOnly_flag=False,
               plev=None,
               qc_flag=False,
               quiver_flag=False,
               quiverDs=None,
               quiverDiffDs=None,
               quiverKey_flag=True,
               quiverNorm_flag=False,
               quiverScale=0.4,
               quiverScaleVar=None,
               quiverUnits='inches',
               returnM_flag=True,  # MUST be true for now (bug)
               rmRegLatLim=None,
               rmRegLonLim=None,
               rmRegMean_flag=False,
               rmse_flag=False,
               save_flag=False,
               saveDir=None,
               stampDate_flag=True,
               subSamp=None,
               tSteps=None,
               tStepLabel_flag=True,
               useDiffDs_flag=False,
               uRef=0.1,
               uVar='TAUX',
               vVar='TAUY',
               verbose_flag=False,
               **kwargs
               ):
    """
    Plot a map of a given dataset averaged over the specified timesteps

    Version Date:
        2018-03-13
    """

    # Set lats/lons to label if not provided
    if latlbls is None:
        latlbls = mwp.getlatlbls(latLim)
    if lonlbls is None:
        lonlbls = mwp.getlonlbls(lonLim)

    # Get levels for contouring if not provided
    if levels is None:
        if diffAsPct_flag:
            levels = np.arange(-1, 1.001, 0.1)
        else:
            levels = getmapcontlevels(
                (plotVar  # +
                 # (plev if np.ndim(ds[plotVar]) == 4
                 #  else '')
                 ),
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
    if verbose_flag:
        print(tSteps)
        if diff_flag:
            print(diffTSteps)

    # Determine differencing pressure level if not provided
    if diffPlev is None:
        diffPlev = plev

    # Set caseString for plotting
    if caseString is None:
        if any([diff_flag, useDiffDs_flag]):
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

    # Pull data for plotting color contours
    (pData, uData, vData) = getlatlonpdata((diffDs
                                            if useDiffDs_flag
                                            else ds),
                                           plotVar,
                                           tSteps,
                                           diff_flag=diff_flag,
                                           diffAsPct_flag=diffAsPct_flag,
                                           diffDs=diffDs,
                                           diffPlev=diffPlev,
                                           diffTSteps=diffTSteps,
                                           diffVar=diffVar,
                                           ocnOnly_flag=ocnOnly_flag,
                                           plev=plev,
                                           qc_flag=qc_flag,
                                           quiver_flag=quiver_flag,
                                           quiverDs=quiverDs,
                                           quiverDiffDs=quiverDiffDs,
                                           quiverNorm_flag=quiverNorm_flag,
                                           quiverScaleVar=quiverScaleVar,
                                           rmRegMean_flag=rmRegMean_flag,
                                           rmRegLatLim=rmRegLatLim,
                                           rmRegLonLim=rmRegLonLim,
                                           uVar=uVar,
                                           vVar=vVar,
                                           )
    
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

    # Create figure for plotting
    if makeFigure_flag:
        hf = plt.figure()
        if figDims is not None:
            hf.set_size_inches(figDims)
        hf.canvas.set_window_title(ds.id +
                                   ('-' + diffDs.id
                                    if diff_flag else '') +
                                   ': ' + plotVar + ' (latlon)')

    # Plot map
    im1, ax, hMap = mwp.plotmap(ds.lon,
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
                                extend=['both', 'max'][plotVar in
                                                       maxExtendVars],
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
                                quiverKey_flag=((not quiverNorm_flag) and
                                                quiverKey_flag),
                                quiverScale=quiverScale,
                                quiverUnits=quiverUnits,
                                returnM_flag=returnM_flag,
                                U=uData,
                                Uname=(quiverDs[uVar].name
                                       if uVar in quiverDs.data_vars
                                       else None),
                                Uunits=((quiverDs[quiverScaleVar].units
                                         if quiverScaleVar is not None else
                                         quiverDs[uVar].units)
                                        if uVar in quiverDs.data_vars
                                        else None),
                                Uref=uRef,  # 0.1,
                                V=vData,
                                subSamp=((3 if subSamp is None else subSamp)
                                         # ds['TAUX'].shape[1]/36
                                         if uVar in quiverDs.data_vars
                                         else None),
                                tStepLabel_flag=False,
                                **kwargs
                                )
    
    # Add overlayed line contours
    if lineCont_flag:
        
        if lineContVar is None:
            lineContVar = plotVar
        if lineContDiffVar is None:
            lineContDiffVar = diffVar
        if lineContUseDiffDs_flag is True:
            lineContDs = diffDs
        else:
            lineContDs = ds

        # Get levels for contouring if not provided
        if lineContLevels is None:
            if lineContDiffAsPct_flag:
                lineContLevels = np.arange(-1, 1.001, 0.1)
            else:
                lineContLevels = getmapcontlevels(
                    (lineContVar  # +
                     # (plev if np.ndim(ds[plotVar]) == 4
                     #  else '')
                     ),
                    diff_flag=any([lineContDiff_flag,
                                   rmRegMean_flag])
                    )[::2]
        
        if lineContDiff_flag:
            if diffDs is None:
                raise TypeError('diffDs is None...')
        
        lineContData, _, _ = getlatlonpdata(lineContDs,
                                            lineContVar,
                                            tSteps,
                                            diff_flag=lineContDiff_flag,
                                            diffAsPct_flag=lineContDiffAsPct_flag,
                                            diffDs=diffDs,
                                            diffPlev=diffPlev,
                                            diffTSteps=diffTSteps,
                                            diffVar=lineContDiffVar,
                                            ocnOnly_flag=ocnOnly_flag,
                                            plev=plev,
                                            rmRegMean_flag=rmRegMean_flag,
                                            rmRegLatLim=rmRegLatLim,
                                            rmRegLonLim=rmRegLonLim,
                                            )
        
        # Create lat/lon meshed grids
        lonG, latG = np.meshgrid(ds.lon, ds.lat)
        
        CS = hMap.contour(lonG, latG,
                     lineContData,
                     levels=lineContLevels,
                     # extend=False,
                     latlon=True,
                     colors='k'
                     )
        plt.clabel(CS, fontsize=9, inline=1)
        
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

    # Save figure if requested
    if save_flag:

        # Set directory for saving
        if saveDir is None:
            saveDir = os.path.dirname(os.path.realpath(__file__))

        # Set file name for saving
        tString = 'mon'
        if diff_flag:
            # Get variable name for saving
            varName = plotVar
            if np.ndim(ds[plotVar]) == 4:
                # Add level if original field is 4d
                varName = varName + str(plev)
            if plev != diffPlev:
                # Add differencing of levels if plotting differences and
                #   plev and diffPlev are not the same (for plotting shears)
                varName = varName + '-' + str(diffPlev)

                # Set name of case
                caseString = ds.id
            else:
                caseString = ds.id + '-' + diffDs.id

            saveFile = (varName +
                        '_latlon_' +
                        caseString + '_' +
                        tString +
                        '{:03.0f}'.format(tSteps[0]) + '-' +
                        '{:03.0f}'.format(tSteps[-1]))
        else:
            saveFile = (
                plotVar + '_latlon_' +
                ds.id + '_' +
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

    if returnM_flag:
        return (im1, ax, compcont, hMap)
    else:
        return (im1, ax, compcont)


def plotmultilatlon(dsDict,
                    plotIdList,
                    plotVar,
                    box_flag=False,
                    cbar_flag=True,
                    cbarOrientation='vertical',
                    compcont_flag=True,
                    diff_flag=False,
                    diffAsPct_flag=False,
                    diffIdList=None,
                    diffDs=None,
                    diffPlev=None,
                    diffVar=None,
                    figSize=None,
                    fontSize=12,
                    latLim=np.array([-30, 30]),
                    latlbls=None,
                    lonLim=np.array([119.5, 270.5]),
                    lonlbls=None,
                    obsDs=None,
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

    # Set obs properties if needed
    if 'obs' in plotIdList:
        obsVar = {'OMEGA': 'w',
                  'PRECT': 'precip',
                  'TAUX': 'iews',
                  'TS': 'sst',
                  }[plotVar]
        # obsQuivDs = {'TAUX': eraiDs,
        #              'U': erai3dDs}[uVar]
        # obsUVar = {'TAUX': 'iews',
        #            'U': 'u',
        #            }[uVar]
        # obsVVar = {'TAUY': 'inss',
        #            'V': 'v',
        #            }[vVar]

    # Create figure for plotting
    hf = plt.figure()
    if len(plotIdList) == 2:
        if cbarOrientation == 'vertical':
            # Set figure window size
            hf.set_size_inches(9, 2, forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(1, 3,
                                   # height_ratios=[20, 1, 20, 1, 20],
                                   # hspace=0.3,
                                   width_ratios=[30, 30, 1],
                                   )
            gs.update(left=0.07, right=0.95, top=0.95, bottom=0.05)

            # Set gridspec colorbar location
            cbColInd = 2
            cbRowInd = 0
            cbar_xoffset = -0.04

        elif cbarOrientation == 'horizontal':
            # Set figure window size
            hf.set_size_inches(7.5, 4, forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(2, 2,
                                   height_ratios=[20, 1],
                                   )

            # Set gridspec colorbar location
            cbColInd = 0
            cbRowInd = 1

        # Set gridpsec index order
        colInds = [0, 1]
        rowInds = [0, 0]
    if len(plotIdList) == 3:
        if cbarOrientation == 'vertical':
            # Set figure window size
            hf.set_size_inches(9, 8, forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(3, 2,
                                   # height_ratios=[20, 1, 20, 1, 20],
                                   hspace=0.3,
                                   width_ratios=[30, 1],
                                   )
            gs.update(left=0.07, right=0.95, top=0.95, bottom=0.05)

            # Set gridspec colorbar location
            cbColInd = 1
            cbRowInd = 0
            cbar_xoffset = -0.04

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
        rowInds = [0, 1, 2]

    elif len(plotIdList) == 4:
        if cbarOrientation == 'vertical':
            # Set figure window size
            if figSize is None:
                hf.set_size_inches(9, 3, forward=True)
            else:
                hf.set_size_inches(figSize[0], figSize[1],
                                   forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(2, 3,
                                   # height_ratios=[20, 1, 20, 1, 20],
                                   hspace=0.1,
                                   width_ratios=[30, 30, 1],
                                   )
            gs.update(left=0.05, right=0.92, top=0.95, bottom=0.05)

            # Set gridpsec index order
            colInds = [0, 1, 0, 1]
            rowInds = [0, 0, 1, 1]
            cbar_xoffset = -0.04

            # Set gridspec colorbar location
            cbColInd = 2
            cbRowInd = 0

    elif len(plotIdList) == 6:
        if cbarOrientation == 'vertical':
            # Set figure window size
            hf.set_size_inches(10, 6, forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(3, 3,
                                   # height_ratios=[20, 1, 20, 1, 20],
                                   hspace=0.3,
                                   width_ratios=[30, 30, 1],
                                   )
            gs.update(left=0.05, right=0.92, top=0.95, bottom=0.05)

            # Set gridpsec index order
            colInds = [0, 0, 0, 1, 1, 1]
            rowInds = [0, 1, 2, 0, 1, 2]

            # Set gridspec colorbar location
            cbColInd = 2
            cbRowInd = 0
            cbar_xoffset = -0.02
    
            # Set gridspec colorbar location
            cbColInd = 3
            cbRowInd = 0
            cbar_xoffset = -0.01
    elif len(plotIdList) == 8:
        if cbarOrientation == 'vertical':
            # Set figure window size
            if figSize is None:
                if np.diff(latLim) >= 50:
                    hf.set_size_inches(10, 6.5, forward=True)
                else:
                    hf.set_size_inches(10, 6.5, forward=True)
            else:
                hf.set_size_inches(figSize[0], figSize[1],
                                   forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(4, 3,
                                   height_ratios=[1, 1, 1, 1],
                                   hspace=0.05,
                                   width_ratios=[30, 30, 1],
                                   wspace=0.2,
                                   left=0.04,
                                   right=0.96,
                                   bottom=0.00,
                                   top=1.0,
                                   )

            # Set gridspec colorbar location
            cbColInd = 2
            cbRowInd = 0
            cbar_xoffset = -0.01
            
        # Set gridspec index order
        colInds = [0, 1, 0, 1, 0, 1, 0, 1]
        rowInds = [0, 0, 1, 1, 2, 2, 3, 3]
            
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
            cbar_xoffset = -0.01

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
    skippedPlotCount = 0
    for jSet, plotId in enumerate(plotIdList):
        if plotId is None:
            skippedPlotCount = skippedPlotCount + 1
            print('skipping {:d}'.format(jSet))
            continue
        plt.subplot(gs[rowInds[jSet], colInds[jSet]])
        if diff_flag:
            diffId = diffIdList[jSet]
            # print(plotId + ' - ' + diffId)
            im1, ax, compcont, _ = plotlatlon(
                dsDict[plotId],
                plotVar,
                box_flag=box_flag[jSet],
                cbar_flag=False,
                compcont_flag=compcont_flag,
                diff_flag=diff_flag,
                diffAsPct_flag=diffAsPct_flag,
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
                makeFigure_flag=False,
                plev=plev,
                quiver_flag=quiver_flag,
                quiverKey_flag=(jSet == 0),
                quiverScale=quiverScale,
                quiverUnits=quiverUnits,
                rmse_flag=rmse_flag,
                save_flag=False,
                stampDate_flag=stampDate_flag,
                tSteps=tSteps,
                tStepLabel_flag=(jSet == 0),
                **kwargs
                )
        else:
            # print(plotId)
            if plotId == 'obs':
                (a, ax, c, m) = plotlatlon(
                    obsDs,  # hadIsstDs
                    obsVar,
                    box_flag=box_flag[jSet],
                    cbar_flag=False,
                    compcont_flag=compcont_flag,
                    diff_flag=False,
                    fontSize=fontSize,
                    latLim=latLim,  # np.array([-20, 20]),
                    latlbls=latlbls,
                    lonLim=lonLim,  # np.array([119.5, 270.5]),
                    lonlbls=lonlbls,
                    makeFigure_flag=False,
                    plev=plev,
                    quiver_flag=False,  # True,
                    # quiverDs=obsQuivDs,
                    # quiverLat=obsQuivDs['lat'],
                    # quiverLon=obsQuivDs['lon'],
                    # quiverNorm_flag=False,
                    # quiverScale=quiverProps['quiverScale'],
                    # quiverScaleVar=None,
                    save_flag=False,
                    stampDate_flag=stampDate_flag,
                    tSteps=tSteps,
                    tStepLabel_flag=(jSet == 0),
                    **kwargs
                    )
            else:
                im1, ax, compcont, _ = plotlatlon(
                    dsDict[plotId],
                    plotVar,
                    box_flag=box_flag[jSet],
                    cbar_flag=False,
                    compcont_flag=compcont_flag,
                    diff_flag=False,
                    diffDs=(
                        (dsDict[diffIdList[jSet]]
                         if any([diffDs is None,
                                diffDs == dsDict])
                         else diffDs)
                        if diff_flag else None),
                    fontSize=fontSize,
                    latLim=latLim,
                    latlbls=latlbls,
                    lonLim=lonLim,
                    lonlbls=lonlbls,
                    makeFigure_flag=False,
                    plev=plev,
                    quiver_flag=quiver_flag,
                    quiverKey_flag=(jSet == 0),
                    quiverScale=quiverScale,
                    quiverUnits=quiverUnits,
                    save_flag=False,
                    stampDate_flag=stampDate_flag,
                    tSteps=tSteps,
                    tStepLabel_flag=(jSet == 0),
                    **kwargs
                    )

        # Add subplot label (subfigure number)
        ax.annotate('(' + chr(jSet +
                              ord(subFigCountStart) -
                              skippedPlotCount
                              ) +
                    ')',
                    # xy=(-0.12, 1.09),
                    xy=(-0.08, 1.07),
                    xycoords='axes fraction',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    fontweight='bold',
                    )

        # Get id for a good plot for colorbar making
        if plotId != 'obs':
            goodPlotId = plotId

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

        # Get variable name for colorbar
        varName = mwp.getplotvarstring(dsDict[goodPlotId][plotVar].name)
        # Get variable name for colorbar label
        if np.ndim(dsDict[goodPlotId][plotVar]) == 4:
            # Add level if original field is 4d
            varName = varName + str(plev)
        if all([diff_flag, plev != diffPlev]):
            # Add differencing of levels if plotting differences and
            #   plev and diffPlev are not the same (for plotting shears)
            varName = varName + '-' + str(diffPlev)

        # Create colorbar
        if cbarOrientation == 'vertical':
            # Place colorbar on figure
            cbar_ax.set_position([pcb.x0 + cbar_xoffset,
                                  pcb.y0 + pcb.height/6.,
                                  0.015, pcb.height*2./3.])

            # Label colorbar with variable name and units
            cbar_ax.set_ylabel(
                (r'$\Delta$' if diff_flag else '') +
                varName + ' (' +
                mwfn.getstandardunitstring(
                    dsDict[goodPlotId][plotVar].units) +
                ')')

        elif cbarOrientation == 'horizontal':
            # Place colorbar on figure
            cbar_ax.set_position([pcb.x0, pcb.y0 - 0.015,
                                  pcb.width*1., 0.015])

            # Label colorbar with variable name and units
            cbar_ax.set_xlabel(
                (r'$\Delta$' if diff_flag else '') +
                varName + ' (' +
                mwfn.getstandardunitstring(
                    dsDict[goodPlotId][plotVar].units) +
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
                diffStr = 'd' + diffIdList[0] + '_'
            else:
                diffStr = ''
            # Get variable name for saving
            varName = plotVar
            if np.ndim(dsDict[goodPlotId][plotVar]) == 4:
                # Add level if original field is 4d
                varName = varName + str(plev)
            if plev != diffPlev:
                # Add differencing of levels if plotting differences and
                #   plev and diffPlev are not the same (for plotting shears)
                varName = varName + '-' + str(diffPlev)
            saveFile = ('d' + varName +
                        '_latlon_comp{:d}_'.format(len(plotIdList)) +
                        diffStr +
                        tString +
                        '{:03.0f}'.format(tSteps[0]) + '-' +
                        '{:03.0f}'.format(tSteps[-1]))
        else:
            if len(plotIdList) > 3:
                caseSaveString = 'comp{:d}'.format(len(plotIdList))
            else:
                caseSaveString = '_'.join(plotIdList)
            # Get variable name for saving
            varName = plotVar
            if np.ndim(dsDict[goodPlotId][plotVar]) == 4:
                # Add level if original field is 4d
                varName = varName + str(plev)
            saveFile = (
                varName + '_latlon_' +
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


def plotpressurelat(ds,
                    colorVar,
                    # caseString=None,
                    cbar_flag=True,
                    cbar_dy=-0.1,
                    cbar_height=0.02,
                    cMap=None,
                    colorConts=None,
                    dCont_flag=False,
                    dContCase=None,
                    diff_flag=False,
                    diffDs=None,
                    dt=1,
                    latLbls=None,
                    latLim=np.array([-30, 30]),
                    latSubSamp=3,
                    logP_flag=True,
                    lonLim=np.array([240, 270]),
                    lineCont_flag=False,
                    lineContDiff_flag=None,
                    lineConts=None,
                    lineContVar=None,
                    lineContDs=None,
                    lineContDiffDs=None,
                    makeFigure_flag=True,
                    pLbls=None,
                    pLim=np.array([1000, 200]),
                    quiver_flag=True,
                    quiverKey_flag=True,
                    quiverScale=3,
                    quiverUnits='inches',
                    save_flag=False,
                    saveDir=None,
                    saveSubDir=None,
                    tLim=np.array([0, 12]),
                    tLimLabel_flag=True,
                    wScale=100,
                    ):

    # Set datasets for plotting with black contours
    if lineCont_flag:
        if lineContDs is None:
            lineContDs = ds
        if lineContDiff_flag is None:
            lineContDiff_flag = diff_flag
        if lineContDiff_flag and (lineContDiffDs is None):
                lineContDiffDs = diffDs

    # Compute zonal mean over requested longitudes
    dsZm = ds.loc[
        dict(lon=slice(lonLim[0], lonLim[-1]),
             lat=slice(latLim[0]-2, latLim[-1]+2))
        ].mean(dim='lon')
    if diff_flag:
        diffDsZm = diffDs.loc[
            dict(lon=slice(lonLim[0], lonLim[-1]),
                 lat=slice(latLim[0]-2, latLim[-1]+2))
            ].mean(dim='lon')
    if lineCont_flag:
        lineDsZm = lineContDs.loc[
            dict(lon=slice(lonLim[0], lonLim[-1]),
                 lat=slice(latLim[0]-2, latLim[-1]+2))
            ].mean(dim='lon')
    if lineContDiff_flag:
        lineDiffDsZm = lineContDiffDs.loc[
            dict(lon=slice(lonLim[0], lonLim[-1]),
                 lat=slice(latLim[0]-2, latLim[-1]+2))
            ].mean(dim='lon')

    # Mean data over requested plotting time period
    dsZm = dsZm.isel(time=slice(tLim[0], tLim[-1], dt)).mean(dim='time')
    if diff_flag:
        diffDsZm = diffDsZm.isel(
            time=slice(tLim[0], tLim[-1], dt)).mean(dim='time')
    if lineCont_flag:
        lineDsZm = lineDsZm.isel(
            time=slice(tLim[0], tLim[-1], dt)).mean(dim='time')
    if lineContDiff_flag:
        lineDiffDsZm = lineDiffDsZm.isel(
            time=slice(tLim[0], tLim[-1], dt)).mean(dim='time')

    # Get contours for plotting filled contours
    if colorConts is None:
        try:
            if diff_flag:
                colorConts = {
                    'AREI': np.arange(-10, 10.1, 1),
                    'AREL': np.arange(-2, 2.1, 0.2),
                    'AWNC': np.arange(-1e7, 1.01e7, 1e6),
                    'AWNI': np.arange(-1e4, 1.01e4, 1e4),
                    'CLDICE': np.arange(-3e-6, 3.01e-6, 3e-7),
                    'CLDLIQ': np.arange(-4e-5, 4.01e-5, 4e-6),
                    'CLOUD': np.arange(-0.2, 0.201, 0.02),
                    'ICIMR': np.arange(-3e-6, 3.01e-6, 3e-7),
                    'ICWMR': np.arange(-1e-4, 1.01e-4, 1e-5),
                    'OMEGA': (np.arange(-0.03, 0.03001, 0.003)),
                    'RELHUM': np.arange(-20, 20.1, 2),
                    'T': np.arange(-2, 2.1, 0.2),
                    'U': np.arange(-5, 5.01, 0.5),
                    'V': np.arange(-1, 1.01, 0.1),
                    }[colorVar]
            else:
                colorConts = {
                    'CLOUD': np.arange(0, 0.301, 0.02),
                    'RELHUM': np.arange(0, 100.1, 5),
                    'T': np.arange(225, 295.1, 5),
                    'V': np.arange(-4, 4.1, 0.5),
                    'Z3': np.arange(0, 15001, 1000),
                    }[colorVar]
        except KeyError:
            colorConts = None

    # Get contours for plotting lined contours
    try:
        if lineContDiff_flag:
            lineConts = {
                'AREL': np.arange(-2, 2.1, 0.2),
                'CLOUD': np.arange(-0.2, 0.201, 0.02),
                'RELHUM': np.arange(-20, 20.1, 2),
                'T': np.arange(-2, 2.1, 0.2),
                'V': np.arange(-3, 3.1, 0.3),
                }[lineContVar]
        else:
            lineConts = {
                'CLOUD': np.arange(0, 0.301, 0.05),
                'RELHUM': np.arange(0, 100.1, 10),
                'T': np.arange(225, 295.1, 10),
                'V': np.arange(-4, 4.1, 0.5),
                'Z3': np.arange(0, 15001, 1500),
                }[lineContVar]
    except KeyError:
        lineConts = None

    # Create figure for plotting
    if makeFigure_flag:
        hf = plt.figure()
        hf.canvas.set_window_title(
            ds.id +
            ('-{:s}'.format(diffDs.id) if diff_flag
             else ''))

    # Plot meridional mean slice with filled contours
    if colorConts is None:
        cset1 = plt.contourf(dsZm['lat'],
                             (np.log10(dsZm['plev'])
                              if logP_flag
                              else dsZm['plev']),
                             dsZm[colorVar] -
                             (diffDsZm[colorVar] if diff_flag
                              else 0),
                             cmap=mwp.getcmap(colorVar,
                                              diff_flag=diff_flag),
                             extend='both')
    else:
        cset1 = plt.contourf(
            dsZm['lat'],
            (np.log10(dsZm['plev'])
             if logP_flag
             else dsZm['plev']),
            dsZm[colorVar] -
            (diffDsZm[colorVar] if diff_flag
             else 0),
            colorConts,
            cmap=mwp.getcmap(colorVar,
                             diff_flag=diff_flag),
            extend='both')

    # Plot meridional mean slice with black contours
    if lineCont_flag:
        cset2 = plt.contour(lineDsZm['lat'],
                            (np.log10(lineDsZm['plev'])
                             if logP_flag
                             else lineDsZm['plev']),
                            lineDsZm[lineContVar] -
                            (lineDiffDsZm[lineContVar] if lineContDiff_flag
                             else 0),
                            lineConts,
                            colors='k')
        plt.clabel(cset2)

    # Compute vector field if requested
    if quiver_flag:
        R = 287.058  # [J/kg/K]
        g = 9.80662  # [m/s^2]

        # Compute w
        dsW = (-dsZm['OMEGA']*R*dsZm['T'] /
               (dsZm['plev']*100*g))  # *100 converts hPa to Pa
        if diff_flag:
            diffDsW = (-diffDsZm['OMEGA']*R*diffDsZm['T'] /
                       (diffDsZm['plev']*100*g))  # *100 converts hPa to Pa

        # Plot quivers
        q1 = plt.quiver(dsZm['lat'][::latSubSamp],
                        (np.log10(dsZm['plev'])
                         if logP_flag
                         else dsZm['plev']),
                        dsZm['V'][:, ::latSubSamp] -
                        (diffDsZm['V'][:, ::latSubSamp] if diff_flag
                         else 0),
                        wScale*(dsW[:, ::latSubSamp] -
                                (diffDsW[:, ::latSubSamp] if diff_flag
                                 else 0)
                                ),
                        units=quiverUnits,
                        scale=quiverScale
                        )
        # Ad quiver key (reference vector)
        if quiverKey_flag:
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
    plt.xlim(latLim)

    # Set labels on x axis
    if latLbls is not None:
        plt.xticks(latLbls)

    # Set y axis to adjust for logP if requested
    if logP_flag:
        plt.ylim(np.log10(pLim))
        if pLbls is None:
            pLbls = np.arange(1000, 199, -100)
        plt.yticks(np.log10(pLbls),
                   pLbls)
    else:
        plt.ylim(pLim)

    # Label axes
    plt.xlabel('Latitude')
    plt.ylabel('Pressure ({:s})'.format(dsZm['plev'].units))

    # Add colorbar
    if cbar_flag:
        hcb = plt.colorbar(cset1,
                           label='{:s} ({:s})'.format(
                               mwp.getplotvarstring(colorVar),
                               ds[colorVar].units))
        if colorVar == 'OMEGA':
            plt.annotate('(up)',
                         xy=(0.85, 0.1),
                         xycoords='figure fraction',
                         horizontalalignment='right',
                         verticalalignment='bottom')

    # Add case number
    ax.annotate(ds.id +
                ('-{:s}'.format(diffDs.id) if diff_flag
                 else ''),
                xy=[0, 1],
                xycoords='axes fraction',
                horizontalalignment='left',
                verticalalignment='bottom')

    # Add time range
    if tLimLabel_flag:
        tStepString = 't = [{:0d}, {:0d}]'.format(tLim[0], tLim[-1]-1)
        ax.annotate(tStepString,
                    xy=[1, 1],
                    xycoords='axes fraction',
                    horizontalalignment='right',
                    verticalalignment='bottom')

    if save_flag:
        # Set directory for saving
        if saveDir is None:
            saveDir = os.path.dirname(os.path.realpath(__file__))

        # Set filename for saving
        saveFile = (('d' if diff_flag else '') +
                    colorVar +
                    ('_VW' if quiver_flag else '') +
                    '_' + ds.id +
                    ('-{:s}'.format(diffDs.id) if diff_flag else '') +
                    '_' + mwp.getlatlimstring(latLim, '') +
                    '_' + mwp.getlonlimstring(lonLim, '') +
                    '_mon{:02d}-{:02d}'.format(tLim[0], tLim[-1]-1)
                    )

        # Set saved figure size (inches)
        fx = hf.get_size_inches()[0]
        fy = hf.get_size_inches()[1]

        # Save figure
        if any([saveSubDir is None,
                saveSubDir == '']):
            tempSub = saveSubDir
            saveSubDir = 'atm/meridslices/'
        print(saveDir + saveSubDir + saveFile)
        print('gets here')
        mwp.savefig(saveDir + saveSubDir + saveFile,
                    shape=np.array([fx, fy]))
        saveSubDir = tempSub
        plt.close('all')

    return ax, cset1, q1


def plotmultipressurelat(dsDict,
                         plotIdList,
                         colorVar,
                         cbar_flag=True,
                         cbarOrientation='vertical',
                         diff_flag=False,
                         diffIdList=None,
                         dt=1,
                         figSize=None,
                         fontSize=12,
                         latLim=np.array([-30, 30]),
                         lonLim=np.array([240, 270]),
                         obsDs=None,
                         plev=None,
                         quiver_flag=False,
                         quiverScale=0.4,
                         quiverUnits='inches',
                         rmse_flag=False,
                         # rmRegMean_flag=False,
                         save_flag=False,
                         saveDir=None,
                         stampDate_flag=False,
                         saveSubDir=None,
                         saveThenClose_flag=True,
                         subFigCountStart='a',
                         tLim=None,
                         verbose_flag=False,
                         wScale=100,
                         **kwargs
                         ):
    """
    Plot meridional cross-sections from multiple cases for comparison

    Version Date:
        2018-06-14
    """

    # Ensure diffIdList or diffDs provided if diff_flag
    if all([diff_flag, (diffIdList is None)]):
        raise ValueError('diffIdList must be provided to plot ' +
                         'differences')

    # Determine time step parameters
    if tLim is None:
        tLim = np.array([0, dsDict[plotIdList[0]][colorVar].shape[0]])

    # Create figure for plotting
    hf = plt.figure()

    # Set figure window title
    hf.canvas.set_window_title(('d' if diff_flag else '') +
                               'complatP: ' + colorVar
                               )

    # Set gridspec values for subplots
    if len(plotIdList) == 2:
        if cbarOrientation == 'vertical':
            # Set figure window size
            hf.set_size_inches(9, 2, forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(1, 3,
                                   # height_ratios=[20, 1, 20, 1, 20],
                                   # hspace=0.3,
                                   width_ratios=[30, 30, 1],
                                   )
            gs.update(left=0.07, right=0.95, top=0.95, bottom=0.05)

            # Set gridspec colorbar location
            cbColInd = 2
            cbRowInd = 0
            cbar_xoffset = -0.04

        elif cbarOrientation == 'horizontal':
            # Set figure window size
            hf.set_size_inches(7.5, 4, forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(2, 2,
                                   height_ratios=[20, 1],
                                   )

            # Set gridspec colorbar location
            cbColInd = 0
            cbRowInd = 1

        # Set gridpsec index order
        colInds = [0, 1]
        rowInds = [0, 0]
    if len(plotIdList) == 3:
        if cbarOrientation == 'vertical':
            # Set figure window size
            hf.set_size_inches(9, 8, forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(3, 2,
                                   # height_ratios=[20, 1, 20, 1, 20],
                                   hspace=0.3,
                                   width_ratios=[30, 1],
                                   )
            gs.update(left=0.07, right=0.95, top=0.95, bottom=0.05)

            # Set gridspec colorbar location
            cbColInd = 1
            cbRowInd = 0
            cbar_xoffset = -0.04

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
        rowInds = [0, 1, 2]

    elif len(plotIdList) == 4:
        if cbarOrientation == 'vertical':
            # Set figure window size
            if figSize is None:
                hf.set_size_inches(9, 3, forward=True)
            else:
                hf.set_size_inches(figSize[0], figSize[1],
                                   forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(2, 3,
                                   # height_ratios=[20, 1, 20, 1, 20],
                                   hspace=0.1,
                                   width_ratios=[30, 30, 1],
                                   )
            gs.update(left=0.05, right=0.92, top=0.95, bottom=0.05)

            # Set gridpsec index order
            colInds = [0, 1, 0, 1]
            rowInds = [0, 0, 1, 1]
            cbar_xoffset = -0.04

            # Set gridspec colorbar location
            cbColInd = 2
            cbRowInd = 0

    elif len(plotIdList) == 6:
        if cbarOrientation == 'vertical':
            # Set figure window size
            hf.set_size_inches(13, 6,
                               forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(2, 4,
                                   # height_ratios=[20, 1, 20, 1, 20],
                                   # hspace=0.3,
                                   width_ratios=[30, 30, 30, 1],
                                   wspace=0.35,
                                   )
            gs.update(left=0.1, right=0.92, top=0.95, bottom=0.1)

            # Set gridpsec index order
            colInds = [0, 1, 2, 0, 1, 2]
            rowInds = [0, 0, 0, 1, 1, 1]

            # Set gridspec colorbar location
            cbColInd = 3
            cbRowInd = 0
            cbar_xoffset = -0.03
        else:
            raise NotImplementedError('Horizontal. colorbar not supported ' +
                                      'for 6 panel plot')

    elif len(plotIdList) == 9:
        if cbarOrientation == 'vertical':
            # Set figure window size
            hf.set_size_inches(13, 9, forward=True)

            # Set up subplots
            gs = gridspec.GridSpec(3, 4,
                                   height_ratios=[1, 1, 1],
                                   hspace=0.2,
                                   width_ratios=[30, 30, 30, 1],
                                   wspace=0.33,
                                   left=0.04,
                                   right=0.96,
                                   bottom=0.00,
                                   top=1.0,
                                   )

            # Set gridspec colorbar location
            cbColInd = 3
            cbRowInd = 0
            cbar_xoffset = -0.03

            # Set gridspec index order
            colInds = [0, 1, 2, 0, 1, 2, 0, 1, 2]
            rowInds = [0, 0, 0, 1, 1, 1, 2, 2, 2]
            
        elif cbarOrientation == 'horizontal':
            raise NotImplementedError('Horizontal. colorbar not supported ' +
                                      'for 6 panel plot')

    # Plot maps
    skippedPlotCount = 0
    for jSet, plotId in enumerate(plotIdList):

        if plotId is None:
            skippedPlotCount = skippedPlotCount + 1
            print('skipping {:d}'.format(jSet))
            continue

        plt.subplot(gs[rowInds[jSet], colInds[jSet]])

        if diff_flag:
            diffId = diffIdList[jSet]
            if verbose_flag:
                print(plotId + ' - ' + diffId)
        else:
            if verbose_flag:
                print(plotId)
        if plotId != 'obs':
            ax, im1, q1 = plotpressurelat(
                dsDict[plotId],
                colorVar,
                cbar_flag=False,
                # colorConts=None,
                # dCont_flag=False,
                # dContCase=None,
                diff_flag=diff_flag,
                diffDs=(dsDict[diffId] if diff_flag else None),
                dt=dt,
                latLim=latLim,
                lonLim=lonLim,
                # lineCont_flag=lineCont_flag,
                # lineContDiff_flag=lineContDiff_flag,
                # lineConts=None,
                # lineContVar=colorVar,
                # lineContDs=dataSets_rg[(plotCase
                #                        if lineContDiff_flag
                #                        else diffCase)],
                # lineContDiffDs=dataSets_rg[diffCase],
                makeFigure_flag=False,
                # pLim=pLim,
                quiver_flag=quiver_flag,
                quiverKey_flag=0,
                # quiverScale=3,
                # quiverUnits='inches',
                save_flag=False,
                tLim=tLim,
                tLimLabel_flag=(jSet == 0),
                wScale=wScale,
                **kwargs
                )
        else:
            raise NotImplementedError('Cannot plot obs yet')

        if colInds[jSet] != 0:
            plt.ylabel('')
        if rowInds[jSet] != np.max(rowInds):
            plt.xlabel('')

        # Add subplot label (subfigure number)
        ax.annotate('(' + chr(jSet +
                              ord(subFigCountStart) -
                              skippedPlotCount
                              ) +
                    ')',
                    # xy=(-0.12, 1.09),
                    xy=(-0.2, 1.03),
                    xycoords='axes fraction',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    fontweight='bold',
                    )

        # Get id for a good plot for colorbar making
        if plotId != 'obs':
            goodPlotId = plotId

    # Add common colorbar

    # Create axis for colorbar
    cbar_ax = plt.subplot(gs[cbRowInd:, cbColInd:])

    # Create colorbar and set position
    hcb = plt.colorbar(im1,
                       cax=cbar_ax,
                       # format='%0.0f',
                       orientation=cbarOrientation,
                       )
    pcb = cbar_ax.get_position()

    # Get variable name for colorbar
    varName = mwp.getplotvarstring(dsDict[goodPlotId][colorVar].name)

    # Create colorbar
    if cbarOrientation == 'vertical':
        # Place colorbar on figure
        cbar_ax.set_position([pcb.x0 + cbar_xoffset,
                              pcb.y0 + pcb.height/6.,
                              0.015, pcb.height*2./3.])

        # Label colorbar with variable name and units
        cbar_ax.set_ylabel(
            (r'$\Delta$' if diff_flag else '') +
            varName + ' (' +
            mwfn.getstandardunitstring(
                dsDict[goodPlotId][colorVar].units) +
            ')')

    elif cbarOrientation == 'horizontal':
        # Place colorbar on figure
        cbar_ax.set_position([pcb.x0, pcb.y0 - 0.015,
                              pcb.width*1., 0.015])

        # Label colorbar with variable name and units
        cbar_ax.set_xlabel(
            (r'$\Delta$' if diff_flag else '') +
            varName + ' (' +
            mwfn.getstandardunitstring(
                dsDict[goodPlotId][colorVar].units) +
            ')')

    # Add colorbar ticks and ensure 0 is labeled if differencing
    if diff_flag:
        try:
            hcb.set_ticks(im1.levels[::2]
                          if np.min(np.abs(im1.levels[::2] - 0)
                                    ) < 1e-10
                          else im1.levels[1::2])
        except TypeError:
            pass

    # Prevent colorbar from using offset (i.e. sci notation)
    # hcb.ax.yaxis.get_major_formatter().set_useOffset(False)

    # Add quiver key
    if quiver_flag:
        plt.quiverkey(q1,
                      pcb.x0 + 0.01,
                      pcb.y0 + pcb.height/6 - 0.05,
                      1,
                      '[v ({:d} {:s}), \n'.format(
                          1,
                          mwfn.getstandardunitstring('m/s')) +
                      'w ({:0.0e} {:s})]'.format(
                          1/wScale,
                          mwfn.getstandardunitstring('m/s')),
                      coordinates='figure',
                      labelpos='S')

    # Expand plot(s) to fill figure window
    # gs.tight_layout(hf)

    # Save figure if requested
    if save_flag:
        # Set directory for saving
        if saveDir is None:
            saveDir = os.path.dirname(os.path.realpath(__file__))

        # Set string of latitude limits
        lonLimString = ''.join(['{:02.0f}'.format(np.abs(lonLim[x])) +
                                ('W' if (lonLim[x] < 0) else
                                 ('E' if (lonLim[x] > 0) else ''))
                                for x in range(lonLim.size)])

        # Set file name for saving
        tString = 'mon'
        if diff_flag:
            if all([diffIdList[j] == diffIdList[0]
                    for j in range(len(diffIdList))]):
                diffStr = 'd' + diffIdList[0] + '_'
            else:
                diffStr = ''
            # Get variable name for saving
            saveFile = ('d{:s}_'.format(colorVar) +
                        ('VW_' if quiver_flag else '') +
                        'latP_comp{:d}_'.format(len(plotIdList)) +
                        diffStr +
                        '{:s}_'.format(lonLimString) +
                        tString +
                        '{:02.0f}'.format(tLim[0]) + '-' +
                        '{:02.0f}'.format(tLim[-1]-1)
                        )
        else:
            if len(plotIdList) > 3:
                caseSaveString = 'comp{:d}'.format(len(plotIdList))
            else:
                caseSaveString = '_'.join(plotIdList)
            saveFile = (
                '{:s}_'.format(colorVar) +
                ('VW_' if quiver_flag else '') +
                'latP_' +
                caseSaveString + '_' +
                '{:s}_'.format(lonLimString) +
                tString +
                '{:02.0f}'.format(tLim[0]) + '-' +
                '{:02.0f}'.format(tLim[-1]-1) +
                ('_nocb' if not cbar_flag else '')
                )

        # Set saved figure size (inches)
        fx = hf.get_size_inches()[0]
        fy = hf.get_size_inches()[1]

        # Save figure
        print(saveDir + saveSubDir + saveFile)
        mwp.savefig(saveDir + saveSubDir + saveFile,
                    shape=np.array([fx, fy]))
        if saveThenClose_flag:
            plt.close('all')


def regriddatasets(dataSets,
                   fileBaseDict=None,
                   mp_flag=False,
                   ncDir=None,
                   ncSubDir=None,
                   newLevs=None,
                   regrid2file_flag=False,
                   regridOverwrite_flag=False,
                   regridVars=None,
                   versionIds=None,
                   ):
    """
    Vertically regrid a dictionary of datasets of CESM output
    """

    # Set new levels for regridding
    if newLevs is None:
        newLevs = np.array([100, 200, 275, 350, 425,
                            500, 550, 600, 650, 700,
                            750, 800, 850, 900, 950,
                            975, 1000])

    # Set variables to regrid if not provided
    if regridVars is None:
        regridVars = ['V', 'OMEGA', 'RELHUM', 'CLOUD', 'T', 'U',
                      ]

    # Set cases to regrid
    if versionIds is None:
        versionIds = list(dataSets.keys())

    # Set location to look for previously regridded files
    if ncDir is None:
        ncDir, ncSubDir, _ = setfilepaths()

    # Get full case name for each version id if not provided
    if fileBaseDict is None:
        fileBaseDict = getcasebase()

    # Set flag to tell if need to do regridding for each dataset
    regridIds = []

    # First attempt to load each case from file
    #   add cases to list to be regridded as they fail certain checks
    dataSets_rg = dict()
    threeDdir = dict()
    threeDfile = dict()
    for vid in versionIds:
        # Set directory for saving netcdf file of regridded output
        if gethostname() in getuwmachlist():
            threeDdir[vid] = (ncDir + fileBaseDict[vid] + '/' +
                              ('atm/hist/'
                               if 'f' in vid
                               else ncSubDir) +
                              '3dregrid/')
        elif gethostname()[0:6] in getncarmachlist():
            threeDdir[vid] = (ncDir + fileBaseDict[vid] + '/' +
                              ncSubDir +
                              '3dregrid/')

        # Set filename for saving netcdf file of regridded output
        threeDfile[vid] = (fileBaseDict[vid] +
                           '.plevs.nc')

        # Attempt to load previously regridded case from file
        try:
            dataSets_rg[vid] = xr.open_dataset(threeDdir[vid] + threeDfile[vid])
        except OSError:
            regridIds.append(vid)
            print('Previously regridded file unavaialble. ' +
                  'Will regrid {:s}'.format(vid))
            continue

        # Ensure all requested variables are present
        if not all([x in dataSets_rg[vid].data_vars
                    for x in regridVars]):
            regridIds.append(vid)
            print('Requested variables not all present. ' +
                  'Will regrid {:s}'.format(vid))
            continue

        # Check if all requested levels are present
        if not all([x in dataSets_rg[vid]['plev'].values
                    for x in newLevs]):
            regridIds.append(vid)
            print('Not all levels present. ' +
                  'Will regrid {:s}'.format(vid))
            continue
        
        # Return success of loading from file
        print('Succesfully loaded 3D fields for {:s}'.format(vid))

    # Perform regridding if cannot load appropriate regridded cases from
    #   previously regridded files
    if regridIds:
        # Start timing clock
        regridStartTime = datetime.datetime.now()
        print(regridStartTime.strftime('--> Regrid start time: %X'))

        # Regrid using multiprocessing
        #   Seems buggier of late. Use at own risk.
        if mp_flag:
            # Regrid 3D variables using multiprocessing
            #   Parallelizing over cases(?)
            #   Need to be wary here to not run out of memory.
            mpPool = mp.Pool(1)

            # Load all datasets to memory to enable multiprocessing
            for vid in regridIds:
                print(vid)
                for regridVar in regridVars:
                    dataSets[vid][regridVar].load()
                dataSets[vid]['PS'].load()
                dataSets[vid]['hyam'].load()
                dataSets[vid]['hybm'].load()
                dataSets[vid]['P0'].load()

            # Create input tuple for regridding to pressure levels
            mpInList = [(dataSets[vid],
                         regridVars,
                         newLevs,
                         {'hCoeffs': {
                             'hyam': dataSets[vid]['hyam'].mean(
                                 dim='time').values,
                             'hybm': dataSets[vid]['hybm'].mean(
                                 dim='time').values,
                             'P0': dataSets[vid]['P0'].values[0]},
                          'modelid': 'cesm',
                          'psVar': 'PS',
                          'verbose_flag': False}
                         )
                        for vid in regridIds]

            # Call multiprocessing of regridding
            # regriddedVars = mpPool.map(mwfn.regriddssigmatopres_mp,
            #                            mpInList)
            # regriddedVars = mpPool.map_async(mwfn.regriddssigmatopres_mp,
            #                                 mpInList)
            dsOut = mpPool.map_async(mwfn.convertsigmatopresds_mp,
                                     mpInList)
            # dsOut = mpPool.map(mwfn.convertsigmatopresds_mp,
            #                   mpInList)

            # Close multiprocessing pool
            dsOut = dsOut.get()
            mpPool.close()
            mpPool.terminate()  # Not proper,
            #                   #    but may be needed to work properly
            mpPool.join()

            # Convert dsOut from list of datasets to dictionary of datasets
            dataSets_rg = {dsOut[j].id: dsOut[j]
                           for j in range(len(dsOut))}
        else:
            # Regrid without multiprocessing
            #   Some cases error out with mp for unknown reasons
            for vid in regridIds:
                print('Regridding {:s}...'.format(vid))
                dataSets_rg[vid] = mwfn.convertsigmatopresds(
                    dataSets[vid],
                    regridVars,
                    newLevs,
                    hCoeffs={'hyam': dataSets[vid]['hyam'].mean(
                                 dim='time').values,
                             'hybm': dataSets[vid]['hybm'].mean(
                                 dim='time').values,
                             'P0': dataSets[vid]['P0'].values[0]
                             },
                    modelid='cesm',
                    psVar='PS',
                    verbose_flag=False,
                    )

        # Write time elapsed at end of regridding
        print('\n##------------------------------##')
        print('Time to regrid with mp:')
        print(datetime.datetime.now() - regridStartTime)
        print('##------------------------------##\n')

        # Write regridded datasets to file for quick future reloading.
        if regrid2file_flag:
            for vid in regridIds:
                if gethostname() in getuwmachlist():
                    # Set directory for saving netcdf file of regridded output
                    threeDdir = (ncDir + fileBaseDict[vid] + '/' +
                                 ('atm/hist/'
                                  if 'f' in vid
                                  else ncSubDir) +
                                 '3dregrid/')
                elif gethostname()[0:6] in getncarmachlist(6):
                    # Set directory for saving netcdf file of regridded output
                    threeDdir = (ncDir + fileBaseDict[vid] + '/' +
                                 ncSubDir +
                                 '3dregrid/')
                # Set filename for saving netcdf file of regridded output
                threeDfile = (fileBaseDict[vid] +
                              '.plevs.nc')

                # Create directory if needed
                if not os.path.exists(threeDdir):
                    os.makedirs(threeDdir)

                # Save netcdf file if possible
                try:
                    if os.path.exists(threeDdir + threeDfile):
                        try:
                            print('Writing {:s}'.format(
                                  threeDdir + threeDfile))
                            dataSets_rg[vid].to_netcdf(
                                path=threeDdir + threeDfile,
                                mode='a')
                        except OSError as ose:
                            if regridOverwrite_flag:
                                print('Overwriting existing file at:\n' +
                                      threeDdir + threeDfile)
                                os.remove(threeDdir + threeDfile)
                                dataSets_rg[vid].to_netcdf(
                                    path=threeDdir + threeDfile,
                                    mode='w')
                            else:
                                raise OSError(
                                    'File already exists:\n' +
                                    threeDdir + threeDfile +
                                    ' Thus cannot write for {:s}'.format(vid))
                        except RuntimeError:
                            continue
                    else:
                        print('Writing {:s}'.format(
                              threeDdir + threeDfile))
                        dataSets_rg[vid].to_netcdf(
                            path=threeDdir + threeDfile,
                            mode='w')
                except ValueError:
                    raise ValueError('probably related to datetime.')

    return dataSets_rg


def setfilepaths(loadClimo_flag=True,
                 newRuns_flag=False):
    """
    Set host specific variables and filepaths

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2018-06-22

    Args:
        N/A

    Kwargs:
        loadClimo_flag - True to load climatology rather than full output
            (NCAR systems only)
        newRuns_flag - True to change directorys for new runs
            (on yslogin only!)

    Returns:
        ncDir - directory in which netcdf case directories are stored
        ncSubDir - directory within case directory to search for netcdfs
        saveDir - directory to which figures will be saved

    Notes:
        fullPathForHistoryFileDirectory = (ncDir + fullCaseName +
                                           os.sep + ncSubDir)
    """

    if gethostname() in getuwmachlist():
        ncDir = '/home/disk/eos9/woelfle/cesm/nobackup/cesm1to2/'
        ncSubDir = '0.9x1.25/'
        saveDir = ('/home/disk/user_www/woelfle/cesm1to2/')

    elif gethostname() == 'woelfle-laptop':
        ncDir = 'C:\\Users\\woelfle\\Documents\\UW\\CESM\\hist\\'
        ncSubDir = ''
        saveDir = 'C:\\Users\\woelfle\\Documents\\UW\\CESM\\figs\\'

    elif gethostname()[0:6] in getncarmachlist(6):
        if loadClimo_flag:
            ncDir = '/glade/work/woelfle/cesm1to2/climos/'
            ncSubDir = ''
        elif newRuns_flag:
            ncDir = '/glade/scratch/woelfle/archive/'
            ncSubDir = 'atm/hist/'
        else:
            ncDir = '/glade/p/cgd/amp/people/hannay/amwg/climo/'
            ncSubDir = '0.9x1.25/'
        saveDir = '/glade/work/woelfle/figs/cesm1to2/'

    return (ncDir, ncSubDir, saveDir)
