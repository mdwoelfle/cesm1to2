#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:31:48 2017

@author: woelfle
"""

# %% Import modules as needed

import datetime
from mdwtools import mdwfunctions as mwfn
import multiprocessing as mp
import numpy as np
from socket import gethostname   # used to determine which machine we are
#                                #   running on
import xarray as xr  # for handling nd things (netcdfs)

# %% Define functions as needed


def getfilebase(vid):
    """
    Return file base as a function of version ID (vid)
    """
    return {'01': 'b.e15.B1850G.f09_g16.pi_control.01',
            '28': 'b.e15.B1850G.f09_g16.pi_control.28',
            '36': 'b.e15.B1850.f09_g16.pi_control.36',
            'ga7.66': 'b.e15.B1850.f09_g16.pi_control.all_ga7.66',
            '119': 'b.e15.B1850.f09_g16.pi_control.all.119',
            '125': 'b.e20.B1850.f09_g16.pi_control.all.125',
            '161': 'b.e20.BHIST.f09_g17.20thC.161_01',
            '194': 'b.e20.B1850.f09_g17.pi_control.all.194',
            '195': 'b.e20.B1850.f09_g17.pi_control.all.195'
            }[vid]


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


# %% Main section for running code
if __name__ == '__main__':

    # Set flags
    mp_flag = True

    # Set file location info
    ncDir, ncSubDir, saveDir = setfilepaths()

    # Set values for constructing file names
    for versionId in ['01',
                      '28',  '36',
                      'ga7.66', '119', '125',
                      '161', '194', '195']:
        fileBase = getfilebase(versionId)
        loadSuffixes = ['_' + '{:02d}'.format(mon + 1) + '_climo.nc'
                        for mon in range(12)]
        loadFileList = [ncDir + fileBase + '/' +
                        ncSubDir +
                        fileBase + loadSuffix
                        for loadSuffix in loadSuffixes]

        # Load output from hybrid sigma coordinates
        ds = xr.open_mfdataset(loadFileList)

        # Add id to ds
        ds.attrs['id'] = versionId

        # Regrid output to pressure levels
        # Regridding with multiprocessing
        if mp_flag:

            print('>> Regridding vertical levels <<')
            startTime = datetime.datetime.now()
            # Set new levels for regridding
            #   > Timing works out to about 30s per level (seems long...)
            newLevs = np.array([10, 125, 150, 175, 200, 225, 250,
                                300, 350, 400, 450, 500, 550, 600,
                                650, 700, 750, 775, 800, 825, 850,
                                875, 900, 925, 950, 975])
            regridVars = ['U', 'V']

            # Regrid 3D variables using multiprocessing
            mpPool = mp.Pool(8)

            # Load all variables to be regridded to memory
            #   to enable multiprocessing
            for regridVar in regridVars:
                ds[regridVar].load()
            ds['PS'].load()
            ds['hyam'].load()
            ds['hybm'].load()

            # Create input tuple for regridding to pressure levels
            mpInList = [(ds,
                         regridVar,
                         newLevs,
                         'cesm')
                        for regridVar in regridVars]

            # Call multiprocessing of regridding
            # regriddedVars = mpPool.map(mwfn.regriddssigmatopres_mp,
            #                            mpInList)
            regriddedVars = mpPool.map_async(mwfn.regriddssigmatopres_mp,
                                             mpInList)

            # Close multiprocessing pool
            regriddedVars = regriddedVars.get()
            mpPool.close()
            mpPool.terminate()  # Not proper, but may be needed to work properly
            mpPool.join()

            print('\n##------------------------------##')
            print('Time to regrid with mp:')
            print(datetime.datetime.now() - startTime)
            print('##------------------------------##\n')

            # Unpack regriddedVars
            # dataSets_rg = dict()
            # for jVar, regridVar in enumerate(regridVars):
            #     dataSets_rg[regridVar] = (
            #         regriddedVars[jVar].to_dataset(name=regridVar))
            #     dataSets_rg[regridVar].attrs['id'] = regriddedVars[jVar].id
            ds_rg = xr.merge(regriddedVars)
            # ds_rg.attrs['id'] = versionId

        else:
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
    #        newDas = dict()
    #        for vid in versionIds:
    #            print('---Processing {:s}---'.format(vid))
    #            newDas[vid] = mwfn.regriddssigmatopres(dataSets[vid],
    #                                                   regridVar,
    #                                                   newLevs,
    #                                                   modelId='cesm',
    #                                                   verbose_flag=True)

            print('\n##------------------------------##')
            print('Time to regrid with loop:')
            print(datetime.datetime.now() - startTime)
            print('##------------------------------##\n')

        # Kludge to make ncview happy
        ds_rg.coords['plev'].values = ds_rg['plev'].values.astype(float)

        # Kludge to let xarray write out netcdf to file
        ds_rg.coords['time'].values = np.arange(12.)
    #
    #    # Write out new file with pressure coordinates
        newSaveDir = (ncDir + getfilebase(versionId) + '/' + ncSubDir)
        newFileName = (getfilebase(versionId) + '_monclimo_3d.nc')
        ds_rg.to_netcdf(newSaveDir + newFileName,
                        mode='w',
                        )

# %% Test by reloading
#    vid = '01'
#    ds_reload = xr.open_dataset(ncDir + getfilebase(vid) + '/' +
#                                ncSubDir + getfilebase(vid) +
#                                '_monclimo_3d.nc')
