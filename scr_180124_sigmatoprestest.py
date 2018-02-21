#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:37:01 2018

@author: woelfle
"""

from mdwtools import mdwfunctions as mwfn  # For averaging things
import datetime  # For keeping track of run times
import numpy as np

newlevs = np.array([500])  # , 850])
hCoeffs={'hyam': dataSets['125']['hyam'].mean(dim='time').values,
         'hybm': dataSets['125']['hybm'].mean(dim='time').values,
         'P0': dataSets['125']['P0'].values[0]}
psVar = 'PS'
regridVars = ['U']  # , 'V', 'T']

# Test speeds
# Version 1:
a = dict()
startTime = datetime.datetime.now()
print(startTime)
for regridVar in regridVars:
    a[regridVar] = mwfn.convertsigmatopres(
        dataSets['125'][regridVar].values,
        dataSets['125'][psVar].values,
        newlevs,
        hCoeffs=hCoeffs,
        modelid='cesm',
        verbose_flag=False)
print('\n##------------------------------##')
print('Time to regrid with old setup:')
print(datetime.datetime.now() - startTime)
print('##------------------------------##\n')

startTime = datetime.datetime.now()
print(startTime)
b = mwfn.convertsigmatopresds(dataSets['125'],
                              regridVars,
                              newlevs,
                              hCoeffs=hCoeffs,
                              modelid='cesm',
                              psVar=psVar,
                              verbose_flag=False)
print('\n##------------------------------##')
print('Time to regrid with newsetup:')
print(datetime.datetime.now() - startTime)
print('##------------------------------##\n')
