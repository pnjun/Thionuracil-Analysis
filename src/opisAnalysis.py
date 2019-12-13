#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict
from scipy.optimize import curve_fit

import sys
import opisUtils as ou

import pickle

cfg = {    'data'     : { 'path'     : '/media/Fast1/ThioUr/processed/',
                          'filename' : 'opistest.h5'
                        },
           'time'     : { #'start' : datetime(2019,3,31,6,54,0).timestamp(),
                          #'stop'  : datetime(2019,3,31,6,56,0).timestamp(),
                          'start' : datetime(2019,4,1,2,51,0).timestamp(),
                          'stop'  : datetime(2019,4,1,2,55,0).timestamp(),
                        },
      }

cfg = AttrDict(cfg)


h5data  = pd.HDFStore(cfg.data.path + cfg.data.filename, mode = 'r')

pulses = h5data.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')
pulsesLims = (pulses.index[0], pulses.index[-1])

tofs =   [ h5data.select(f'tof{n}',
               where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]',
                      'pulseId in pulses.index'] )
           for n in range(4) ]


traces   = [ tofs[n] for n in range(4) ]
evConv = ou.evConv()

GUESSEV = 222
TRACEID = 35

amplR = np.linspace(-60, -5, 16)
enerR = np.linspace(GUESSEV-1.5, GUESSEV+1.5, 64)

fitter = ou.evFitter(ou.GAS_ID_AR)
fitter.loadTraces(traces, evConv, GUESSEV)
diffs = fitter.getOffsets(getdiffs=True)
fit = fitter.leastSquare(amplR, enerR)

#invalidate low signal shots
mask, rawm = fitter.ignoreMask(getRaw=True)
fitm = fit.copy()
fitm[mask] = np.NaN

fit = fit.reshape((fit.shape[0]//49,49,2))
fitm = fitm.reshape((fitm.shape[0]//49,49,2))
rawm = rawm.reshape((rawm.shape[0]//49,49,4))

plt.figure('WL histogram', figsize=(7, 5))
for i in range(6):
    idx = i*9
    plt.subplot(2,3,i+1, title = f'shot {idx}')
    plt.hist(fit[idx,:,1])

plt.figure('AMPLI histogram', figsize=(7, 5))
for i in range(6):
    idx = i*9
    plt.subplot(2,3,i+1, title = f'shot {idx}')
    plt.hist(-fit[idx,:,0])

plt.figure('INTEG histogram', figsize=(10, 7))
for i in range(6):
    idx = i*9
    plt.subplot(2,3,i+1, title = f'shot {idx}')
    plt.hist(rawm[idx])


fit = np.nanmean( fit, axis=0 )
fitm = np.nanmean( fitm , axis=0 )

plt.figure('mask')
plt.plot(np.mean( mask.reshape((mask.shape[0]//49,49)), axis=0))
plt.figure('wavelenght')
plt.plot(fit[:,1])
plt.plot(fitm[:,1])
plt.figure('AMPLI')
plt.plot(-fit[:,0])
plt.plot(-fitm[:,0])

'''
fitter.plotFitted(TRACEID,show=False)
'''

plt.show()
h5data.close()
