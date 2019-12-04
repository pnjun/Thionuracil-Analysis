#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict
from scipy.optimize import curve_fit

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

CH = 1
tof = h5data.select('tof1',
                     where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]',
                            'pulseId in pulses.index'] )
tof = tof.mean(level=1)
evConv = ou.evConv()

maxIdx = tof.idxmin(axis=1)
maxVal = tof.min(axis=1)
maxEv = evConv[CH](maxIdx)
integ  = tof.sum(axis=1)

def getFWHM(row):
    a =  np.where(np.diff(np.signbit(row)))[0]
    return [a[0], a[1]]


#FWHM calulation
boundsIdx = tof.sub( maxVal/2, axis=0 ).apply(getFWHM, axis=1, raw=True)
boundsTof = [ tof.columns[bounds] for bounds in boundsIdx ]
boundsEv  = [ evConv[CH](bTof) for bTof in boundsTof ]
fwhm = [ bEv[0] - bEv[1] for bEv in boundsEv ]

for n, trace in tof.iterrows():
    if n % 5 == 0:
        plt.plot( evConv[CH](tof.iloc[0].index) , trace + n*5 )

def fit(x, m, q):
    return x*m + q

maxEv = np.array(maxEv)
maxEv += 15.8 # Photon energy = electron energy + bounding energy
x = np.array(range(len(maxEv)))
popt, pconv = curve_fit(fit, x, maxEv)

plt.figure('energy')
plt.plot(x, maxEv, 'o')
plt.plot(x, fit(x, *popt))
print(popt)
plt.figure('fwhm')
plt.plot(x, fwhm)
plt.figure('scatter')
plt.plot(maxEv, fwhm, 'o')
#plt.figure()
#plt.plot(x, integ)
plt.show()
