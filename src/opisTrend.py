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
                          'stop'  : datetime(2019,4,1,2,51,50).timestamp(),
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
integ =  []
fwhm = []

maxIdx = tof.idxmin(axis=1)
maxima = evConv[CH](maxIdx)
integ  = tof.sum(axis=1)


for n, trace in tof.iterrows():
    if n % 5 == 0:
        plt.plot( evConv[CH](tof.iloc[0].index) , trace + n*5 )

def fit(x, m, q):
    return x*m + q

maxima = np.array(maxima)
maxima += 15.8 # Photon energy = electron energy + bounding energy
x = np.array(range(len(maxima)))
popt, pconv = curve_fit(fit, x, maxima)

plt.figure()
plt.plot(x, maxima, 'o')
plt.plot(x, fit(x, *popt))
print(popt)
plt.figure()
plt.plot(x, integ)
plt.show()
