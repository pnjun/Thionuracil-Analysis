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
                          'stop'  : datetime(2019,4,1,2,52,0).timestamp(),
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

evConv = ou.evConv()
maxima = []
integ =  []
for i in range(49):
    max = evConv[CH](tof.iloc[i::49].mean().idxmin())
    maxima.append(max)
    integ.append(tof.iloc[i::49].mean().sum())
    if i % 5 == 0:
        plt.plot( evConv[CH](tof.iloc[0].index) ,   tof.iloc[i::49].mean() + i*5 )

def fit(x, m, q):
    return x*m + q

x = np.array(range(len(maxima)))
popt, pconv = curve_fit(fit, x, maxima)

plt.figure()
plt.plot(x, maxima, 'o')
#plt.plot(x, integ)
plt.plot(x, fit(x, *popt))
print(popt)
plt.show()
