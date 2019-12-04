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
                          'filename' : 'opistest_.h5'
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

tofs =   [ h5data.select(f'tof{n}',
               where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]',
                      'pulseId in pulses.index'] )
           for n in range(4) ]


traces   = [ tofs[n] for n in range(4) ]
evConv = ou.evConv()

amplR = np.linspace(-80, -0, 12)
enerR = np.linspace(210, 230, 48)

TRACEID = 26
fitter = ou.evFitter(ou.GAS_ID_AR)
fitter.loadTraces(traces, evConv, 222)
fitter.getOffsets()
fit = fitter.leastSquare(amplR, enerR)

fit = fit.reshape((fit.shape[0]//49,49,2))
plt.plot(fit.mean(axis=0)[:,1])
plt.plot(fit.mean(axis=0)[:,0])
plt.show()
#fitter.plotFitted(TRACEID)
