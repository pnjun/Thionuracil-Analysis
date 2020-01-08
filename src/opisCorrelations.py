#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict
from scipy.optimize import curve_fit

import sys
import utils

import pickle

cfg = { 'data' : { 'path'     : '/media/Fast2/ThioUr/processed/',
                   'filename' : 'tracesOpistest.h5',
                   'indexf'   : 'index.h5'
                 },
        'time' : { 'start' : datetime(2019,4,1,3,15,0).timestamp(),
                   'stop'  : datetime(2019,4,1,3,16,20).timestamp(),
                 },
        'plots':
                 {
                    'tracePlot'     : None,
                    'ampliGMDsc'    : False,
                    'evPhotolinesc' : True,
                    'traceMax'      : None
                 }
      }

cfg = AttrDict(cfg)

h5data  = pd.HDFStore(cfg.data.path + cfg.data.filename, mode = 'r')
index   = pd.HDFStore(cfg.data.path + cfg.data.indexf, mode = 'r')

pulses = index.select('pulses',
                      where='time >= cfg.time.start and time < cfg.time.stop')

opisData  = utils.h5load('opisFit' , h5data, pulses)
shotsTof  = utils.h5load('shotsTof', h5data, pulses)
shotsData = utils.h5load('shotsData', h5data, pulses)

#take only umpumped shots
#opisData = opisData.query('ampli > 35 and ampli < 60')
opisData = opisData.query('shotNum % 2 == 0')
opisData = opisData.dropna()

#Filter out shots for wich there is no opisFit data
shotsTof  = shotsTof.query('index in @opisData.index')
shotsData = shotsData.query('index in @opisData.index')

#Average together by shot number
opisData  = opisData.mean(level=1)
shotsTof  = shotsTof.mean(level=1)
shotsData = shotsData.mean(level=1)

evConv = utils.mainTofEvConv(pulses.retarder.mean())
evs = evConv(shotsTof.columns)

if cfg.plots.ampliGMDsc:
    plt.figure('ampliGMDsc')
    plt.plot(opisData.ampli, shotsData.GMD, 'o')
    print(f'GMD-Ampli Pearson: {np.corrcoef(opisData.ampli, shotsData.GMD)[0,1]}')

if cfg.plots.evPhotolinesc:
    photoline = slice(np.abs(evs - 65).argmin() , np.abs(evs - 45).argmin())
    photoEn = []
    for index, trace in shotsTof.iterrows():
        photoEn.append(evs[photoline][trace[photoline].values.argmin()])

    photoEn = pd.DataFrame(photoEn, columns=['ev'], index=opisData.index)

    plt.figure('evPhotolinesc')
    plt.plot(opisData.ev, photoEn.ev, 'o')

    ''' for i in range(0,9):
        id = i*4
        opisData =  opisData.query('index == @id')
        photoEn = photoEn.query('index == @id')
        plt.subplot(f'33{i}')
        plt.plot(opisData.ev, photoEn.ev, 'o')
        print(f'ev-photoline Pear {i}: {np.corrcoef(opisData.ev, photoEn.ev)[0,1]}')'''

    if cfg.plots.traceMax:
        plt.figure('traceMax')
        plt.plot(evs[photoline], shotsTof.iloc[cfg.plots.traceMax][photoline])
        plt.axvline(x=photoEn[cfg.plots.traceMax])

if cfg.plots.tracePlot:
    plt.figure('tracePlot')
    plt.plot(evs, shotsTof.iloc[cfg.plots.tracePlot])

plt.show()
h5data.close()
index.close()
