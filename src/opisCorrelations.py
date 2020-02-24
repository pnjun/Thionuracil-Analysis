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
                   'filename' : 'second_block.h5',
                   'indexf'   : 'index.h5'
                 },
        'time' : { 'start' : datetime(2019,3,30,20,40,20).timestamp(),
                   'stop'  : datetime(2019,3,30,20,41,20).timestamp(),
                 },

        'averageShots'  : False,
        'ignoreMask'    : True,
        'photoline'     : 171,

        'decimate'    : False, #Decimate macrobunches before

        'plots':
                 {
                    'tracePlot'     : None,
                    'ampliGMDsc'    : True,
                    'evPhotosc'     : True,
                    'evPhotoShotbyShot' : False,
                    'traceMax'      : 21
                 }
      }

cfg = AttrDict(cfg)

h5data  = pd.HDFStore(cfg.data.path + cfg.data.filename, mode = 'r')
index   = pd.HDFStore(cfg.data.path + cfg.data.indexf, mode = 'r')

pulses = index.select('pulses',
                      where='time >= cfg.time.start and time < cfg.time.stop')

if cfg.decimate:
    print("Decimating...")
    pulses = pulses.query('index % 10 == 0')

opisData  = utils.h5load('opisFit' , h5data, pulses)
shotsTof  = utils.h5load('shotsTof', h5data, pulses)
shotsData = utils.h5load('shotsData', h5data, pulses)

#take only umpumped shots
#opisData = opisData.query('ampli > 35 and ampli < 60')
#opisData = opisData.query('shotNum % 2 == 0')


if cfg.ignoreMask:
    opisData = opisData.query('ignoreMask == False')

#Filter out shots for wich there is no opisFit data
shotsTof  = shotsTof.query('index in @opisData.index')
shotsData = shotsData.query('index in @opisData.index')

#shotsTof  = shotsTof.mean(level=1)
#shotsData = shotsData.mean(level=1)

evConv = utils.mainTofEvConv(pulses.retarder.mean())
evs = evConv(shotsTof.columns)

if cfg.plots.ampliGMDsc:
    if cfg.averageShots:
        ampli  = opisData.ampli.mean(level=1)
        gmd    = shotsData.GMD.mean(level=1)
    else:
        ampli  = opisData.ampli
        gmd    = shotsData.GMD

    plt.figure('ampliGMDsc')
    plt.plot(ampli, gmd, 'o')
    print(f'GMD-Ampli Pearson: {np.corrcoef(ampli, gmd)[0,1]}')

if cfg.plots.evPhotosc or cfg.plots.evPhotoShotbyShot:
    opisEv = opisData.ev.mean()
    photoStart = opisEv - cfg.photoline - 7
    photoEnd   = opisEv - cfg.photoline + 7
    photoline = slice(np.abs(evs - photoEnd).argmin() ,
                      np.abs(evs - photoStart).argmin() )
    photoEn = []
    for _, trace in shotsTof.iterrows():
        photoEn.append(evs[photoline][trace[photoline].values.argmin()])

    photoEn = pd.DataFrame(photoEn, columns=['ev'], index=opisData.index)

    #Average together by shot number
    if cfg.averageShots:
        opisData  = opisData.mean(level=1)
        photoEn   = photoEn.mean(level=1)

    plt.figure('evPhotolinesc')
    if cfg.plots.evPhotosc:
        plt.gca().set_aspect('equal')
        plt.gca().set_xlim([opisEv-3,opisEv+3])
        plt.plot(opisData.ev, photoEn.ev, 'o')
        print(f'ev-photoline Pear: {np.corrcoef(opisData.ev, photoEn.ev)[0,1]}')
    elif cfg.plots.evPhotoShotbyShot and not cfg.averageShots:
        for i in range(0,9):
            id = i*4
            opis =  opisData.query('shotNum == @id').ev
            photo = photoEn.query ('shotNum == @id').ev

            plt.subplot(f'33{i}')
            plt.gca().set_aspect('equal')
            plt.gca().set_xlim([opisEv-3,opisEv+3])
            plt.plot(opis, photo, 'o')
            print(f'ev-photoline Pear{id}: {np.corrcoef(opis, photo)[0,1]}')
    else:
        print('invalid plot options')

    if cfg.plots.traceMax:
        plt.figure('traceMax')
        plt.plot(evs[photoline], shotsTof.iloc[cfg.plots.traceMax][photoline])
        plt.axvline(x=photoEn.iloc[cfg.plots.traceMax].to_numpy())

if cfg.plots.tracePlot:
    plt.figure('tracePlot')
    plt.plot(evs, shotsTof.iloc[cfg.plots.tracePlot])

plt.show()
h5data.close()
index.close()
