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
                   'filename' : 'trOpistest2019-03-30T1900.h5',
                   'indexf'   : 'idOpistest2019-03-30T1900.h5'
                 },
        'time' : { 'start' : datetime(2019,3,30,20,36,2).timestamp(),
                   'stop'  : datetime(2019,3,30,20,36,58).timestamp(),
                 },

        'averageShots'  : True,

        'ignoreMask'    : True,
        'photoline'     : 171,
        'resortOpis'    : False, #Rescale opis energy subtracting undulator setting

        'decimate'    : False, #Decimate macrobunches before

        'plots':
                 {
                    'tracePlot'     : None,
                    'ampliGMDsc'    : None,
                    'evPhotosc'     : True,
                    'evPhotoShotbyShot' : True,
                    'traceMax'      : None
                 }
      }

cfg = AttrDict(cfg)

h5data  = pd.HDFStore(cfg.data.path + cfg.data.filename, mode = 'r')
index   = pd.HDFStore(cfg.data.path + cfg.data.indexf, mode = 'r')

pulses = index.select('pulses',
                      where='time >= cfg.time.start and time < cfg.time.stop')

print(f"undulator energies {pulses.undulatorEV.unique()}")

if cfg.decimate:
    print("Decimating...")
    pulses = pulses.query('index % 10 == 0')

opisData  = utils.h5load('opisFit' , h5data, pulses)
shotsTof  = utils.h5load('shotsTof', h5data, pulses)
shotsData = utils.h5load('shotsData', h5data, pulses)

#take only umpumped shots
#opisData = opisData.query('ampli > 35 and ampli < 60')
#opisData = opisData.query('shotNum % 2 == 0')

opisEv = opisData.ev.mean()
if cfg.resortOpis:
    opisData.query('shotNum < 48')
    opisData.ev = -utils.shotsDelay(pulses.undulatorEV, opisData.ev) #imporper use of shotsDelay, but it works

if cfg.ignoreMask:
    opisData = opisData.query('ignoreMask == False')

#Filter out shots for wich there is no opisFit data
shotsTof  = shotsTof.query('index in @opisData.index')
shotsData = shotsData.query('index in @opisData.index')

#shotsTof  = shotsTof.mean(level=1)
#shotsData = shotsData.mean(level=1)

evConv = utils.mainTofEvConv(pulses.retarder.mean())
evs = evConv(shotsTof.columns)

def gauss(x, ampli, center, fwhm):
    return ampli*np.exp( - 4*np.log(2.)*(x-center)**2/fwhm**2)

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
    photoStart = opisEv - cfg.photoline - 7
    photoEnd   = opisEv - cfg.photoline + 7
    photoline = slice(np.abs(evs - photoEnd).argmin() ,
                      np.abs(evs - photoStart).argmin() )
    photoEn = []
    for _, trace in shotsTof.iterrows():
        photoEn.append(evs[photoline][trace[photoline].values.argmin()])
        '''
        try:
            popt, pconv = curve_fit(gauss, evs[photoline], trace[photoline], p0=[-600,opisEv - cfg.photoline,5] )
        except Exception:
            photoEn.append(evs[photoline][trace[photoline].values.argmin()])
        else:
            photoEn.append(popt[1])'''

    photoEn = pd.DataFrame(photoEn, columns=['ev'], index=opisData.index)

    #Average together by shot number
    if cfg.averageShots:
        opisData  = opisData.mean(level=1)
        photoEn   = photoEn.mean(level=1)

    plotCenter = opisData.ev.mean()
    plt.figure('evPhotolinesc')
    if cfg.plots.evPhotosc:
        plt.gca().set_aspect('equal')
        plt.gca().set_xlim([plotCenter-3,plotCenter+3])
        plt.plot(opisData.ev, photoEn.ev, 'o')
        print(f'ev-photoline Pear: {np.corrcoef(opisData.ev, photoEn.ev)[0,1]}')
    elif cfg.plots.evPhotoShotbyShot and not cfg.averageShots:
        for i in range(0,9):
            id = i*4
            opis =  opisData.query('shotNum == @id').ev
            photo = photoEn.query ('shotNum == @id').ev

            plt.subplot(f'33{i}')
            plt.gca().set_aspect('equal')
            plt.gca().set_xlim([plotCenter-3,plotCenter+3])
            plt.plot(opis, photo, 'o')
            print(f'ev-photoline Pear{id}: {np.corrcoef(opis, photo)[0,1]}')
    else:
        print('invalid plot options')

    if cfg.plots.traceMax:
        plt.figure('traceMax')
        plt.plot(evs[photoline], shotsTof.iloc[cfg.plots.traceMax][photoline])
        plt.plot(evs[photoline], gauss(evs[photoline], -900, photoEn.iloc[cfg.plots.traceMax].to_numpy(), 4) )
        plt.axvline(x=photoEn.iloc[cfg.plots.traceMax].to_numpy())

if cfg.plots.tracePlot:
    plt.figure('tracePlot')
    plt.plot(evs, shotsTof.iloc[cfg.plots.tracePlot])

plt.show()
h5data.close()
index.close()
