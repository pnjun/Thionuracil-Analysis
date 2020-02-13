#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict
from scipy.optimize import curve_fit

import sys
import opisUtils as ou
from utils import h5load

import pickle

cfg = { 'data' : { 'path'     : '/media/Fast1/ThioUr/processed/',
                   'filename' : 'opistest.h5',
                   'indexf'   : 'index.h5'
                 },
        'time' : { #'start' :datetime(2019,3,31,8,25,0).timestamp(),
                   #'stop'  :datetime(2019,3,31,8,25,10).timestamp(),
                   'start' : datetime(2019,4,1,3,10,0).timestamp(),
                   'stop'  : datetime(2019,4,1,3,10,2).timestamp(),
                 },
        'plots':
                 {
                    'traceID' : 29,
                    'diffs'      : True,
                    'wlHist'     : False,
                    'ampliHist'  : False,
                    'integHist'  : False,
                    'fwhmHist'   : False,
                    'mask'       : False,
                    'wlTrend'    : False,
                    'ampliTrend' : False,
                    'fwhmTrend'  : False,
                    'integTrend' : False,
                    'rawTraces'  : False,
                    'fittedTr'   : True,
                    'intAmpSc'   : False,
                    'AmpFhwmSc'   : False,
                    'integAreaSc' : False
                 }
      }

cfg = AttrDict(cfg)

h5data  = pd.HDFStore(cfg.data.path + cfg.data.filename, mode = 'r')
index   = pd.HDFStore(cfg.data.path + cfg.data.indexf, mode = 'r')

pulses = index.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')

tofs =   [ h5load(f'tof{n}', h5data, pulses) for n in range(4) ]
traces   = [ tofs[n] for n in range(4) ]

#evConv = ou.geometricEvConv(170)
evConv = ou.calibratedEvConv()
GUESSEV = pulses.undulatorEV.iloc[0]

if False:
    amplR = np.linspace(5, 60, 20)
    enerR = np.linspace(-5, 5, 16) + GUESSEV
    fwhmR = np.linspace(1,3,8)

    fitter = ou.evFitter(ou.GAS_ID_AR)
    fitter.loadTraces(traces, evConv, GUESSEV)
    diffs = fitter.getOffsets(getDiffs=cfg.plots.diffs)
    fit   = fitter.leastSquare3Params(amplR, enerR, fwhmR)
    #fit   = fitter.leastSquare(amplR, enerR)
else:
    tr = cfg.plots.traceID
    GUESSEV = 215
    fitter = ou.evFitter2ElectricBoogaloo(ou.GAS_ID_AR)
    fitter.loadTraces(traces, evConv, GUESSEV)

    '''
    import cupy as cp
    fft = cp.fft.rfft(fitter.traces[tr,0, :])
    lowPass = 0
    fft[-lowPass:] = (1-cp.arange(0,1,lowPass))
    fitter.traces[tr,0, :] = cp.fft.irfft(fft, n=fitter.traceLen)
    '''

    diffs = fitter.getOffsets(getXC = cfg.plots.diffs)
    fit   = fitter.leastSquare(GUESSEV)
    print(fitter.of02[tr])

    '''
    ene = fitter._evs(0,0).get()
    trace = fitter.traces[tr,0].get()

    for f in fit:
        plt.figure()
        print(f[tr,0], f[tr,1], f[tr,2])
        plt.plot(ene, trace)
        plt.plot(ene, fitter.spectrumFit(ene, f[tr,0], f[tr,1], f[tr,2]))
        plt.show()'''

print(f'Trace fit results: {fit[cfg.plots.traceID]}')


##########################################################

if cfg.plots.diffs:
    plt.figure()
    plt.plot(diffs[0][cfg.plots.traceID])
    plt.plot(diffs[1][cfg.plots.traceID])

#invalidate low signal shots
mask, rawm = fitter.ignoreMask(0,getRaw=True)
fitm = fit.copy()
fitm[mask] = np.NaN
print(f'Trace integrals: {rawm[cfg.plots.traceID]}')

def linFit(x, m, q):
    return x*m + q

if cfg.plots.intAmpSc:
    plt.figure('intAmpSc')
    integ = rawm.sum(axis=1)
    plt.plot(integ, -fit[:,1], 'o')
    print(f'Integral-Amplitude Pearson: {np.corrcoef(integ, -fit[:,1])[0,1]}')

if cfg.plots.AmpFhwmSc:
    plt.figure('AmpFhwmSc')
    plt.plot(fit[:,2], -fit[:,1], 'o')
    print(f'Amplitude-FWHM Pearson: {np.corrcoef(fit[:,2], -fit[:,1])[0,1]}')

if cfg.plots.integAreaSc:
    plt.figure('integAreaSc')
    integ = rawm.sum(axis=1)
    area  = fit[:,2] * -fit[:,1]
    popt, pconv = curve_fit(linFit, integ, area)
    plt.plot(integ, area-linFit(integ, *popt), 'o')
    #plt.plot(integ, linFit(integ, *popt))
    print(f'Fit results {popt}')
    print(f'Integral-Area Pearson: {np.corrcoef(integ, area)[0,1]}')

fit = fit.reshape((fit.shape[0]//49,49,3))
fitm = fitm.reshape((fitm.shape[0]//49,49,3))
rawm = rawm.reshape((rawm.shape[0]//49,49,4))

if cfg.plots.wlHist:
    plt.figure('WL histogram', figsize=(7, 5))
    for i in range(6):
        idx = i*9
        plt.subplot(2,3,i+1, title = f'shot {idx}')
        plt.hist(fit[:,idx,0])

if cfg.plots.ampliHist:
    plt.figure('AMPLI histogram', figsize=(7, 5))
    for i in range(6):
        idx = i*9
        plt.subplot(2,3,i+1, title = f'shot {idx}')
        plt.hist(-fit[:,idx,1])

if cfg.plots.fwhmHist:
    plt.figure('FHWM histogram', figsize=(7, 5))
    for i in range(6):
        idx = i*9
        plt.subplot(2,3,i+1, title = f'shot {idx}')
        plt.hist(fit[:,idx,2])

if cfg.plots.integHist:
    plt.figure('INTEG histogram', figsize=(10, 7))
    for i in range(6):
        idx = i*9
        plt.subplot(2,3,i+1, title = f'shot {idx}')
        plt.hist(rawm[:,idx])

fit = np.nanmean( fit, axis=0 )
fitm = np.nanmean( fitm , axis=0 )

if cfg.plots.mask:
    plt.figure('mask')
    plt.plot(np.mean( mask.reshape((mask.shape[0]//49,49)), axis=0))
if cfg.plots.wlTrend:
    plt.figure('wavelenght')
    plt.plot(fit[:,0])
    plt.plot(fitm[:,0])
if cfg.plots.ampliTrend:
    plt.figure('AMPLI')
    plt.plot(-fit[:,1])
    plt.plot(-fitm[:,1])
if cfg.plots.fwhmTrend:
    plt.figure('FWHM')
    plt.plot(fit[:,2])
    plt.plot(fitm[:,2])

if cfg.plots.rawTraces:
    fitter.plotTraces(cfg.plots.traceID,show=False)

if cfg.plots.fittedTr:
    fitter.plotFitted(cfg.plots.traceID,show=False)

plt.show()
h5data.close()
index.close()
