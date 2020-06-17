#!/usr/bin/python3
import pandas as pd
import numpy as np
from datetime import datetime
from attrdict import AttrDict
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as tck
from scipy.optimize import curve_fit
from scipy.signal import correlate
from scipy import special
import time

import sys
import utils

import pickle

import gc
import os
import psutil
process = psutil.Process(os.getpid())

cfg = {    'data'     : { 'path'     : '/media/Fast2/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'fisrt_block.h5'
                          #'trace'    : 'second_block.h5'
                          #'trace'    : 'third_block.h5'
                        },
           'output'   : { 'path'     : './data/',
                          'fname'    : 'excitedScanB30_'
                        },
           'time'     : { 'start' : datetime(2019,3,26,18,56,0).timestamp(), #18
                          'stop'  : datetime(2019,3,27,7,7,0).timestamp(),   #7
           #'time'     : { 'start' : datetime(2019,4,1,1,43,0).timestamp(),
           #               'stop'  : datetime(2019,4,1,2,56,0).timestamp(),
                        },
           'filters'  : { 'undulatorEV' : (260.,275.),
                          'retarder'    : (-81,-1),
                          #'delay'       : (1170, 1185.0),
                          'waveplate'   : (6,11)
                        },
           'interactive' : False,       #Set to true to suppress initial plots. delay range defaults to -0.5,3
           'timeBounds'  : (-0.6,2.2),  #Delay bounds used if not interactive
           'decimate'    : False,       #Decimate macrobunches before analizing. Use for quick evalutation of large datasets
           'timeZero'    : 1178.45,     #Used to correct delays

           'sdfilter' : "GMD > 0.5 & BAM != 0", # filter for shotsdata parameters used in query method
           'delayBin_mode'  : 'QUANTILE', # Binning mode, must be one of CUSTOM, QUANTILE, CONSTANT
           'delayBinStep'   : 0.2,     # Size of bins, only relevant when delayBin_mode is CONSTANT
           'delayBinNum'    : 30,     # Number if bis to use, only relevant when delayBin_mode is QUANTILE

           'gmdNormalize': True,
           'useBAM'      : True,
           'ioChunkSize' : 50000,

           'augerROI'      : (125,155),
           'exc_frac'      : 0.22,        #Excited fraction to use for excited spectrum reconstruction

           'pulse_t0'      :-0.020,       #Additional offset on UV pulse center (so that delayscale remains consisten with online value)
           'pulse_len'     : 0.100,       #UV pulse lenght in ps, for excited_frac convolution. Use None for constant exct_frac
           'exc_normalize' : True,        #Set to True to renormalize the excited spectrum by 1/exct_frac
           'min_frac'      : 0.05,        #Ignore bins where the excited fraction is less than setpoint

           'plots' : {
                       'tracePlots'     : False,
                       'delay2d'        : True,
                       'delay3d'        : (110,160),   #'waterfall' 3d plot
                       'augerShift'     : True,        #Only works with exc_frac data
                       'frac_plot'      : False,
                     },

           'writeOutput' : True, #Set to true to write out data in csv
           'onlyplot'    : True, #Set to true to load data form 'output' file and
                                 #plot only.
      }

cfg = AttrDict(cfg)

if not cfg.onlyplot:
    idx = pd.HDFStore(cfg.data.path + cfg.data.index, mode = 'r')
    tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, mode = 'r')

    #Get all pulses within time limits
    pulses = idx.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')
    #Filter only pulses with parameters in range
    fpulses = utils.filterPulses(pulses, cfg.filters)

    #Get corresponing shots
    if(not len(fpulses)):
        print(f"Avg Retarder  {pulses.retarder.mean()}")
        print(f"Avg Undulator {pulses.undulatorEV.mean()}")
        print(f"Avg Waveplate {pulses.waveplate.mean()}")
        raise Exception("No pulses satisfy filter condition")
    else:
        pulses = fpulses

    if cfg.decimate:
        print("Decimating...")
        #pulses = pulses.query('index % 2 == 0 and index % 10 != 0')
        pulses = pulses.query('index % 10 == 0')

    shotsData = utils.h5load('shotsData', tr, pulses)


    #Remove pulses with no corresponing shots
    pulses = pulses.drop( pulses.index.difference(shotsData.index.levels[0]) )

    #Plot relevant parameters
    if cfg.interactive: utils.plotParams(shotsData)

    uvEven = shotsData.query("shotNum % 2 == 0").uvPow.mean()
    uvOdd = shotsData.query("shotNum % 2 == 1").uvPow.mean()

    if uvEven > uvOdd:
        print("Even shots are UV pumped.")
    else:
        print("Odd shots are UV pumped. Terminating")
        exit()

    if cfg.gmdNormalize:
        gmdData = shotsData.GMD
    else:
        gmdData = None

    #Add Bam info
    shotsNum = len(shotsData.index.levels[1]) // 2
    shotsData = shotsData.query('shotNum % 2 == 0')

    if cfg.useBAM:
        shotsData.BAM = shotsData.BAM.fillna(0)
        averageBamShift = shotsData.query(cfg.sdfilter).BAM.mean()
        print(f"Correcting delays with BAM data. Average shift is {averageBamShift:.3f} ps")
        shotsData['delay'] = utils.shotsDelay(pulses.delay.to_numpy(), shotsData.BAM.to_numpy())
    else:
        shotsData['delay'] = utils.shotsDelay(pulses.delay.to_numpy(), shotsNum = shotsNum)
        averageBamShift = np.float32(0.)
    shotsData.delay = cfg.timeZero - averageBamShift - shotsData.delay

    shotsCount = shotsData.shape[0]*2
    print(f"Loading {shotsCount} shots")


    #chose binning dependent on delaybin_mode
    if cfg.delayBin_mode == 'CUSTOM': # insert your binning intervals here
        print(f"Setting up customized bins")
        interval = pd.IntervalIndex.from_arrays(utils.CUSTOM_BINS_LEFT - averageBamShift,
                                                utils.CUSTOM_BINS_RIGHT - averageBamShift)
        delayBins = shotsData.groupby( pd.cut(shotsData.delay, interval) )

    else:
    	#choose from a plot generated
        if cfg.interactive:
            binStart, binEnd = utils.getROI(shotsData, limits=(-5,20))
        else:
            binStart, binEnd = cfg.timeBounds
        print(f"Binning interval {binStart} : {binEnd}")

        #Bin data on delay
        if cfg.delayBin_mode == 'CONSTANT':
            delayBins = shotsData.groupby( pd.cut( shotsData.delay,
                                              np.arange(binStart, binEnd, cfg.delayBinStep) ) )
        elif cfg.delayBin_mode == 'QUANTILE':
       	    shotsData = shotsData[ (shotsData.delay > binStart) & (shotsData.delay < binEnd) ]
       	    delayBins = shotsData.groupby( pd.qcut( shotsData.delay, cfg.delayBinNum ) )
        else:
            raise Exception("binning mode not valid")

    #Read in TOF data and calulate difference, in chunks
    shotsTof  = utils.h5load('shotsTof', tr, pulses, chunk=cfg.ioChunkSize)

    #Accumulators for ground state and excited state spectra
    exAcc  = np.zeros(( len(delayBins), 3009 ))
    gsAcc  = np.zeros( 3009 )

    binCount = np.zeros(  len(delayBins) )
    gsCount  = 0
    tStart= time.time()

    #Iterate over data chunks and sort by delaybin
    for counter, chunk in enumerate(shotsTof):
        print( f"loading chunk {counter} of {shotsCount//cfg.ioChunkSize}", end='\r' )
        gsTraces = chunk.query('shotNum % 2 == 1')
        gsAcc   += -utils.traceAverage(gsTraces, accumulate=True)
        gsCount += gsTraces.shape[0]

        for binId, delayBin in enumerate(delayBins):
            delayName, group = delayBin
            group = group.query(cfg.sdfilter)

            delayBinIdx   = group.index.intersection(chunk.index)
            delayBinTrace = chunk.reindex(delayBinIdx)

            exAcc[binId]    += -utils.traceAverage(delayBinTrace, accumulate=True)
            binCount[binId] += delayBinTrace.shape[0]

    print(f"Binning took {time.time()-tStart} s to run")

    idx.close()
    tr.close()

    #get axis labels
    delays = np.array( [name.mid for name, _ in delayBins] )

    evConv = utils.mainTofEvConv(pulses.retarder.mean())
    evs = evConv(chunk.iloc[0].index.to_numpy(dtype=np.float32))

    exAcc /= binCount[:,None]
    gsAcc /= gsCount
    exAcc = utils.jacobianCorrect(exAcc, evs)
    gsAcc = utils.jacobianCorrect(gsAcc, evs)

    # make dataframe and save data
    if cfg.writeOutput:
        np.savez(cfg.output.path + cfg.output.fname,
                 exAcc = exAcc, gsAcc=gsAcc, evs=evs, delays=delays)

if cfg.onlyplot:
    dataZ = np.load(cfg.output.path + cfg.output.fname + ".npz")
    exAcc = dataZ['exAcc']
    gsAcc = dataZ['gsAcc']

    delays = dataZ['delays']
    evs    = dataZ['evs']


if cfg.pulse_len:
    f = cfg.exc_frac * 0.5*(special.erf( (delays-cfg.pulse_t0) / cfg.pulse_len )+1)
else:
    f = np.ones(exAcc.shape[0])*cfg.exc_frac

f = f.reshape((-1,1))
exAcc = (exAcc - (1-f)*gsAcc )

if cfg.exc_normalize:
    exAcc /= f

if cfg.plots.frac_plot:
    plt.plot(delays, f)

if cfg.min_frac:
    exAcc  = exAcc[ f[:,0] > cfg.min_frac, : ]
    delays = delays[ f[:,0] > cfg.min_frac ]

if cfg.plots.tracePlots:
    plt.figure(figsize=(9, 7))
    plt.plot(evs, gsAcc, label="Ground State")
    plt.plot(evs, exAcc[-1], label="Excited State")
    plt.legend()

#plot resulting image
if cfg.plots.delay2d:
    ROI = slice(np.abs(evs - 270).argmin() , None)
    plt.figure(figsize=(9, 7))
    plt.suptitle("Kinetic Energy vs Delay.")
    cmax = np.percentile(np.abs(exAcc[:,ROI]),99.5)
    plt.pcolormesh(evs[ROI], delays, exAcc[:,ROI],
                   cmap='bwr', vmax=cmax, vmin=-cmax)
    plt.xlabel("Kinetic energy (eV)")
    plt.ylabel("Delay (ps)")

if cfg.plots.delay3d:
    from mpl_toolkits.mplot3d import Axes3D

    ROI = slice(np.abs(evs - cfg.plots.delay3d[1]).argmin() , np.abs(evs - cfg.plots.delay3d[0]).argmin())
    #Slice Traces over Auger ROI
    sliced = exAcc[:, ROI].copy()
    slicedEvs = evs[ROI]

    fig = plt.figure(figsize=(11, 8))
    ax = plt.axes(projection='3d')

    X, Y = np.meshgrid(slicedEvs, delays)
    ax.plot_surface(X,Y, sliced, edgecolor='black')

if cfg.plots.augerShift:
    EDGE_RATIO = 0.2 #Edge is detected whe signal goes over 20% of peak height

    ROI = slice(np.abs(evs - cfg.augerROI[1]).argmin() , np.abs(evs - cfg.augerROI[0]).argmin())
    #Slice Traces over Auger ROI
    sliced    = exAcc[:, ROI].copy()
    slicedGS  = gsAcc[ROI].copy()
    slicedEvs = evs[ROI]
    len = sliced.shape[1]

    #lowPass = 50
    #sliced = np.fft.rfft(sliced, axis=1)
    #sliced[:,-lowPass:] *= (1-np.arange(0,1,lowPass))
    #sliced = np.fft.irfft(sliced, axis=1, n=len)

    peakIdx   = sliced.argmax(axis=1)
    peakGSIdx = slicedGS.argmax()
    peakPos = slicedEvs[peakIdx] - slicedEvs[peakGSIdx]
    peakVal = sliced[ np.arange(peakIdx.shape[0]) , peakIdx ].reshape((-1,1))

    edgeIdx   = np.abs( sliced   - peakVal*EDGE_RATIO ).argmin(axis=1)
    edgeGSIdx = np.abs( slicedGS - slicedGS[peakGSIdx]*EDGE_RATIO ).argmin()
    edgePos = slicedEvs[edgeIdx] - slicedEvs[edgeGSIdx]

    f= plt.figure(figsize=(9, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    ax1 = f.add_subplot(gs[0])
    plt.gca().xaxis.set_minor_locator(tck.AutoMinorLocator())
    plt.ylabel("peak shift [eV]")
    plt.plot(delays[:], peakPos[:], '+', color='C0')
    plt.plot(delays[1:-1], utils.movAvg(peakPos[:],3), '-', color='C0')

    ax1 = f.add_subplot(gs[1], sharex=ax1)
    plt.xlabel("delay [ps]")
    plt.ylabel("edge shift [eV]")
    plt.plot(delays[:], edgePos[:], 'x', color='C1')
    plt.plot(delays[1:-1], utils.movAvg(edgePos[:],3), '-', color='C1')

plt.show()
