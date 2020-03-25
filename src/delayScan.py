#!/usr/bin/python3
import pandas as pd
import numpy as np
from datetime import datetime
from attrdict import AttrDict
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as tck
from scipy.optimize import curve_fit

import sys
import utils

import pickle

cfg = {    'data'     : { 'path'     : '/media/Fast2/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'fisrt_block.h5'
                          #'trace'    : 'second_block.h5'
                          #'trace'    : 'third_block.h5'
                        },
           'output'   : { 'path'     : './data/',
                          'fname'    : 'bootstrapTest'
                        },
           'time'     : { 'start' : datetime(2019,3,26,22,56,0).timestamp(), #18
                          'stop'  : datetime(2019,3,27,0,7,0).timestamp(),   #7
           #'time'     : { 'start' : datetime(2019,4,1,1,43,0).timestamp(),
           #               'stop'  : datetime(2019,4,1,2,56,0).timestamp(),
                        },
           'filters'  : { 'undulatorEV' : (260.,275.),
                          'retarder'    : (-81,-1),
                          #'delay'       : (1170, 1185.0),
                          'waveplate'   : (6,11)
                        },

           'sdfilter' : "GMD > 0.5 & BAM != 0", # filter for shotsdata parameters used in query method
           'delayBin_mode'  : 'QUANTILE', # Binning mode, must be one of CUSTOM, QUANTILE, CONSTANT
           'delayBinStep'   : 0.2,     # Size of bins, only relevant when delayBin_mode is CONSTANT
           'delayBinNum'    : 10,     # Number if bis to use, only relevant when delayBin_mode is QUANTILE
           'ioChunkSize' : 50000,
           'gmdNormalize': True,
           'useBAM'      : True,
           'timeZero'    : 1178.45,   #Used to correct delays
           'decimate'    : False, #Decimate macrobunches before analizing. Use for quick evalutation of large datasets

           'bootstrap'   : 5,  #Number of bootstrap samples to make for variance estimation. Use only for augerShift.
                                  #Set to None for everything else

           'augerROI'    : (120,160),
           'plots' : {
                       'delay2d'        : False,
                       'photoShift'     : False,
                       'auger2d'        : False,
                          'augerIntensity' : True, #Only used when auger 2d is true
                       'augerShift'     : True,
                       'valence'        : False,
           },
           'writeOutput' : True, #Set to true to write out data in csv
           'onlyplot'    : True, #Set to true to load data form 'output' file and
                                 #plot only.
      }

cfg = AttrDict(cfg)

if cfg.bootstrap:
    if cfg.plots.delay2d or cfg.plots.photoShift or cfg.plots.auger2d or cfg.plots.valence == True:
        print("Bootstrap mode can be used only for augerShift")
        exit()

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
        pulses = pulses.query('index % 10 == 0')

    shotsData = utils.h5load('shotsData', tr, pulses)

    #Remove pulses with no corresponing shots
    pulses = pulses.drop( pulses.index.difference(shotsData.index.levels[0]) )

    #Plot relevant parameters
    #utils.plotParams(shotsData)

    uvEven = shotsData.query("shotNum % 2 == 0").uvPow.mean()
    uvOdd = shotsData.query("shotNum % 2 == 1").uvPow.mean()

    if uvEven > uvOdd:
        print("Even shots are UV pumped.")
    else:
        print("Odd shots are UV pumped.")

    if cfg.gmdNormalize:
        gmdData = shotsData.GMD
    else:
        gmdData = None

    #Add Bam info
    shotsNum = len(shotsData.index.levels[1]) // 2
    if uvEven > uvOdd:
        shotsData = shotsData.query('shotNum % 2 == 0')
    else:
        shotsData = shotsData.query('shotNum % 2 == 1')

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
        binStart, binEnd = -0.5, 1#utils.getROI(shotsData, limits=(-5,20))
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

    #Create empty ouptut image in diffAcc (difference accumulator)
    #binCount counts how many shots go in each bin, to average them
    if cfg.bootstrap:
        #For each delaybin, we need to create cfg.bootstrap samples of it
        #bsBin is a dictionary associating a list of bs samples to each
        #delay bin key
        bsBins = {}
        for name, group in delayBins:
            bsBins[name] = []
            for n in range(cfg.bootstrap):
                bsBins[name].append( group.sample(frac=1, replace=True).index )

        #Output arrays have an extra dimension to index each bs sample
        bsDiffAcc  = np.zeros(( len(delayBins), cfg.bootstrap, 3009 ))
        bsBinCount = np.zeros(( len(delayBins), cfg.bootstrap ))

    #Normally we just have a trace for each delay
    diffAcc  = np.zeros(( len(delayBins), 3009 ))
    binCount = np.zeros(  len(delayBins) )

    #Iterate over data chunks and accumulate them in diffAcc
    for counter, chunk in enumerate(shotsTof):
        print( f"loading chunk {counter} of {shotsCount//cfg.ioChunkSize}", end='\r' )
        shotsDiff = utils.getDiff(chunk, gmdData)

        for binId, delayBin in enumerate(delayBins):
            delayName, group = delayBin
            group = group.query(cfg.sdfilter)

            #If we are bootstrapping, we need to iterate over the bs samples as well
            if cfg.bootstrap:
                for bsBinId, bsGroupIdx in enumerate(bsBins[name]):
                    bsBinIdx   = shotsDiff.index.intersection(bsGroupIdx)
                    bsBinTrace = shotsDiff.reindex(bsBinIdx)
                    bsDiffAcc[binId, bsBinId]  += -bsBinTrace.sum(axis=0)
                    bsBinCount[binId, bsBinId] += bsBinTrace.shape[0]

            delayBinIdx   = shotsDiff.index.intersection(group.index)
            delayBinTrace = shotsDiff.reindex(delayBinIdx)
            diffAcc[binId]  += -delayBinTrace.sum(axis=0)
            binCount[binId] += delayBinTrace.shape[0]


    idx.close()
    tr.close()

    if cfg.bootstrap:
        bsDiffAcc /= bsBinCount[:,:,None]
    diffAcc /= binCount[:,None]

    #get axis labels
    delays = np.array( [name.mid for name, _ in delayBins] )
    #delays = np.array([bins["delay"].mean().values, bins["delay"].std().values])

    evConv = utils.mainTofEvConv(pulses.retarder.mean())
    evs = evConv(shotsDiff.iloc[0].index.to_numpy(dtype=np.float32))

    # make dataframe and save data
    if cfg.writeOutput:
        if cfg.bootstrap:
            np.savez(cfg.output.path + cfg.output.fname,
                     diffAcc = diffAcc, bsDiffAcc=bsDiffAcc, evs=evs, delays=delays)

if cfg.onlyplot:
    dataZ = np.load(cfg.output.path + cfg.output.fname + ".npz")
    diffAcc = dataZ['diffAcc']
    delays = dataZ['delays']
    evs    = dataZ['evs']

    try:
        bsDiffAcc = dataZ['bsDiffAcc']
    except KeyError:
        pass

#plot resulting image
if cfg.plots.delay2d:
    ROI = slice(np.abs(evs - 270).argmin() , None)
    plt.figure(figsize=(9, 7))
    plt.suptitle("Kinetic Energy vs Delay.")
    cmax = np.percentile(np.abs(diffAcc[:,ROI]),99.5)
    plt.pcolormesh(evs[ROI], delays, diffAcc[:,ROI],
                   cmap='bwr', vmax=cmax, vmin=-cmax)
    plt.xlabel("Kinetic energy (eV)")
    plt.ylabel("Delay (ps)")

if cfg.plots.augerShift:
    #Calulates cross correlation between each row of a and b
    def xCorr(a, b):
        return np.fft.irfft( np.fft.rfft(a) * np.conj( np.fft.rfft(b) ), n=len)

    def getZeroCrossing(diffTraces):
        ROI = slice(np.abs(evs - cfg.augerROI[1]).argmin() , np.abs(evs - cfg.augerROI[0]).argmin())
        #Slice Traces over Auger ROI
        sliced = diffTraces[:, ROI]
        slicedEvs = evs[ROI]
        len = sliced.shape[1]

        #Offset traces so that they are on average centered around 0 (using start and end data)
        avg = len // 10
        offset = ( sliced[:,:avg].mean(axis = 1) + sliced[:,-avg:].mean(axis = 1) ) / 2
        sliced -= offset[:,None]

        #Find 0 crossing by maximizing the cross correlation between
        #the traces and sign function
        A = np.zeros(sliced.shape) + np.arange(len)
        sign = np.sign( A - len//2 )
        corr = xCorr(sliced, sign)
        zeroXidx = corr.argmax(axis=1)
        zeroX = slicedEvs[zeroXidx]

        #Get first moments for positive and negative sides
        posCenter = np.empty(diffAcc.shape[0])
        negCenter = np.empty(diffAcc.shape[0])
        avgDiff   = np.empty(diffAcc.shape[0]) #Difference in pos - neg avg signal
        for n, weights in enumerate(sliced):
            negCenter[n] = np.average(slicedEvs[zeroXidx[n]:], weights=weights[zeroXidx[n]:])
            posCenter[n] = np.average(slicedEvs[:zeroXidx[n]], weights=weights[:zeroXidx[n]])
            avgDiff[n] = slicedEvs[:zeroXidx[n]].mean() - slicedEvs[zeroXidx[n]:].mean()
        avgCenter = ( posCenter + negCenter ) / 2
        return zeroX, avgCenter, posCenter, negCenter

    zeroX, avgCenter, posCenter, negCenter = getZeroCrossing(diffAcc)

    f= plt.figure(figsize=(9, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = f.add_subplot(gs[0])
    plt.plot(delays, zeroX, label='zero crossing')
    plt.plot(delays, posCenter, label='positive center')
    plt.plot(delays, negCenter, label='negative center')
    plt.plot(delays, avgCenter, label='center average')
    ax1.xaxis.set_minor_locator(tck.AutoMinorLocator())
    plt.xlabel("delay [ps]")
    plt.ylabel("peak position [eV]")
    plt.legend()
    ax1 = f.add_subplot(gs[1], sharex=ax1)
    plt.xlabel("delay [ps]")
    plt.ylabel("Pos Avg - Neg Avg [au]")
    plt.plot(delays, avgDiff)

if cfg.plots.auger2d:
    ROI = slice(np.abs(evs - cfg.augerROI[1]).argmin() , np.abs(evs - cfg.augerROI[0]).argmin())

    f = plt.figure(figsize=(11, 7))
    if cfg.plots.augerIntensity:
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax1 = f.add_subplot(gs[1])

        #Normalize and shift intensity
        integ = diffAcc[:,ROI].sum(axis=1)
        integ -= integ[:2].mean()
        integ /= np.linalg.norm(integ)

        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.tick_params(axis='y', labelsize=0, length = 0)

        plt.plot(integ,delays)
        plt.xlabel(f"Integrated Auger Intensity")
        plt.legend()
        f.add_subplot(gs[0], sharey=ax1)
        f.subplots_adjust(left=0.08, bottom=0.07, right=0.96, top=0.95, wspace=None, hspace=0.05)

    plt.suptitle("Auger Kinetic Energy vs Delay.")
    cmax = np.percentile(np.abs(diffAcc[:,ROI]),99.5)
    plt.pcolormesh(evs[ROI], delays, diffAcc[:,ROI],
                   cmap='bwr', vmax=cmax, vmin=-cmax)
    plt.xlabel("Kinetic energy (eV)")
    plt.ylabel("Delay (ps)")

if cfg.plots.photoShift:
    # quick plots for check. for more detailed analysis use photolineAnalysis.py
    # plot line graph of integral over photoline features
    plt.figure()
    photoline1 = slice(np.abs(evs - 101.5).argmin() , np.abs(evs - 97.5).argmin())
    photoline2 = slice(np.abs(evs - 97.5).argmin() , np.abs(evs - 95.5).argmin())
    photoline3 = slice(np.abs(evs - 107).argmin() , np.abs(evs - 101.5).argmin())

    NegPhLine, = plt.plot(delays, abs(diffAcc.T[photoline3].sum(axis=0)), label = 'negative photoline shift')
    PosPhLine1, = plt.plot(delays, abs(diffAcc.T[photoline1].sum(axis=0)), label = 'positive photoline shift 1')
    PosPhLine2, = plt.plot(delays, abs(diffAcc.T[photoline2].sum(axis=0)), label = 'positive photoline shift 2')
    plt.xlabel("Delay (ps)")
    plt.ylabel("Integrated signal")
    plt.legend(handles=[PosPhLine1, PosPhLine2, NegPhLine])
    plt.tight_layout()

    # plot extrema as rough indication for possible shifts
    plt.figure()
    neg = diffAcc.T[photoline2].argmin(axis=0)
    pos = diffAcc.T[photoline1].argmax(axis=0)

    NegMax, = plt.plot(delays, evs[photoline2][neg], label = 'negative photoline shift')
    PosMax, = plt.plot(delays, evs[photoline1][pos], label = 'negative photoline shift')
    plt.xlabel("Delay (ps)")
    plt.ylabel("Peak position (eV)")
    plt.legend(handles=[PosMax, NegMax])
    plt.tight_layout()

if cfg.plots.valence:
    plt.figure()
    valence = slice( np.abs(evs - 145).argmin() , np.abs(evs - 140).argmin())
    plt.plot(delays, diffAcc.T[valence].sum(axis=0))

plt.show()
