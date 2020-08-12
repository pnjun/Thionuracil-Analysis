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
                          #'fname'    : 'DelayScanZoom3_q22_270eV',
                          'fname'    : 'DelayScanZoom2_q30_270eV'
                          #'fname'    : 'excFrac_test'
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
           'delayBinNum'    : 20,     # Number if bis to use, only relevant when delayBin_mode is QUANTILE

           'gmdNormalize': True,
           'useBAM'      : True,

           'ioChunkSize' : 50000,
           'bootstrap'   : None,  #Number of bootstrap samples to make for variance estimation. Use only for aaugerZeroX.
                                  #Set to None for everything else
           'augerROI'    : (125,155),
           'delayOffset' : -0.04,  #Additional shift to time zero for plotting
           'plots' : {
                       'delay2d'        : False,
                       'photoShift'     : False,
                       'auger2d'        : True,        #None, "STANDARD" or "CONTOUR"
                          'augerIntensity' : False,     #Only used when auger 2d is true
                       'augerZeroX'     : True,
                       'timeZero'       : False#(130,140)
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
        bsDiffAcc  = np.zeros(( cfg.bootstrap, len(delayBins), 3009 ))
        bsBinCount = np.zeros(( cfg.bootstrap, len(delayBins) ))

    #Normally we just have a trace for each delay
    diffAcc  = np.zeros(( len(delayBins), 3009 ))
    binCount = np.zeros(  len(delayBins) )
    tStart= time.time()
    #Iterate over data chunks and accumulate them in diffAcc
    for counter, chunk in enumerate(shotsTof):
        print( f"loading chunk {counter} of {shotsCount//cfg.ioChunkSize}", end='\r' )
        shotsDiff = utils.getDiff(chunk, gmdData)

        for binId, delayBin in enumerate(delayBins):
            delayName, group = delayBin
            group = group.query(cfg.sdfilter)
            #print( f"{binId} | process mem  {process.memory_info().rss/1e6} Mb")
            #If we are bootstrapping, we need to iterate over the bs samples as well
            if cfg.bootstrap:
                for bsBinId, bsGroupIdx in enumerate(bsBins[delayName]):
                    try:
                        bsBinTrace = shotsDiff.iloc[ shotsDiff.index.isin(bsGroupIdx) ]
                        #bsBinTrace = shotsDiff.query('pulseId in @bsGroupIdx.get_level_values(0)')
                        #bsBinTrace = shotsDiff.reindex(bsGroupIdx).dropna()
                        bsDiffAcc[bsBinId, binId]  += -bsBinTrace.sum(axis=0)
                        bsBinCount[bsBinId, binId] += bsBinTrace.shape[0]
                    except KeyError:
                        pass

            delayBinIdx   = group.index.intersection(shotsDiff.index)
            delayBinTrace = shotsDiff.reindex(delayBinIdx)
            diffAcc[binId]  += -delayBinTrace.sum(axis=0)
            binCount[binId] += delayBinTrace.shape[0]

    print(f"Binning took {time.time()-tStart} s to run")

    idx.close()
    tr.close()

    #get axis labels
    delays = np.array( [name.mid for name, _ in delayBins] )
    #delays = np.array([bins["delay"].mean().values, bins["delay"].std().values])

    evConv = utils.mainTofEvConv(pulses.retarder.mean())
    evs = evConv(shotsDiff.iloc[0].index.to_numpy(dtype=np.float32))

    if cfg.bootstrap:
        bsDiffAcc /= bsBinCount[:,:,None]
    diffAcc /= binCount[:,None]

    diffAcc = utils.jacobianCorrect(diffAcc, evs)

    # make dataframe and save data
    if cfg.writeOutput:
        if cfg.bootstrap:
            #If a file with same name already exists, append the new bs samples to it without overwriting
            #This sidesteps the limit on bs samples we can run in one go due to the memory leak in pd.index.isin
            #and allows us to split the calculation in multiple batches.
            #Warning, appendind only makes sense if the time frames and binning parameters are the same
            #for each run!!
            try:
                dataZ = np.load(cfg.output.path + cfg.output.fname + ".npz")
                oldBs = dataZ['bsDiffAcc']
                bsDiffAcc = np.vstack((bsDiffAcc, oldBs))
            except (FileNotFoundError, KeyError):
                pass

            np.savez(cfg.output.path + cfg.output.fname,
                     diffAcc = diffAcc, bsDiffAcc=bsDiffAcc, evs=evs, delays=delays)
        else:
            np.savez(cfg.output.path + cfg.output.fname,
                     diffAcc = diffAcc, evs=evs, delays=delays)

if cfg.onlyplot:
    dataZ = np.load(cfg.output.path + cfg.output.fname + ".npz")
    diffAcc = dataZ['diffAcc']
    delays = dataZ['delays']
    evs    = dataZ['evs']

    try:
        bsDiffAcc = dataZ['bsDiffAcc']
    except KeyError:
        pass

if '_noJacobian' in cfg.output.fname:
    print('correcting jacobian')
    diffAcc   = utils.jacobianCorrect(diffAcc, evs)
    try:
        bsDiffAcc = utils.jacobianCorrect(bsDiffAcc, evs)
        np.savez(cfg.output.path + cfg.output.fname[:-11],
                 diffAcc = diffAcc, bsDiffAcc=bsDiffAcc, evs=evs, delays=delays)
    except:
        np.savez(cfg.output.path + cfg.output.fname[:-11],
                 diffAcc = diffAcc, evs=evs, delays=delays)

if cfg.delayOffset:
    delays -= cfg.delayOffset

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


if cfg.plots.augerZeroX:
    #Calulates cross correlation between a and sign function, seems to work
    def signCorr(a):
        cuma = np.cumsum(a, axis=1)
        suma = np.sum(a, axis=1)
        return 2*cuma - suma[:,None]

    def getZeroCrossing(diffTraces):
        ROI = slice(np.abs(evs - cfg.augerROI[1]).argmin() , np.abs(evs - cfg.augerROI[0]).argmin())
        #Slice Traces over Auger ROI
        sliced = diffTraces[:, ROI]#.copy()
        slicedEvs = evs[ROI]
        len = sliced.shape[1]

        #Offset traces so that they are on average centered around 0
        sliced -= sliced[:,:].mean()
        corr = signCorr(sliced)

        zeroXidx = corr.argmax(axis=1)
        zeroX = slicedEvs[zeroXidx]

        #Get first moments for positive and negative sides
        posCenter = np.empty(diffAcc.shape[0])
        negCenter = np.empty(diffAcc.shape[0])
        avgDiff   = np.empty(diffAcc.shape[0]) #Difference in pos - neg avg signal
        for n, weights in enumerate(sliced):
            if zeroXidx[n] == 0: zeroXidx[n] += 1
            if zeroXidx[n] == len: zeroXidx[n] -= 1
            negCenter[n] = np.average(slicedEvs[zeroXidx[n]:], weights=np.abs(weights[zeroXidx[n]:]))
            posCenter[n] = np.average(slicedEvs[:zeroXidx[n]], weights=np.abs(weights[:zeroXidx[n]]))
            avgDiff[n] = sliced[n,sliced[n,:] > 0].sum() - sliced[n,sliced[n,:] < 0].sum()
        avgCenter = ( posCenter + negCenter ) / 2
        return zeroX, avgCenter, posCenter, negCenter, avgDiff

    zeroX, avgCenter, posCenter, negCenter, avgDiff = getZeroCrossing(diffAcc)

    f= plt.figure(figsize=(9, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    ax1 = f.add_subplot(gs[1])
    plt.text(0.97, 0.85, 'b)', transform=ax1.transAxes)
    plt.xlabel("delay [ps]")
    plt.ylabel("Diff. signal intensity [au]")
    plt.plot(delays, avgDiff)

    '''def stepCorr(a):
        cuma = np.cumsum(a)
        suma = np.sum(a)
        return suma - cuma

    xcorr = stepCorr(avgDiff - avgDiff.mean())
    plt.plot(delays,xcorr)'''

    startIdx = np.abs(avgDiff - 3).argmin()

    ax1 = f.add_subplot(gs[0], sharex=ax1)
    plt.text(0.97, 0.95, 'a)', transform=ax1.transAxes)
    ax1.xaxis.set_minor_locator(tck.AutoMinorLocator())
    plt.xlabel("delay [ps]")
    plt.ylabel("Kinetic energy [eV]")
    plt.plot(delays[startIdx:], zeroX[startIdx:], '+', label='zero crossing', color='C0')
    plt.plot(delays[startIdx:], avgCenter[startIdx:], 'x', label='differential signal centroid', color='C3')

    #plt.plot(delays[startIdx:], posCenter[startIdx:], '+',  label='positive centroid', color='C1')
    #plt.plot(delays[startIdx:], negCenter[startIdx:], '+', label='negative centroid', color='C2')

    plt.plot(delays[startIdx+1:-1], utils.movAvg(zeroX[startIdx:],3), '-', color='C0')
    plt.plot(delays[startIdx+1:-1], utils.movAvg(avgCenter[startIdx:],3), '-', color='C3')
    #plt.plot(delays[startIdx+1:-1], movAvg(posCenter[startIdx:],3), '-', color='C1')
    #plt.plot(delays[startIdx+1:-1], movAvg(negCenter[startIdx:],3), '-', color='C2')

    #If bootstrap data is present, use to to estimate errorbars
    try:
        raise NameError
        #avgCenter for bootstrap samples
        results = [ getZeroCrossing(sample) for sample in bsDiffAcc]
        print(f"{len(results)} bootstrap samples found, plotting errorbars")
        zeroXs  = np.array([ res[0] for res in results]).T
        centers = np.array([ res[1] for res in results]).T

        bsXVar = zeroXs.var(axis=1)  #Variance of bootstrap samples
        bsCVar = centers.var(axis=1)
        #bcCMea = centers.mean(axis=1)
        #plt.plot(delays, bcCMea, label='center ADSF', color='C7')
        print(f"Total bs variance {bsCVar.sum():.1f}  {bsXVar.sum():.1f}")

        plt.fill_between(delays, zeroX-bsXVar,     zeroX+bsXVar,     facecolor='C0', alpha=0.15)
        plt.fill_between(delays, avgCenter-bsCVar, avgCenter+bsCVar, facecolor='C3', alpha=0.15)
        plt.gca().set_ylim([136,142])
        #plt.gca().set_xlim([-0.4,3.2])
    except NameError:
        pass

    plt.legend()

if cfg.plots.auger2d:
    ROI = slice(np.abs(evs - cfg.augerROI[1]).argmin() , np.abs(evs - cfg.augerROI[0]).argmin())

    f = plt.figure(figsize=(11, 7))
    if cfg.plots.augerIntensity:
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax1 = f.add_subplot(gs[1])
        plt.text(0.1, 0.95, 'b)', transform=ax1.transAxes)

        #Normalize and shift intensity
        integ = diffAcc[:,ROI].sum(axis=1)
        integ -= integ[:2].mean()
        #integ /= np.linalg.norm(integ)

        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.tick_params(axis='y', labelsize=0, length = 0)

        plt.plot(integ,delays, 'x', color='C0')
        plt.plot(utils.movAvg(integ,3), delays[1:-1], '-', color='C0')

        plt.xlabel(f"Integrated differential\nintensity[a.u.]")
        f.add_subplot(gs[0], sharey=ax1)
        f.subplots_adjust(left=0.08, bottom=0.1, right=0.96, top=0.95, wspace=None, hspace=0.03)
        plt.text(0.05, 0.95, 'a)', transform=plt.gca().transAxes)

    #plt.suptitle("Auger Kinetic Energy vs Delay. Photon Energy 270 eV ")
    cmax = np.percentile(np.abs(diffAcc[:,ROI]),99.5)

    if cfg.plots.auger2d == "CONTOUR":
        plt.contourf(evs[ROI], delays, diffAcc[:,ROI],
                  cmap='bwr', vmax=cmax, vmin=-cmax, levels=100)
    else:
        plt.pcolormesh(evs[ROI], delays, diffAcc[:,ROI],
                   cmap='bwr', vmax=cmax, vmin=-cmax)

    plt.colorbar()

    plt.xlabel("Kinetic energy [eV]")
    plt.ylabel("Delay [ps]")

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


if cfg.plots.timeZero:
    #Calulates cross correlation between a and step function, seems to work
    def stepCorr(a):
        cuma = np.cumsum(a)
        suma = np.sum(a)
        return suma - cuma

    ROI = slice(np.abs(evs - cfg.plots.timeZero[1]).argmin() , np.abs(evs - cfg.plots.timeZero[0]).argmin())
    integ = diffAcc[:, ROI].sum(axis=1)
    integ-= integ.mean()

    xcorr = stepCorr(integ)

    plt.figure()
    plt.plot(delays,integ)
    plt.plot(delays,xcorr)

plt.show()
