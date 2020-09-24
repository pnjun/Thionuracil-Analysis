#!/usr/bin/python3
import pandas as pd
import numpy as np
from datetime import datetime
from attrdict import AttrDict
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
import utils

import pickle

cfg = {    'data'     : { 'path'     : '/media/Fast2/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'third_block.h5'
                        },
           'output'   : { 'path'     : './data/',
                          'fname'    : 'MULTIPLOT_2s_pPol'
                        },
           'time'     : { 'start' : datetime(2019,4,5,1,50,0).timestamp(),
                          'stop'  : datetime(2019,4,5,8,11,0).timestamp(),
                        },
           'adHocFilter' : True, #A couple of ad-hoc filters
           'sPolFilter'  : False,
           'filters'  : { 'undulatorEV' : (215.,230),
                          'retarder'    : (-11,-9),
                          #'delay'       : (1170, 1185.0),
                          'waveplate'   : (12,14)
                        },

           # filter for shotsdata parameters used in query method
           'sdfilter' : "uvPow > 20 & GMD > 0.5 & BAM != 0",
           'delayBin_mode'  : 'QUANTILE', # Binning mode, must be one of CUSTOM, QUANTILE, CONSTANT
           'delayBinStep'   : 0.2,     # Size of bins, only relevant when delayBin_mode is CONSTANT
           'delayBinNum'    : 6,      # Number if bis to use, only relevant when delayBin_mode is QUANTILE
           'ioChunkSize' : 50000,
           'gmdNormalize': True,
           'useBAM'      : True,
           'timeZero'    : 1260.3,     #Used to correct delays

           #either MULTIPLOT for one nexafs 2d plot per dealay
           #or INTEGRAL for one single 2dplot with integrated data over ROI
           'mode'        : 'MULTIPLOT',
           'integROIeV'  : (25,80),
           'decimate'    : False, #Decimate macrobunches before analizing. Use for quick evalutation of large datasets

           'plots' : {   'rescaleDiffs': False, #SET to reset the 0-difference point to the UV late signal average

                         'trnexafs'    : True,
                         'traceDelay'  : 1,    #[ps] Plots the NEXAFS traces for negative (earliest) vs positive delay (specified delay)
                         'diffHist'    : False
           },
           'writeOutput' : True, #Set to true to write out data in csv
           'onlyplot'    : False, #Set to true to load data form 'output' file and
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

    #A couple of ad-hoc filters for when standard filtering is not enough
    if cfg.adHocFilter:
        #There is a botched timeZero scan with low uvPow setting that we need to take out
        #from TRNEXAFS 2S data( 5.4.19 1:50 TO 8:11 )
        fpulses = fpulses.query('undulatorEV < 221.99 or undulatorEV > 222.01')
    if cfg.sPolFilter:
        #There is a chunk of s polarization data in the middle of p polarization scan
        #That we need to take out the hard way
        startTime=datetime(2019,4,6,5,28,0).timestamp()
        stopTime =datetime(2019,4,6,13,28,0).timestamp(),
        fpulses = fpulses.query('time < @startTime or time > @stopTime')


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
    assert not pulses.opisEV.isnull().any(), "Some opisEV values are NaN"

    #Plot relevant parameters as sanity check
    utils.plotParams(shotsData)

    uvEven = shotsData.query("shotNum % 2 == 0").uvPow.mean()
    uvOdd  = shotsData.query("shotNum % 2 == 1").uvPow.mean()

    if uvEven > uvOdd:
        print("Even shots are UV pumped.")
    else:
        print("Odd shots are UV pumped.")

    if cfg.gmdNormalize:
        gmdData = shotsData.GMD
    else:
        gmdData = None

    #Add Bam info    eStep = energy[1] - energy[0]
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

    #Energy binning
    energyBins = pulses.groupby( 'undulatorEV' )

    #chose binning dependent on delaybin_mode
    if cfg.delayBin_mode == 'CUSTOM': # insert your binning intervals here
        print(f"Setting up customized bins")
        interval = pd.IntervalIndex.from_arrays(utils.CUSTOM_BINS_LEFT - averageBamShift,
                                                utils.CUSTOM_BINS_RIGHT - averageBamShift)
        delayBins = shotsData.groupby( pd.cut(shotsData.delay, interval) )

    else:
    	#choose from a plot generated
        binStart, binEnd = utils.getROI(shotsData, limits=(-3,20))
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

    delays = np.array( [name.mid for name, _ in delayBins] )
    energy = np.array( [name for name, _ in energyBins] )

    #Read in TOF data and calulate difference, in chunks
    shotsTof  = utils.h5load('shotsTof', tr, pulses, chunk=cfg.ioChunkSize)

    #Create empty ouptut image image
    if cfg.mode == 'INTEGRAL':
        outShape = ( len(energyBins),  len(delayBins) )
    elif cfg.mode == 'MULTIPLOT':
        outShape = ( len(energyBins),  len(delayBins) , 3009)
    else:
        raise ValueError("mode not valid")

    binCount = np.zeros( outShape )
    diffAcc  = np.zeros( outShape )
    if cfg.mode == 'INTEGRAL':
        pumpAcc  = np.zeros( outShape )

    evConv = utils.mainTofEvConv(pulses.retarder.mean())

    #Iterate over data chunks and accumulate them in img
    for counter, chunk in enumerate(shotsTof):
        print( f"loading chunk {counter} of {shotsCount//cfg.ioChunkSize}",
               end='\r' )

        if counter == 0:
            evs = evConv(chunk.columns)
            ROI = slice( np.abs(evs - cfg.integROIeV[1]).argmin() ,
                         np.abs(evs - cfg.integROIeV[0]).argmin() )

        #calculate difference spectra and integrate over ROITrue
        if cfg.mode == 'INTEGRAL':
            pumpData, diffData = utils.getInteg(chunk, ROI, gmd = gmdData, getDiff = True, jacobian_evs=evs)
        else:
            diffData = utils.getDiff(chunk, gmd = gmdData)

        #iterate over delay bins
        for delayIdx, delayBin in enumerate(delayBins):
            _, group = delayBin
            group = group.query(cfg.sdfilter)
            delayBinIdx   = diffData.index.intersection(group.index)
            diffDelayBinTrace = diffData.reindex(delayBinIdx)
            if cfg.mode == 'INTEGRAL':
                pumpDelayBinTrace = pumpData.reindex(delayBinIdx)

            #iterate over energy bins (note taht energy bins are just pulseIds)
            for energyIdx, energyBin in enumerate(energyBins):
                _, energyGroup = energyBin
                diffTrace = diffDelayBinTrace.query('pulseId in @energyGroup.index')
                # - sign is because signal is negative
                diffAcc[energyIdx, delayIdx] += -diffTrace.sum().to_numpy()
                if cfg.mode == 'INTEGRAL':
                    pumpTrace = pumpDelayBinTrace.query('pulseId in @energyGroup.index')
                    pumpAcc[energyIdx, delayIdx] += -pumpTrace.sum().to_numpy()
                binCount[energyIdx, delayIdx] += diffTrace.shape[0]

    idx.close()
    tr.close()

    diffAcc /= binCount

    if cfg.mode == 'MULTIPLOT':
        diffTrace = utils.jacobianCorrect(diffTrace, evs)

    if cfg.mode == 'INTEGRAL': pumpAcc /= binCount

    if cfg.writeOutput:
        if cfg.mode == 'INTEGRAL':
            np.savez(cfg.output.path + cfg.output.fname,
                     diffAcc = diffAcc, pumpAcc = pumpAcc, delays=delays, energy=energy)
        else:
            np.savez(cfg.output.path + cfg.output.fname,
                     diffAcc = diffAcc, evs=evs, delays=delays, energy=energy)

if cfg.onlyplot:
        dataZ = np.load(cfg.output.path + cfg.output.fname + ".npz")
        diffAcc = dataZ['diffAcc']
        delays = dataZ['delays']
        energy = dataZ['energy']
        # If the  data was saved with mode == INTEGRAL, we also have the pump array
        if 'pumpAcc' in dataZ.files:
            pumpAcc = dataZ['pumpAcc']
        # If mode is MULTIPLOT, we have the evs array
        else:
            evs = dataZ['evs']

if cfg.plots.rescaleDiffs:
    diffAcc -= diffAcc[:,:,:].mean()

if cfg.plots.trnexafs and 'pumpAcc' in globals(): #Plot style for INTEGRAL mode
    f = plt.figure(figsize=(9, 10))

    if cfg.plots.traceDelay:
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
        ax1 = f.add_subplot(gs[0])
        delayIdx = np.abs(cfg.plots.traceDelay - delays).argmin()

        #diff acc is created subtracting pump - unpumped. Therefore the umpumped is pump - diff
        plt.plot(energy, pumpAcc[:,delayIdx] - diffAcc[:,delayIdx], label="UV Off")
        plt.plot(energy, pumpAcc[:,delayIdx], label=f"UV On @ {delays[delayIdx]:.4}ps delay")
        plt.ylabel(f"Integrated Auger")
        plt.legend()
        f.add_subplot(gs[1], sharex=ax1)
        f.subplots_adjust(left=0.12, bottom=0.11, right=0.95, top=0.95, wspace=None, hspace=0.14)

    eStep = energy[1] - energy[0]
    yenergy = [e - eStep/2 for e in energy ] + [ energy[-1] + eStep/2 ]
    xdelays = [d for d in delays]  + [ delays[-1] + (delays[-1] - delays[-2]) ] #Assume last bin is a wide as second to last

    plt.suptitle(f"Integrated Auger Signal {cfg.integROIeV[0]}:{cfg.integROIeV[1]} eV")
    cmax = np.nanpercentile(np.abs(diffAcc),98)

    im = plt.pcolormesh(yenergy,xdelays, diffAcc.T,
                   cmap='bwr', vmax=cmax, vmin=-cmax)
    plt.xlabel("Undulator Energy (eV)")
    plt.ylabel("Delay (ps)")
    cbar_ax = f.add_axes([0.12, 0.04, 0.8, 0.02])
    plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    ticks = cbar_ax.xaxis.get_ticklabels()
    ticks[0] = 'pump depleted'
    ticks[-1] = 'pump enhanced'
    cbar_ax.set_xticklabels(ticks)


if cfg.plots.trnexafs and 'pumpAcc' not in globals(): #Plot style for MULTIPLOT mode
    plotGridX = int(np.ceil(np.sqrt(delays.size)))
    plotGridY = int(np.ceil( delays.size / plotGridX ))

    eStep = energy[1] - energy[0]
    yenergy = [e - eStep/2 for e in energy ] + [ energy[-1] + eStep/2 ]
    xevs = [e for e in evs]  + [ evs[-1] + (evs[-1] - evs[-2]) ]

    ROIslice = slice( np.abs(evs - cfg.integROIeV[1]).argmin() ,
                      np.abs(evs - cfg.integROIeV[0]).argmin() )

    f = plt.figure(figsize=(9, 14))
    gs = gridspec.GridSpec(plotGridX, plotGridY)

    cmax = np.nanpercentile(np.abs(diffAcc),99.8)
    for n in range(delays.size):
        ax = f.add_subplot(gs[n])
        ax.title.set_text(f'Delay {delays[n]:.3}')
        im = plt.pcolormesh(xevs[ROIslice],yenergy, diffAcc[:,n,ROIslice], cmap='bwr', vmax=cmax, vmin=-cmax)

    f.subplots_adjust(left=0.08, bottom=0.11, right=0.92, top=0.96)
    cbar_ax = f.add_axes([0.12, 0.04, 0.8, 0.02])
    plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    ticks = cbar_ax.xaxis.get_ticklabels()
    ticks[0] = 'pump depleted'
    ticks[-1] = 'pump enhanced'
    cbar_ax.set_xticklabels(ticks)

if cfg.plots.diffHist:
    plt.figure()
    plt.hist(diffAcc.flatten())

plt.show()
