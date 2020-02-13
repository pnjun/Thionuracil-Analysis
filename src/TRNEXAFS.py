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
                          'fname'    : 'TrNexafs2p'
                        },
           'time'     : { 'start' : datetime(2019,4,5,20,59,0).timestamp(),
                          'stop'  : datetime(2019,4,6,5,13,0).timestamp(),
                        },
           'filters'  : { 'undulatorEV' : (160.,173),
                          'retarder'    : (-11,-9),
                          #'delay'       : (1170, 1185.0),
                          'waveplate'   : (12,14)
                        },
           'sdfilter' : "GMD > 3.5 & BAM != 0", # filter for shotsdata parameters used in query method
           'delayBin_mode'  : 'QUANTILE', # Binning mode, must be one of CUSTOM, QUANTILE, CONSTANT
           'delayBinStep'   : 0.2,     # Size of bins, only relevant when delayBin_mode is CONSTANT
           'delayBinNum'    : 10,     # Number if bis to use, only relevant when delayBin_mode is QUANTILE
           'ioChunkSize' : 50000,
           'gmdNormalize': True,
           'useBAM'      : True,
           'timeZero'    : 1261.7,   #Used to correct delays
           'integROIeV'  : (100,170),
           'decimate'    : False, #Decimate macrobunches before analizinintegROIeVg. Use for quick evalutation of large datasets

           'plots' : {   'trnexafs'    : True,
                         'traceDelay' : 1,     # Plots the NEXAFS traces for negative (earliest) vs positive delay (specified delay)
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
        binStart, binEnd = utils.getROI(shotsData, limits=(-5,20))
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

    energyBins = pulses.groupby( 'undulatorEV' )

    delays = np.array( [name.mid for name, _ in delayBins] )
    energy = np.array( [name for name, _ in energyBins] )

    #Read in TOF data and calulate difference, in chunks
    shotsTof  = utils.h5load('shotsTof', tr, pulses, chunk=cfg.ioChunkSize)

    #Create empty ouptut image image
    diffAcc  = np.zeros( ( len(energyBins),  len(delayBins)))
    pumpAcc  = np.zeros( ( len(energyBins),  len(delayBins)))
    binCount = np.zeros( ( len(energyBins),  len(delayBins)))

    evConv = utils.mainTofEvConv(pulses.retarder.mean())

    #Iterate over data chunks and accumulate them in img
    for counter, chunk in enumerate(shotsTof):
        print( f"loading chunk {counter} of {shotsCount//cfg.ioChunkSize}",
               end='\r' )

        if counter == 0:
            evs = evConv(chunk.columns)
            ROI = slice( np.abs(evs - cfg.integROIeV[1]).argmin() ,
                         np.abs(evs - cfg.integROIeV[0]).argmin() )

        #calculate difference spectra and integrate over photoline
        shotsInteg, integDiff = utils.getInteg(chunk, ROI, gmd = gmdData, getDiff = True)

        #iterate over delay bins
        for delayIdx, delayBin in enumerate(delayBins):
            _, group = delayBin
            group = group.query(cfg.sdfilter)
            delayBinIdx   = integDiff.index.intersection(group.index)
            diffDelayBinTrace = integDiff.reindex(delayBinIdx)
            pumpDelayBinTrace = shotsInteg.reindex(delayBinIdx)

            #iterate over energy bins (note taht energy bins are just pulseIds)
            for energyIdx, energyBin in enumerate(energyBins):
                _, energyGroup = energyBin
                diffTrace = diffDelayBinTrace.query('pulseId in @energyGroup.index')
                pumpTrace = pumpDelayBinTrace.query('pulseId in @energyGroup.index')
                pumpAcc[energyIdx, delayIdx] += -pumpTrace.sum().to_numpy() # - sign is because signal is negative
                diffAcc[energyIdx, delayIdx] += -diffTrace.sum().to_numpy()
                binCount[energyIdx, delayIdx] += diffTrace.shape[0]

    idx.close()
    tr.close()

    diffAcc /= binCount
    pumpAcc /= binCount
    if uvOdd > uvEven: diffAcc *= -1

    if cfg.writeOutput:
        df = pd.DataFrame(data = diffAcc,columns=delays, index=energy).fillna(0)
        df.to_csv(cfg.output.path + cfg.output.fname + "_diff.csv", mode="w")
        df = pd.DataFrame(data = pumpAcc,columns=delays, index=energy).fillna(0)
        df.to_csv(cfg.output.path + cfg.output.fname + "_pump.csv", mode="w")

if cfg.onlyplot:
    df = pd.read_csv(cfg.output.path + cfg.output.fname + "_diff.csv", index_col=0)
    diffAcc = df.to_numpy()
    df = pd.read_csv(cfg.output.path + cfg.output.fname + "_pump.csv", index_col=0)
    pumpAcc = df.to_numpy()
    delays = df.columns.to_numpy(dtype=np.float32)
    energy = df.index.to_numpy()

if cfg.plots.trnexafs:
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
        f.subplots_adjust(left=0.12, bottom=0.05, right=0.95, top=0.95, wspace=None, hspace=0.14)

    eStep = energy[1] - energy[0]
    yenergy = [e - eStep/2 for e in energy ] + [ energy[-1] + eStep/2 ]
    xdelays = [d for d in delays]  + [ delays[-1] + (delays[-1] - delays[-2]) ] #Assume last bin is a wide as second to last

    plt.suptitle(f"Integrated Auger Signal {cfg.integROIeV[0]}:{cfg.integROIeV[1]} eV")
    cmax = np.percentile(np.abs(diffAcc),99.5)

    plt.pcolormesh(yenergy,xdelays, diffAcc.T,
                   cmap='bwr', vmax=cmax, vmin=-cmax)
    plt.xlabel("Undulator Energy (eV)")
    plt.ylabel("Delay (ps)")
    plt.colorbar(orientation="horizontal", fraction = 0.05 ,pad=0.1)

plt.show()
