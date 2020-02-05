#!/usr/bin/python3
import pandas as pd
import numpy as np
from datetime import datetime
from attrdict import AttrDict
import matplotlib.pyplot as plt
import sys
import utils

import pickle

cfg = {    'data'     : { 'path'     : '/media/Fast2/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'fisrt_block.h5'
                        },
           'output'   : { 'path'     : './data/',
                          'fname'    : 'DelayShort270ev'
                        },
           'time'     : { 'start' : datetime(2019,3,27,4,28,0).timestamp(),
                          'stop'  : datetime(2019,3,27,5,54,0).timestamp(),
                        },
           'filters'  : { 'undulatorEV' : (260.,280.),
                          'retarder'    : (-90,-70),
                          #'delay'       : (1170, 1185.0),
                          'waveplate'   : (9,11)
                        },
           'sdfilter' : "GMD > 0.5 & BAM != 0", # filter for shotsdata parameters used in query method
           'delayBin_mode'  : 'QUANTILE', # Binning mode, must be one of CUSTOM, QUANTILE, CONSTANT
           'delayBinStep'   : 0.2,     # Size of bins, only relevant when delayBin_mode is CONSTANT
           'delayBinNum'    : 15,     # Number if bis to use, only relevant when delayBin_mode is QUANTILE
           'ioChunkSize' : 50000,
           'gmdNormalize': True,
           'useBAM'      : True,
           'timeZero'    : 1178.4,   #Used to correct delays
           'decimate'    : False, #Decimate macrobunches before analizing. Use for quick evalutation of large datasets

           'plots' : {
                       'delay2d'    : True,
                       'photoShift' : False,
                       'valence'    : False,
                       'auger2d'    : True,
                       'fragmentSearch' : False, #Plot Auger trace at long delays to look for fragmentation
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

    #Plot relevant parameters
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
        bins = shotsData.groupby( pd.cut(shotsData.delay, interval) )

    else:
    	#choose from a plot generated
        binStart, binEnd = utils.getROI(shotsData, limits=(-5,20))
        print(f"Binning interval {binStart} : {binEnd}")

        #Bin data on delay
        if cfg.delayBin_mode == 'CONSTANT':
            bins = shotsData.groupby( pd.cut( shotsData.delay,
                                              np.arange(binStart, binEnd, cfg.delayBinStep) ) )
        elif cfg.delayBin_mode == 'QUANTILE':
       	    shotsData = shotsData[ (shotsData.delay > binStart) & (shotsData.delay < binEnd) ]
       	    bins = shotsData.groupby( pd.qcut( shotsData.delay, cfg.delayBinNum ) )
        else:
            raise Exception("binning mode not valid")

    #Read in TOF data and calulate difference, in chunks
    shotsTof  = utils.h5load('shotsTof', tr, pulses, chunk=cfg.ioChunkSize)

    #Create empty ouptut image image
    diffAcc = np.zeros( ( len(bins), 3009 ))
    binCount = np.zeros( len(bins) )

    if cfg.plots.fragmentSearch:
        evenAcc = np.zeros( ( len(bins), 3009 ))
        oddAcc  = np.zeros( ( len(bins), 3009 ))

    #Iterate over data chunks and accumulate them in diffAcc
    for counter, chunk in enumerate(shotsTof):
        print( f"loading chunk {counter} of {shotsCount//cfg.ioChunkSize}",
               end='\r' )
        shotsDiff = utils.getDiff(chunk, gmdData)
        for binId, bin in enumerate(bins):
            name, group = bin
            group = group.query(cfg.sdfilter)
            binIdx = shotsDiff.index.intersection(group.index)
            binTrace = shotsDiff.reindex(binIdx)

            if cfg.plots.fragmentSearch:
                even = chunk.query('shotNum % 2 == 0')
                odd =  chunk.query('shotNum % 1 == 0')
                odd.index = odd.index.set_levels(odd.index.levels[1] - 1, level=1)
                evenAcc[binId] += even.reindex(binIdx).sum(axis=0)
                oddAcc[binId]  += odd.reindex(binIdx).sum(axis=0)
            diffAcc[binId] += binTrace.sum(axis=0)
            binCount[binId] += binTrace.shape[0]

    idx.close()
    tr.close()

    diffAcc /= binCount[:,None]
    if cfg.plots.fragmentSearch:
        evenAcc /= binCount[:,None]
        oddAcc /= binCount[:,None]

    if uvEven > uvOdd: diffAcc *= -1

    #get axis labels
    delays = np.array( [name.mid for name, _ in bins] )
    #delays = np.array([bins["delay"].mean().values, bins["delay"].std().values])

    evConv = utils.mainTofEvConv(pulses.retarder.mean())
    evs = evConv(shotsDiff.iloc[0].index.to_numpy(dtype=np.float32))

    # make dataframe and save data
    if cfg.writeOutput:
        df = pd.DataFrame(data = diffAcc,columns=evs, index=delays).fillna(0)
        df.to_csv(cfg.output.path + cfg.output.fname + ".csv", mode="w")

if cfg.onlyplot:
    df = pd.read_csv(cfg.output.path + cfg.output.fname + ".csv", index_col=0)
    diffAcc = df.to_numpy()
    evs = df.columns.to_numpy(dtype=np.float32)
    delays = df.index.to_numpy()

#plot resulting image
if cfg.plots.delay2d:
    ROI = slice(np.abs(evs - 270).argmin() , None)
    plt.figure(figsize=(9, 7))
    plt.suptitle("Kinetic Energy vs Delay. Photon Energy 270eV")
    cmax = np.percentile(np.abs(diffAcc[:,ROI]),99.5)
    plt.pcolormesh(evs[ROI], delays, diffAcc[:,ROI],
                   cmap='bwr', vmax=cmax, vmin=-cmax)
    plt.xlabel("Kinetic energy (eV)")
    plt.ylabel("Delay (ps)")

if cfg.plots.auger2d:
    ROI = slice(np.abs(evs - 160).argmin() , np.abs(evs - 120).argmin())
    plt.figure(figsize=(9, 7))
    plt.suptitle("Auger Kinetic Energy vs Delay. Photon Energy 270eV")
    cmax = np.percentile(np.abs(diffAcc[:,ROI]),99.5)
    plt.pcolormesh(evs[ROI], delays, diffAcc[:,ROI],
                   cmap='bwr', vmax=cmax, vmin=-cmax)
    plt.xlabel("Kinetic energy (eV)")
    plt.ylabel("Delay (ps)")

if cfg.plots.photoShift:
    # quick plots for check. for more detailed analysis use photolineAnalysis.py
    # plot line graph of integral over photoline features
    plt.figure()
    photoline1 = slice(np.abs(evs - 101.5).argmin() , np.abs(evs - 95.5).argmin())
    photoline2 = slice(np.abs(evs - 107).argmin() , np.abs(evs - 101.5).argmin())

    NegPhLine, = plt.plot(delays, abs(diffAcc.T[photoline2].sum(axis=0)), label = 'negative photoline shift')
    PosPhLine, = plt.plot(delays, abs(diffAcc.T[photoline1].sum(axis=0)), label = 'positive photoline shift')
    plt.xlabel("Delay (ps)")
    plt.ylabel("Integrated signal")
    plt.legend(handles=[PosPhLine, NegPhLine])
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



if cfg.plots.fragmentSearch and not cfg.onlyplot:
    from matplotlib.widgets import Slider
    f = plt.figure()

    diffPlot, = plt.plot(evs,diffAcc[-1], label='diffs')
    evenPlot, = plt.plot(evs,evenAcc[-1], label='even')
    oddPlot,  = plt.plot(evs,oddAcc[-1], label='odd')
    #plt.plot(evs,diffAcc.values[0])
    f.suptitle(delays[-1])
    def update(val):
        n = int(val)
        diffPlot.set_ydata(diffAcc[n])
        evenPlot.set_ydata(evenAcc[n])
        oddPlot.set_ydata(oddAcc[n])

        f.suptitle(delays[n])
        f.canvas.draw_idle()

    slax = plt.axes([0.1, 0.03, 0.65, 0.04])
    slider = Slider(slax, 'Amp', 0, diffAcc.shape[0]-1, valinit=0)
    slider.on_changed(update)

plt.show()
