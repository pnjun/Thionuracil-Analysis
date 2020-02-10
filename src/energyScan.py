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
                          'fname'    : 'energyScanDifferenceSpol'
                        },
           'time'     : { 'start' : datetime(2019,4,6,16,22,0).timestamp(),
                          'stop'  : datetime(2019,4,6,18,50,0).timestamp(),
                        },
           'filters'  : {'retarder'    : (-15,5),
                          'waveplate'   : (12,14),
                          #'delay'       : (1259.3, 1260.3)
                          'delay'       : (1261.1, 1261.3)
                        },
           'sdfilter' : "GMD > 0.5", # filter for shotsdata parameters used in query method

           'ioChunkSize' : 50000,
           'gmdNormalize': True,
           'decimate'    : False, #Decimate macrobunches before analizing. Use for quick evalutation of large datasets

           'onlyOdd'     : False, #Set to true if ony odd shots should be used, otherwise all shots are used
           'difference'  : True, #Set to true if a even-odd difference spectrum shuold be calculated instead (onlyodd is ignored in this case)
           'timeZero'    : 1261.7,   #Used to correct delays

           'plots' : {
                       'energy2d'      : (0, 250),
                       'ROIIntegral'   : (100, 200),
                       'uvtext'        : ""
           },
           'writeOutput' : True, #Set to true to write out data in csv
           'onlyplot'    : True, #Set to true to load data form 'output' file and
                                 #plot only.

      }

cfg = AttrDict(cfg)

if not cfg.onlyplot:
    idx = pd.HDFStore(cfg.data.path + cfg.data.index, mode = 'r')
    tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, mode = 'r')

    #Get all pulses within time limitsgroupby1
    pulses = idx.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')
    #Filter only pulses with parameters in range
    fpulses = utils.filterPulses(pulses, cfg.filters)

    #Get corresponing shots
    if(not len(fpulses)):
        print(f"Avg Retarder  {pulses.retarder.mean()}")
        print(f"Avg Undulator {pulses.undulatorEV.mean()}")
        print(f"Avg Waveplate {pulses.waveplate.mean()}")
        print(f"Avg Delay     {pulses.delay.mean()}")
        raise Exception("No pulses satisfy filter condition")
    else:
        pulses = fpulses

    if cfg.decimate:
        print("Decimating...")
        pulses = pulses.query('index % 10 == 0')

    shotsData = utils.h5load('shotsData', tr, pulses)
    shotsData = shotsData.query(cfg.sdfilter)
    shotsCount = shotsData.shape[0]
    print(f"Loading {shotsData.shape[0]} shots")

    if cfg.difference:
        #pulses.delay.hist()
        #plt.show()
        delayAvg = cfg.timeZero - pulses.delay.mean()
        print(f"Average Delay {delayAvg:.4}ps")

    #Remove pulses with no corresponing shots
    pulses = pulses.drop( pulses.index.difference(shotsData.index.levels[0]) )

    #Plot relevant parameters
    utils.plotParams(shotsData)

    if cfg.gmdNormalize and not cfg.difference:
        raise NotImplementedError("DUH")
    gmdData = shotsData.GMD if cfg.gmdNormalize else None

    #Bin data
    bins = pulses.groupby( 'undulatorEV' )
    #Read in TOF data
    shotsTof  = utils.h5load('shotsTof', tr, pulses, chunk=cfg.ioChunkSize)

    #Create empty ouptut image image
    traceAcc = np.zeros( ( len(bins), 3009 ))
    binCount = np.zeros( len(bins) )

    if cfg.onlyOdd and not cfg.difference:
        shotsData = shotsData.query('shotNum % 2 == 1')

    #Iterate over data chunks and accumulate them in diffAcc
    for counter, chunk in enumerate(shotsTof):
        print( f"loading chunk {counter} of {shotsCount//cfg.ioChunkSize}",
               end='\r' )

        if cfg.difference:
            chunk = utils.getDiff(chunk, gmdData)

        chunkIdx = shotsData.index.intersection(chunk.index)
        chunk = chunk.reindex(chunkIdx)

        for binId, bin in enumerate(bins):
            name, group = bin
            binTrace = chunk.query('pulseId in @group.index')

            traceAcc[binId] += -binTrace.sum(axis=0) # - sign is because signal is negative
            binCount[binId] += binTrace.shape[0]

    idx.close()
    tr.close()

    traceAcc /= binCount[:,None]

    #get axis labels
    energy = np.array( [name for name, _ in bins] )

    evConv = utils.mainTofEvConv(pulses.retarder.mean())
    evs = evConv(chunk.columns)

    # make dataframe and save data
    if cfg.writeOutput:
        df = pd.DataFrame(data = traceAcc, columns=evs, index=energy).fillna(0)
        df.to_csv(cfg.output.path + cfg.output.fname + ".csv", mode="w")

if cfg.onlyplot:
    df = pd.read_csv(cfg.output.path + cfg.output.fname + ".csv", index_col=0)
    traceAcc = df.to_numpy()
    evs = df.columns.to_numpy(dtype=np.float32)
    energy = df.index.to_numpy()

#plot resulting image
if cfg.plots.energy2d:
    ROI = slice(np.abs(evs - cfg.plots.energy2d[1]).argmin() ,
                np.abs(evs - cfg.plots.energy2d[0]).argmin() )
    eStep = energy[1] - energy[0]
    yenergy = [e - eStep/2 for e in energy ] + [ energy[-1] + eStep/2 ]

    f = plt.figure(figsize=(12, 8))

    if cfg.plots.ROIIntegral:
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax1 = f.add_subplot(gs[1])

        integROI = slice( np.abs(evs - cfg.plots.ROIIntegral[1]).argmin() ,
                          np.abs(evs - cfg.plots.ROIIntegral[0]).argmin() )
        integ = traceAcc[:,integROI].sum(axis=1)
        integ -= integ[:2].mean()
        integ /= np.linalg.norm(integ)

        plt.plot(integ, energy)
        if cfg.difference:
            plt.xlabel("Integrated Differential Intensity")
        else:
            plt.xlabel("Integrated Intensity")

        f.add_subplot(gs[0], sharey=ax1)
        f.subplots_adjust(left=0.08, bottom=0.07, right=0.96, top=0.95, wspace=None, hspace=0.05)

    if cfg.difference:
        plt.suptitle(f"Kinetic Energy vs Photon  Energy {cfg.plots.uvtext}")
        cmax = np.percentile(np.abs(traceAcc),99.5)
        plt.pcolormesh(evs[ROI], yenergy, traceAcc[:,ROI],
                       cmap='bwr', vmax=cmax, vmin=-cmax)
    else:
        plt.suptitle("Kinetic Energy vs Photon  Energy.")
        cmax = np.percentile(np.abs(traceAcc[:,ROI]),91)
        cmin = np.percentile(np.abs(traceAcc[:,ROI]),12)
        plt.pcolormesh(evs[ROI], yenergy, traceAcc[:,ROI],
                        cmap='bone_r', vmax=cmax, vmin=cmin)

    plt.xlabel("Kinetic energy (eV)")
    plt.ylabel("Photon energy (eV)")

    if cfg.difference:
        cb = plt.colorbar()
        ticks = cb.ax.yaxis.get_ticklabels()
        ticks[0] = 'pump depleted'
        ticks[-1] = 'pump enhanced'
        cb.ax.set_yticklabels(ticks)

plt.show()
