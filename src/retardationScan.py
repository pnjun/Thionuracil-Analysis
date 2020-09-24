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
                          'trace'    : 'fisrt_block.h5'
                        },
           'output'   : { 'path'     : './data/',
                          'fname'    : 'retardationScan'
                        },
           'time'     : { 'start' : datetime(2019,3,25,15,14,0).timestamp(),
                          'stop'  : datetime(2019,3,25,15,18,0).timestamp(),
                        },
           'filters'  : {
                          'waveplate'   : (5,10),
                          'undulatorEV' : (260, 275),
                        },

           'ioChunkSize' : 50000,
           'decimate'    : False, #Decimate macrobunches before analizing. Use for quick evalutation of large datasets
           'gmdNormalize': True,

           'plots' : {
                         'waterfall' : True
           },
           'writeOutput' : True,  #Set to true to write out data in csv
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
    shotsCount = shotsData.shape[0]
    print(f"Loading {shotsData.shape[0]} shots")

    #Remove pulses with no corresponing shots
    pulses = pulses.drop( pulses.index.difference(shotsData.index.levels[0]) )

    bins = pulses.groupby( 'retarder' )
    retarder = np.array( [name for name, _ in bins] )

    gmdData = shotsData.GMD

    #Read in TOF data
    shotsTof  = utils.h5load('shotsTof', tr, pulses, chunk=cfg.ioChunkSize)

    #Create empty ouptut image image
    traceAcc = np.zeros( ( len(bins), 3009 ))
    binCount = np.zeros( len(bins) )

    #Iterate over data chunks and accumulate them in diffAcc
    for counter, chunk in enumerate(shotsTof):
        print( f"loading chunk {counter} of {shotsCount//cfg.ioChunkSize}",
               end='\r' )

        chunk = utils.gmdCorrect(chunk, gmdData)
        chunkIdx = shotsData.index.intersection(chunk.index)
        chunk = chunk.reindex(chunkIdx)

        for binId, bin in enumerate(bins):
            name, group = bin
            binTrace = chunk.query('pulseId in @group.index')

            traceAcc[binId] += -binTrace.sum(axis=0)
            binCount[binId] += binTrace.shape[0]

    idx.close()
    tr.close()
    print()
    traceAcc /= binCount[:,None]

    tofs = np.array(chunk.columns.to_numpy())

    # make dataframe and save data
    if cfg.writeOutput:
        np.savez(cfg.output.path + cfg.output.fname,
                 traceAcc = traceAcc, retarder = retarder, tofs=tofs)

if cfg.onlyplot:
    print("Reading data...")
    dataZ = np.load(cfg.output.path + cfg.output.fname + ".npz", allow_pickle=True)
    traceAcc = dataZ['traceAcc']
    retarder   = dataZ['retarder']
    tofs     = dataZ['tofs']

if cfg.plots.waterfall:
    for n, trace in enumerate(traceAcc):
        evConv = utils.mainTofEvConv(retarder[n])
        evs  = evConv(tofs)

        #Offset Correction
        ROI = slice( 0, np.abs(evs - 270).argmin() )
        trace -= trace[ROI].mean()
        #Jacobian
        trace = utils.jacobianCorrect(trace, evs)

        plt.plot(evs, trace + 0.1*n)

plt.show()
