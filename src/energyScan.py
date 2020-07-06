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
                          'fname'    : 'energyScan_2p_thirdBlock49_noJac'
                        },
           'time'     : { 'start' : datetime(2019,4,5,17,26,0).timestamp(),
                          'stop'  : datetime(2019,4,6,5,13,0).timestamp(),
                        },
           'filters'  : { 'retarder'    : (-15,5),
                          #'waveplate'  : (10,15),
                          #'delay'      : (1255.0, 1255.5),
                          'undulatorEV': (150,180)
                        },
           'sdfilter' : "GMD > 2.5", # filter for shotsdata parameters used in query method

           'ioChunkSize' : 50000,
           'decimate'    : False, #Decimate macrobunches before analizing. Use for quick evalutation of large datasets
           'gmdNormalize': False,


           'onlyOdd'     : True, #Set to true if ony odd shots should be used, otherwise all shots are used
           'difference'  : False, #Set to true if a even-odd difference spectrum shuold be calculated instead (onlyodd is ignored in this case)
           'timeZero'    : 1257.2,   #Used to correct delays

           'normAndJac'  : True, #Set to true to use normalize traces and apply Jacobian Correction
           'plots' : {
                       'energy2d'      : (30, 190),
                       'ROIIntegral'   : (30, 190),
                       'plotSlice'     : [3,-1],
                            'Katritzky': False,
                       'uvtext'        : ""#"(P pol, High Uv, 2ps delay)"

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
    gmdData = shotsData.GMD

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
        elif cfg.gmdNormalize:
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

    traceAcc /= binCount[:,None]

    #get axis labels
    energy = np.array( [name for name, _ in bins] )

    evConv = utils.mainTofEvConv(pulses.retarder.mean())
    evs = evConv(chunk.columns)

    # make dataframe and save data
    if cfg.writeOutput:
        np.savez(cfg.output.path + cfg.output.fname,
                 traceAcc = traceAcc, energy = energy, evs=evs)

if cfg.onlyplot:
    print("Reading data...")
    dataZ = np.load(cfg.output.path + cfg.output.fname + ".npz")
    traceAcc = dataZ['traceAcc']
    energy   = dataZ['energy']
    evs      = dataZ['evs']

'''
if '_noJacobian' in cfg.output.fname:
    print('correcting jacobian')
    traceAcc   = utils.jacobianCorrect(traceAcc, evs)
    np.savez(cfg.output.path + cfg.output.fname[:-11],
                 traceAcc = traceAcc, evs=evs, energy = energy)
'''

if cfg.normAndJac:
    traceAcc -= traceAcc.min(axis=1)[:,None]

    NormROI   = slice( None , np.abs(evs - 270).argmin() )
    traceAcc /= traceAcc[:,NormROI].mean(axis=1)[:,None]

    traceAcc = utils.jacobianCorrect(traceAcc, evs)

if cfg.plots.plotSlice:
    ROI = slice(np.abs(evs - cfg.plots.energy2d[1]).argmin() ,
                np.abs(evs - cfg.plots.energy2d[0]).argmin() )
    f = plt.figure(figsize=(9, 3))

    plt.plot(evs[ROI], traceAcc[cfg.plots.plotSlice[0],ROI], label=f'{energy[cfg.plots.plotSlice[0]]:.2f} eV')
    if cfg.plots.Katritzky:
        kData = np.loadtxt('Katritzky_1990.csv', delimiter=',')
        plt.plot(kData[:,0]+(energy[cfg.plots.plotSlice[0]]-21.21),kData[:,1]*30)


    plt.plot(evs[ROI], traceAcc[cfg.plots.plotSlice[1],ROI], label=f'{energy[cfg.plots.plotSlice[1]]:.2f} eV')
    plt.xlabel("Kinetic energy (eV)")
    plt.ylabel(f"Intensity [a.u.]")
    f.subplots_adjust(left=0.08, bottom=0.15, right=0.96, top=0.95)
    plt.legend()

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
        plt.text(0.915, 0.97, 'b)', transform=plt.gca().transAxes)
        integROI = slice( np.abs(evs - cfg.plots.ROIIntegral[1]).argmin() ,
                          np.abs(evs - cfg.plots.ROIIntegral[0]).argmin() )
        integ = traceAcc[:,integROI].sum(axis=1)
        integ -= integ[:2].mean()
        integ /= np.linalg.norm(integ)

        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.tick_params(axis='y', labelsize=0, length = 0)

        plt.plot(integ, energy)
        if cfg.difference:
            plt.xlabel("Integrated Differential Intensity")
        else:
            plt.xlabel("Integrated Intensity")

        f.add_subplot(gs[0], sharey=ax1)
        plt.text(0.97, 0.97, 'a)', transform=plt.gca().transAxes)
        f.subplots_adjust(left=0.08, bottom=0.07, right=0.96, top=0.95, wspace=0.05, hspace=None)

    if cfg.difference:
        plt.suptitle(f"Kinetic Energy vs Photon Energy {cfg.plots.uvtext}")
        cmax = np.percentile(np.abs(traceAcc),99.5)
        plt.pcolormesh(evs[ROI], yenergy, traceAcc[:,ROI],
                       cmap='bwr', vmax=cmax, vmin=-cmax)
    else:
        #cmax = np.percentile(np.abs(traceAcc[:,ROI]),99)
        #cmin = np.percentile(np.abs(traceAcc[:,ROI]),1)
        plt.pcolormesh(evs[ROI], yenergy, traceAcc[:,ROI],
                        cmap='bone_r')#, vmax=cmax, vmin=cmin)

    plt.xlabel("Kinetic energy (eV)")
    plt.ylabel("Photon energy (eV)")

    if cfg.difference:
        cb = plt.colorbar( orientation="vertical",fraction=0.06,anchor=(1.0,0.0) )

        ticks = cb.ax.yaxis.get_ticklabels()
        ticks[0] = 'pump\ndepleted'
        ticks[-1] = 'pump\nenhanced'
        cb.ax.set_yticklabels(ticks)

plt.show()
