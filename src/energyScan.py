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
                          'trace'    : 'second_block.h5'
                        },
           'output'   : { 'path'     : './data/',
                          'fname'    : 'energyScan_2s_9-11_GMD_TOF_GMDBOUND_OPIS'
                          #'fname'    : 'energyScan_2p_thirdBlock49_UNDUL_noGMD'
                        },
           'time'     : { 'start' : datetime(2019,3,30,21,57,0).timestamp(),
                          'stop'  : datetime(2019,3,31,1,4,0).timestamp(),

                          #'start' : datetime(2019,4,5,17,26,0).timestamp(),
                          #'stop'  : datetime(2019,4,5,18,56,0).timestamp(),
                        },
           'filters'  : { 'retarder'    : (-2,2), #(-7,-3), (-2,2)
                          'waveplate'   : (5,16),
                          #'delay'      : (1255.0, 1255.5),
                          'undulatorEV': (203,245), #(150,180) (205,240)
                        },
           'sdfilter' : "GMD > 2.5 & GMD < 7.5", # filter for shotsdata parameters used in query method

           'ioChunkSize' : 50000,
           'decimate'    : False, #Decimate macrobunches before analizing. Use for quick evalutation of large datasets
           'gmdNormalize': True,
           'OPISbins'    : True,    #Use opis for energy bins labels

           'onlyOdd'     : False,   #Set to true if ony odd shots should be used, otherwise all shots are used
           'difference'  : False,  #Set to true if a even-odd difference spectrum shuold be calculated instead (onlyodd is ignored in this case)
           'timeZero'    : 1257.2, #Used to correct delays

           'Jacobian'    : True,   # apply Jacobian Correction
           'NormROI'     : None,   # Range over which to normalize the traces
           'plots' : {
                       'energy2d'      : (20, 250),
                          'inset'      : ((30,70),(219,232)), #Extent of zoomed in inset (ekin, ephoton). None for no inset
                       'ROIIntegral'   : (30, 81),
                       'plotSlice'     : [1,-1],
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

    #Bin data
    if cfg.OPISbins:
        #Get rid of pulses where OPIS was not working
        pulses = pulses.query( '(undulatorEV - opisEV < 6) and (undulatorEV - opisEV > -6)' )

       	#bins = pulses.groupby( pd.qcut( pulses.opisEV, cfg.OPISbins ) )
        bins = pulses.groupby( 'undulatorEV' )
        energy = np.array( [group.opisEV.mean() for _, group in bins] )
    else:
        bins = pulses.groupby( 'undulatorEV' )
        energy = np.array( [name for name, _ in bins] )

    #pulses.plot(kind='scatter', x='undulatorEV', y='opisEV') #The corresponding show is done later by plotParams
    #plt.show()

    #Plot relevant parameters
    utils.plotParams(shotsData)
    gmdData = shotsData.GMD

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
    print()
    traceAcc /= binCount[:,None]

    evConv = utils.mainTofEvConv(pulses.retarder.mean())
    evs  = evConv(chunk.columns)
    tofs = np.array(chunk.columns.to_numpy())

    # make dataframe and save data
    if cfg.writeOutput:
        np.savez(cfg.output.path + cfg.output.fname,
                 traceAcc = traceAcc, energy = energy, evs=evs, tofs=tofs)

if cfg.onlyplot:
    print("Reading data...")
    dataZ = np.load(cfg.output.path + cfg.output.fname + ".npz", allow_pickle=True)
    traceAcc = dataZ['traceAcc']
    energy   = dataZ['energy']
    evs      = dataZ['evs']
    try:
        tofs     = dataZ['tofs']
    except KeyError:
        pass

traceAcc -= traceAcc.min(axis=1)[:,None]
if cfg.Jacobian:
    traceAcc = utils.jacobianCorrect(traceAcc, evs)

if cfg.NormROI:
    NormROI   = slice( np.abs(evs - cfg.NormROI[1]).argmin() , np.abs(evs - cfg.NormROI[0]).argmin() )
    traceAcc /= traceAcc[:,NormROI].mean(axis=1)[:,None]


if cfg.plots.plotSlice:
    ROI = slice(np.abs(evs - cfg.plots.energy2d[1]).argmin() ,
                np.abs(evs - cfg.plots.energy2d[0]).argmin() )
    f = plt.figure(figsize=(9, 3))

    plt.plot(evs[ROI], traceAcc[cfg.plots.plotSlice[0],ROI], label=f'{energy[cfg.plots.plotSlice[0]]:.2f} eV')
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
        cmax = np.percentile(np.abs(traceAcc[:,ROI]),99)
        cmin = np.percentile(np.abs(traceAcc[:,ROI]),1)
        plt.pcolormesh(evs[ROI], yenergy, traceAcc[:,ROI],
                        cmap='bone_r', vmax=cmax, vmin=cmin)

        if cfg.plots.inset:
            #padding and x start position of inset relative to main axes
            INS_PAD = 0.04
            INS_START = 0.6
            x1, x2 = cfg.plots.inset[0]
            y1, y2 = cfg.plots.inset[1]
            #aspet ratio of inset. Used to calculate inset height
            aspect_ratio = ( (y2-y1) / (yenergy[-1] - yenergy[0] ) ) / ( (x2-x1) / (evs[ROI][0] - evs[ROI][-1]) )
            ins_width = 1 - INS_START - INS_PAD
            ax = plt.gca()
            axins = ax.inset_axes([INS_START, (1-INS_PAD) - ins_width*aspect_ratio, ins_width, ins_width*aspect_ratio])

            cmax = np.percentile(np.abs(traceAcc[:,ROI]),95.5)
            cmin = np.percentile(np.abs(traceAcc[:,ROI]),50)
            axins.pcolormesh(evs[ROI], yenergy, traceAcc[:,ROI],
                            cmap='bone_r', vmax=cmax, vmin=cmin)

            # sub region of the original image
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            ax.indicate_inset_zoom(axins)

    plt.xlabel("Kinetic energy (eV)")
    plt.ylabel("Photon energy (eV)")

    if cfg.difference:
        cb = plt.colorbar( orientation="vertical",fraction=0.06,anchor=(1.0,0.0) )

        ticks = cb.ax.yaxis.get_ticklabels()
        ticks[0] = 'pump\ndepleted'
        ticks[-1] = 'pump\nenhanced'
        cb.ax.set_yticklabels(ticks)

plt.show()
