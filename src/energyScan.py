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
                          'fname'    : 'energyScan_2p_thirdblock_50-51_diff_opis'
                        },
           'time'     : { #'start' : datetime(2019,3,30,21,57,0).timestamp(),
                          #'stop'  : datetime(2019,3,31,1,4,0).timestamp(),

                          #'start' : datetime(2019,4,5,17,26,0).timestamp(),
                          #'stop'  : datetime(2019,4,5,18,56,0).timestamp(),

                          'start' : datetime(2019,4,5,18,59,0).timestamp(),
                          'stop'  : datetime(2019,4,5,20,32,0).timestamp(),
                        },
           'filters'  : { 'retarder'    : (-10,0), #(-7,-3), (-2,2)
                          'waveplate'   : (10,15),
                          'delay'       : (1259.5, 1260),
                          'undulatorEV' : (150,180),#(203,245),(150,180)
                        },
           'sdfilter' : "GMD > 0.5", # filter for shotsdata parameters used in query method

           'ioChunkSize' : 50000,
           'decimate'    : False, #Decimate macrobunches before analizing. Use for quick evalutation of large datasets
           'gmdNormalize': True,
           'OPISbins'    : False,    #Use opis for energy bins labels

           'onlyOdd'     : False,   #Set to true if ony odd shots should be used, otherwise all shots are used
           'difference'  : True,  #Set to true if a even-odd difference spectrum shuold be calculated instead (onlyodd is ignored in this case)
           'timeZero'    : 1261.7, #Used to correct delays

           'plots' : {
                       'energy2d'      : (33, 250),
                            'inset2d'  : None,#((30,70),(219,232)), #Extent of zoomed in inset (ekin, ephoton). None for no inset
                       'ROIIntegral'   : None,#(30, 81),
                       'plotSlice'     : [1,-1],
                            'insetSl'  : None,#((160,245),(-0.5,1.6)),
                       'uvtext'        : "(2p edge, S pol, Low Uv, 2 ps delay, opis energies)"

           },
           'writeOutput' : True,  #Set to true to write out data in csv
           'onlyplot'    : False, #Set to true to load data form 'output' file and
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

    if cfg.difference:
        pulses.delay.hist()
        plt.show()
        delayAvg = cfg.timeZero - pulses.delay.mean()
        print(f"Average Delay {delayAvg:.4}ps")

    #Remove pulses with no corresponing shots
    pulses = pulses.drop( pulses.index.difference(shotsData.index.levels[0]) )

    #Bin data
    if cfg.OPISbins:
        #Get rid of pulses where OPIS was not working
        pulses_num = pulses.shape[0]
        pulses = pulses.query( '(undulatorEV - opisEV < 10) and (undulatorEV - opisEV > -10)' )
        print( f"Dropping {(pulses_num - pulses.shape[0]) / pulses_num * 100:.1}% of pulses due to OPIS mismatch")

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
    shotsData = shotsData.query(cfg.sdfilter)

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

    #Offset Correction
    ROI = slice( 0, np.abs(evs - 270).argmin() )
    traceAcc -= traceAcc[:,ROI].mean(axis=1)[:,None]
    #Jacobian
    traceAcc = utils.jacobianCorrect(traceAcc, evs)

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

if not cfg.difference:
    print("***RE-EXPORT STATIC DATA WITH JACOBIAN AND SHIFT BUILT IN***")


if cfg.plots.plotSlice:
    ROI = slice(np.abs(evs - cfg.plots.energy2d[1]).argmin() ,
                np.abs(evs - cfg.plots.energy2d[0]).argmin() )
    f = plt.figure(figsize=(9, 3))

    plt.plot(evs[ROI], traceAcc[cfg.plots.plotSlice[0],ROI], label=f'{energy[cfg.plots.plotSlice[0]]:.2f} eV')
    plt.plot(evs[ROI], traceAcc[cfg.plots.plotSlice[1],ROI], label=f'{energy[cfg.plots.plotSlice[1]]:.2f} eV')
    plt.xlabel("Kinetic energy (eV)")
    plt.ylabel(f"Intensity [a.u.]")
    f.subplots_adjust(left=0.08, bottom=0.17, right=0.96, top=0.95)
    plt.legend()

    if cfg.plots.insetSl:
        #padding and x start position of inset relative to main axes
        INS_YSTART = 0.4
        INS_XSTART = 0.4

        ax = plt.gca()
        x1, x2 = cfg.plots.insetSl[0]
        y1, y2 = cfg.plots.insetSl[1]
        ymin, ymax = ax.get_ylim()
        #aspet ratio of inset. Used to calculate inset height
        aspect_ratio = ( (y2-y1) / (ymax - ymin) ) / ( (x2-x1) / (evs[ROI][0] - evs[ROI][-1]) )
        ins_width = 0.97 - INS_XSTART

        axins = ax.inset_axes([INS_XSTART, INS_YSTART, ins_width, ins_width*aspect_ratio*3.22])

        cmax = np.percentile(np.abs(traceAcc[:,ROI]),95.5)
        cmin = np.percentile(np.abs(traceAcc[:,ROI]),50)
        axins.plot(evs[ROI], traceAcc[cfg.plots.plotSlice[0],ROI])
        axins.plot(evs[ROI], traceAcc[cfg.plots.plotSlice[1],ROI])

        # sub region of the original image
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        ax.indicate_inset_zoom(axins)


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
        cmax = np.percentile(np.abs(traceAcc[:,ROI]),99)
        plt.pcolormesh(evs[ROI], yenergy, traceAcc[:,ROI],
                       cmap='bwr', vmax=cmax, vmin=-cmax)
    else:
        cmax = np.percentile(np.abs(traceAcc[:,ROI]),99)
        cmin = np.percentile(np.abs(traceAcc[:,ROI]),1)
        plt.pcolormesh(evs[ROI], yenergy, traceAcc[:,ROI],
                        cmap='bone_r', vmax=cmax, vmin=cmin)

        if cfg.plots.inset2d:
            #padding and x start position of inset relative to main axes
            INS_PAD = 0.04
            INS_START = 0.6
            x1, x2 = cfg.plots.inset2d[0]
            y1, y2 = cfg.plots.inset2d[1]
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
