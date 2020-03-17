#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict
from matplotlib import gridspec
#from utils import mainTofEvConv

import utils

import pickle

cfg = {    'data'     : { 'path'     : '/media/Fast2/ThioUr/processed/',
                          'index'    : 'index.h5',
                          #'trace'    : 'second_block.h5'
                          'trace'    : 'fisrt_block.h5'
                        },
           'output'   : { 'path'     : './data/',
                          'fname'    : 'uvPowerScan_NonResonantConstant1000'
                        },
           'time'     : { #'start' : datetime(2019,3,31,16,12,0).timestamp(),
                          #'stop'  : datetime(2019,3,31,16,28,0).timestamp(),
                          'start' : datetime(2019,3,26,4,42,0).timestamp(),
                          'stop'  : datetime(2019,3,26,5,0,0).timestamp(),
                        },
           'filters'  : { 'undulatorEV' : (210,275),
                          'retarder'    : (-85,10),
                          #'delay'       : (1170, 1185.0),
                          'waveplate'   : (0,50)
                        },
           'sdfilter' : "GMD > 0.5 & BAM != 0", # filter for shotsdata parameters used in query method
           'uvBin_mode'  : 'CONSTANT', # Binning mode, must be one QUANTILE, CONSTANT
           'uvBinStep'   : 1000, # Size of bins, only relevant when uvBin_mode is CONSTANT
           'uvBinNum'    : 25,  # Number if bis to use, only relevant when uvBin_mode is QUANTILE
           'ioChunkSize' : 20000,
           'gmdNormalize': True,
           'lowPass'     : None,
           'decimate'    : False, #Decimate macrobunches before analizing. Use for quick evalutation of large datasets
           'spectrumAnalizer' : False,
           'writeOutput' : True,  #Set to true to write out data in csv
           'AugerROI'    : (130,155),


           'onlyplot'    : False , #Set to true to load data form 'output' file and
                                 #plot only.
           'plots' : { 'uv2d'           : True,
                       'AugerIntegral'  : True,
                       'fragmentSearch' : False,
                       'lowPassTest'    : False,
           },

      }


cfg = AttrDict(cfg)

if not cfg.onlyplot:
    idx = pd.HDFStore(cfg.data.path + cfg.data.index, mode = 'r')
    tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, mode = 'r')

    #Get all pulses within time limits
    pulses = idx.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')
    #Filter only pulses with parameters in range
    fpulses = utils.filterPulses(pulses, cfg.filters)

    #Get corresponing shotsspectrumAnalizer
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

    print(f"Loading {shotsData.shape[0]*2} shots")

    binStart = 100 #shotsData.uvPow.min()
    binEnd   = shotsData.uvPow.max()

    #Bin data on delay
    if cfg.uvBin_mode == 'CONSTANT':
        bins = shotsData.groupby(pd.cut(shotsData.uvPow,
                                        np.arange(binStart, binEnd, cfg.uvBinStep)))
    elif cfg.uvBin_mode == 'QUANTILE':
        shotsData = shotsData[ (shotsData.uvPow > binStart) & (shotsData.uvPow < binEnd) ]
        bins = shotsData.groupby( pd.qcut( shotsData.uvPow, cfg.uvBinNum ) )
    else:
        raise ValueError("binning mode not valid")


    print(bins.describe())

    '''
    waveplateBins = pulses.waveplate.unique()
    waveplateBins.sort()
    uvPowAvg = np.zeros(waveplateBins.shape)
    uvPowStd = np.zeros(waveplateBins.shape)
    for n, wp in enumerate(waveplateBins):
        idxs = pulses.query('waveplate == @wp').index
        data = shotsData.query('pulseId in @idxs').uvPow
        uvPowAvg[n] = data.mean()
        uvPowStd[n] = data.std()

    fig, ax = plt.subplots()
    ax.errorbar(waveplateBins, uvPowAvg, yerr=uvPowStd, fmt='o')
    plt.show()'''

    #Read in TOF data and calulate difference, in chunks
    shotsTof  = utils.h5load('shotsTof', tr, pulses, chunk=cfg.ioChunkSize)

    #Create empty ouptut image image
    diffAcc = np.zeros( ( len(bins), 3009 ))
    evenAcc = np.zeros( ( len(bins), 3009 ))
    oddAcc  = np.zeros( ( len(bins), 3009 ))
    binCount = np.zeros( len(bins) )

    #Iterate over data chunks and accumulate them in diffAcc
    for counter, chunk in enumerate(shotsTof):
        print( f"loading chunk {counter}", end='\r' )
        shotsDiff = utils.getDiff(chunk, gmdData, lowPass=cfg.lowPass)
        for binId, bin in enumerate(bins):
            name, group = bin
            #group = group.query(cfg.sdfilter)
            binIdx = shotsDiff.index.intersection(group.index)
            binTrace = shotsDiff.reindex(binIdx)

            #get separte df for odd and even shots
            even = chunk.query('shotNum % 2 == 0')
            odd =  chunk.query('shotNum % 1 == 0')
            #reindex odd shots to their corresponding even index,
            #since group.index matches only even shots for filtering (*)
            odd.index = odd.index.set_levels(odd.index.levels[1] - 1, level=1)
            evenAcc[binId] += even.reindex(binIdx).sum(axis=0)
            oddAcc[binId]  += odd.reindex(binIdx).sum(axis=0) #here (*)
            diffAcc[binId] += binTrace.sum(axis=0)
            binCount[binId] += binTrace.shape[0]

    idx.close()
    tr.close()

    diffAcc /= binCount[:,None]
    evenAcc /= binCount[:,None]
    oddAcc /= binCount[:,None]
    if cfg.spectrumAnalizer: speAcc /= binCount[:,None]

    if uvEven > uvOdd: diffAcc *= -1
    print("Done")

    #get axis labels
    binList = np.array( [name.mid for name, _ in bins] )

    evConv = utils.mainTofEvConv(pulses.retarder.mean())
    evs = evConv(shotsDiff.columns)

# make dataframe and save data
if cfg.writeOutput and not cfg.onlyplot:
    print("Writing output...")
    np.savez(cfg.output.path + cfg.output.fname,
            diffAcc = diffAcc, evenAcc = evenAcc, oddAcc=oddAcc, binList=binList, evs=evs)

if cfg.onlyplot:
    print("Reading data...")
    dataZ = np.load(cfg.output.path + cfg.output.fname + ".npz")
    diffAcc = dataZ['diffAcc']
    evenAcc = dataZ['evenAcc']
    oddAcc  = dataZ['oddAcc']
    evs     = dataZ['evs']
    binList = dataZ['binList']

print('Plotting...')

if cfg.spectrumAnalizer:
    FT = np.fft.rfft(diffAcc.to_numpy(), axis=0)

    plt.figure()
    plt.imshow(speAcc[:,:300], aspect=300/speAcc.shape[0], cmap='bone_r')
    plt.xlabel("Harmonic #")
    plt.ylabel("Relative Amplitu")
    plt.tight_layout()

#plot resulting image
if cfg.plots.uv2d:
    f = plt.figure()

    if cfg.plots.AugerIntegral:
        ROIslice = slice( np.abs(evs - cfg.AugerROI[1]).argmin() ,
                          np.abs(evs - cfg.AugerROI[0]).argmin() )

        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax1 = f.add_subplot(gs[1])

        plt.ylabel("Integrated Auger Intensity")
        plt.xlabel("Uv Power")
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.tick_params(axis='y', labelsize=0, length = 0)
        plt.plot(diffAcc[:,ROIslice].sum(axis=1), binList)

        f.add_subplot(gs[0])

    cmax = np.max(np.abs(diffAcc))
    edges = np.append(binList,binList[-1]+cfg.uvBinStep)
    plt.pcolormesh(evs, edges ,diffAcc, cmap='bwr', vmax=cmax, vmin=-cmax)
    plt.xlabel("Kinetic energy (eV)")
    plt.xlim(30,180)
    plt.ylabel("Uv Power")
    plt.tight_layout()
    #plt.savefig(cfg.output.path + cfg.output.fname)

if cfg.plots.fragmentSearch:
    from matplotlib.widgets import Slider
    f = plt.figure()

    diffPlot, = plt.plot(evs,diffAcc[-1], label='diffs')
    evenPlot, = plt.plot(evs,evenAcc[-1], label='even')
    oddPlot,  = plt.plot(evs,oddAcc[-1], label='odd')

    f.suptitle(binList[-1])
    def update(val):
        n = int(val)
        diffPlot.set_ydata(diffAcc[n])
        evenPlot.set_ydata(evenAcc[n])
        oddPlot.set_ydata(oddAcc[n])

        f.suptitle(binList[n])
        f.canvas.draw_idle()

    slax = plt.axes([0.1, 0.03, 0.65, 0.04])
    slider = Slider(slax, 'Amp', 0, diffAcc.shape[0]-1, valinit=0)
    slider.on_changed(update)

if cfg.plots.lowPassTest:
    from matplotlib.widgets import Slider
    f = plt.figure()
    N = 2

    diffPlot, = plt.plot(evs,diffAcc[N], label='diffs')
    evenPlot, = plt.plot(evs,evenAcc[N], label='even')
    len = diffAcc.shape[1]

    f.suptitle(0)
    def update(val):
        cutoff = int(val)

        diffFFT = np.fft.rfft(diffAcc[N])
        evenFFT = np.fft.rfft(evenAcc[N])
        diffFFT[-cutoff:] = (1-np.arange(0,1,cutoff))
        evenFFT[-cutoff:] = (1-np.arange(0,1,cutoff))
        diffPlot.set_ydata(np.fft.irfft(diffFFT, n = len))
        evenPlot.set_ydata(np.fft.irfft(evenFFT, n = len))

        f.suptitle(cutoff)
        f.canvas.draw_idle()

    slax = plt.axes([0.1, 0.03, 0.65, 0.04])
    slider = Slider(slax, 'Cutoff', 1, diffAcc.shape[1]//2, valinit=0)
    slider.on_changed(update)

plt.show()
