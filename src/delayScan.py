#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict
#from utils import mainTofEvConv

import utils

import pickle

cfg = {    'data'     : { 'path'     : '/media/Fast2/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'second_block.h5'
                        },
           'output'   : { 'path'     : './data/',
                          'fname'    : 'stronFieldFragments.csv'
                        },
           'time'     : { 'start' : datetime(2019,3,31,11,30,0).timestamp(),
                          'stop'  : datetime(2019,3,31,11,50,0).timestamp(),
                        },
           'filters'  : { 'undulatorEV' : (205.,250.),
                          'retarder'    : (-10.,10),
                          #'delay'       : (1170, 1185.0),
                          'waveplate'   : (40,50)
                        },
           'sdfilter' : "GMD > 0.5 & BAM != 0", # filter for shotsdata parameters used in query method
           'delayBin_mode'  : 'QUANTILE', # Binning mode, must be one of CUSTOM, QUANTILE, CONSTANT
           'delayBinStep'   : 0.1,     # Size of bins, only relevant when delayBin_mode is CONSTANT
           'delayBinNum'    : 15,     # Number if bis to use, only relevant when delayBin_mode is QUANTILE
           'ioChunkSize' : 50000,
           'gmdNormalize': True,
           'useBAM'      : True,

           'plots' : { 'delay2d'    : True,
                       'photoShift' : False,
                       'valence'    : False,
                       'fragmentSearch' : True, #Plot Auger trace at long delays to look for fragmentation
           },
           'writeOutput' : True #Set to true to write out data in csv
      }


cfg = AttrDict(cfg)

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
print(f"Loading {shotsData.shape[0]*2} shots")


#chose binning dependent on delaybin_mode
if cfg.delayBin_mode == 'CUSTOM': # insert your binning intervals here
    print(f"Setting up customized bins")
    interval = pd.IntervalIndex.from_arrays(utils.CUSTOM_BINS_LEFT - averageBamShift,
                                            utils.CUSTOM_BINS_RIGHT - averageBamShift)
    bins = shotsData.groupby( pd.cut(shotsData.delay, interval) )

else:
	#choose from a plot generated
    binStart, binEnd = utils.getROI(shotsData)
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
    print( f"loading chunk {counter}", end='\r' )
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

#plot resulting image
if cfg.plots.delay2d:
    plt.figure()
    cmax = np.max(np.abs(diffAcc))
    plt.pcolormesh(evs, delays ,diffAcc, cmap='bwr', vmax=cmax, vmin=-cmax)
    plt.xlabel("Kinetic energy (eV)")
    plt.ylabel("Averaged Signal (counts)")
    plt.tight_layout()
    #plt.savefig(cfg.output.path + cfg.output.fname)
    #plt.savefig(f'output-{cfg.time.start}-{cfg.time.stop}')


if cfg.plots.photoShift:
    #plot line graph of integral over photoline
    plt.figure()
    photoline1 = slice(np.abs(evs - 98.5).argmin() , np.abs(evs - 92).argmin())
    photoline2 = slice(np.abs(evs - 100).argmin() , np.abs(evs - 98.5).argmin())
    photoline3 = slice(np.abs(evs - 107).argmin() , np.abs(evs - 102.5).argmin())

    NegPhLine, = plt.plot(delays, diffAcc.T[photoline3].sum(axis=0), label = 'negative photoline shift')
    PosPhLine1, = plt.plot(delays, diffAcc.T[photoline1].sum(axis=0), label = 'positive photoline shift 90-99 eV')
    PosPhLine2, = plt.plot(delays, diffAcc.T[photoline2].sum(axis=0), label = 'positive photoline shift 99-100 eV')
    plt.legend(handles=[PosPhLine1, PosPhLine2, NegPhLine])

if cfg.plots.valence:
    plt.figure()
    valence = slice( np.abs(evs - 145).argmin() , np.abs(evs - 140).argmin())
    plt.plot(delays, diffAcc.T[valence].sum(axis=0))

if cfg.plots.fragmentSearch:
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
