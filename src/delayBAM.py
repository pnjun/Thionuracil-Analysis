#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict

from numba import cuda
import cupy as cp

from utils import mainTofEvConv
from utils import filterPulses

cfg = {    'data'     : { 'path'     : '/media/Data/Beamtime/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'first_block.h5'
                        },
          'time'     : { 'start' : datetime(2019,3,26,10,29,0).timestamp(),
                         'stop'  : datetime(2019,3,26,10,59,0).timestamp(),
           # 'time'     : { 'start' : datetime(2019,3,26,16,25,0).timestamp(),
           #                'stop'  : datetime(2019,3,26,20,40,0).timestamp(),
                        },
           'filters'  : { 'undulatorEV' : (270,271),
                          'retarder'    : (-81,-79),
                          'waveplate'   : (15,25)
                        },
           'delayBinStep': 0.025,
           'ioChunkSize' : 200000
      }
cfg = AttrDict(cfg)

idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')

#Get all pulses within time limits
pulses = idx.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')
pulsesLims = (pulses.index[0], pulses.index[-1])
#Filter only pulses with parameters in range
pulses = filterPulses(pulses, cfg.filters)

#Get corresponing shots
if(len(pulses) == 0):
    print("No pulses satisfy filter condition. Exiting\n")
    exit()
shotsData = tr.select('shotsData', where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]',
                                          'pulseId in pulses.index'] )
#Remove pulses with no corresponing shots
pulses = pulses.drop( pulses.index.difference(shotsData.index.levels[0]) )

#Plot relevant parameters
f1 = plt.figure()
f1.suptitle("Uv Power histogram\nPress esc to continue")
shotsData.uvPow.hist(bins=20)
f2 = plt.figure()
f2.suptitle(f"GMD histogram\nAverage:{shotsData.GMD.mean():.2f}")
shotsData.GMD.hist(bins=70)
def closeFigs(event):
    if event.key == 'escape':
        plt.close(f1)
        plt.close(f2)
f1.canvas.mpl_connect('key_press_event', closeFigs)
f2.canvas.mpl_connect('key_press_event', closeFigs)
plt.show()

#Remove unpumped pulses
shotsData = shotsData.query('shotNum % 2 == 0')

#Define CUDA kernel for delay adjustment
@cuda.jit
def shiftBAM(bam, delay):
    bam[cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x ] += delay[cuda.blockIdx.x]

#Check BAM data integrity
if shotsData.BAM.isnull().to_numpy().any():
    print("BAM data not complete (NaN). Exiting\n")
    exit()

#Copy data to GPU
bam = cp.array(shotsData.BAM.values)
delay = cp.array(pulses.delay.values)
#Shift BAM and add column with new data
shiftBAM[pulses.shape[0], shotsData.index.levels[1].shape[0] // 2](bam,delay)
shotsData['delay'] = bam.get()

#Show histogram and get center point for binning
shotsData.delay.hist(bins=70)
def getBinStart(event):
    global binStart
    binStart = event.xdata
def getBinEnd(event):
    global binEnd
    binEnd = event.xdata
    plt.close(event.canvas.figure)
plt.gcf().suptitle("Drag over ROI for binning")
plt.gcf().canvas.mpl_connect('button_press_event', getBinStart)
plt.gcf().canvas.mpl_connect('button_release_event', getBinEnd)
plt.show()

print(f"Loading {shotsData.shape[0]} shots")
print(f"Binning interval {binStart} : {binEnd}")

#Bin data on delay
bins = shotsData.groupby( pd.cut( shotsData.delay, np.arange(binStart, binEnd, cfg.delayBinStep) ) )

#Function that gets a TOF dataframe and uses CUDA to calculate pump probe difference specturm
def getDiff(tofTrace):
    #Cuda kernel for pump probe difference calculation
    @cuda.jit
    def tofDiff(tof):
        row = 2 * cuda.blockIdx.x
        col = cuda.blockIdx.y*cuda.blockDim.x + cuda.threadIdx.x
        # if gmd:
        #     tof[ row    , col ]  /= gmd[row]
        #     tof[ row + 1, col ]  /= gmd[row + 1]
        tof[ row, col ] -= tof[ row + 1, col ]
    #Get difference data
    tof = cp.array(tofTrace.to_numpy())
    tofDiff[ (tof.shape[0] // 2 , tof.shape[1] // 250) , 250 ](tof)
    return pd.DataFrame( tof[::2].get(), index = tofTrace.index[::2], columns=tofTrace.columns)

#Read in TOF data and calulate difference, in chunks
shotsTof  = tr.select('shotsTof',  where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]',
                                          'pulseId in pulses.index'],
                                   iterator=True, chunksize=cfg.ioChunkSize)

#Create empty ouptut image image
img = np.zeros( ( len(bins), 3000 ))
binCount = np.zeros( len(bins) )
#Iterate over data chunks and accumulate them in img
for counter, chunk in enumerate(shotsTof):
    print( f"loading chunk {counter}", end='\r' )
    shotsDiff = getDiff(chunk)
    for binId, bin in enumerate(bins):
        name, group = bin
        binTrace = shotsDiff.reindex(group.index).mean()
        if not binTrace.isnull().to_numpy().any():
            img[binId] += binTrace
            binCount[binId] += 1

#average all chunks, counter + 1 is the total number of chunks we loaded
img[np.isnan(img)] = 0
img /= binCount[:,None]
delays = np.array( [name.mid for name, _ in bins] )
#plot resulting image
evConv = mainTofEvConv(pulses.retarder.mean())
evs = evConv(shotsDiff.iloc[0].index.to_numpy(dtype=np.float32))

cmax = np.abs(img).max()*0.75
print(cmax)
plt.pcolor(evs, delays ,img, cmap='bwr')#, vmax=750, vmin=-750)

plt.figure()
#photoline = slice(820,877)
photoline = slice( np.abs(evs - 106).argmin() , np.abs(evs - 103.5).argmin() )
plt.plot(delays, img.T[photoline].sum(axis=0))
valence = slice( np.abs(evs - 145).argmin() , np.abs(evs - 140).argmin() )
plt.plot(delays, img.T[valence].sum(axis=0))
plt.savefig('output')
#plt.savefig(f'output-{cfg.time.start}-{cfg.time.stop}')
plt.show()

idx.close()
tr.close()
