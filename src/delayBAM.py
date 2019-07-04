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
           'time'     : { 'start' : datetime(2019,3,26,20,59,0).timestamp(),
                          'stop'  : datetime(2019,3,26,21,53,0).timestamp(),
                        },
           'filters'  : { 'undulatorEV' : (270,271),
                          'retarder'    : (-81,-79),
                          'waveplate'   : (7,11)
                        },
           'delaybins': np.arange(1175,1179,0.01)
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
    print("No pulses satisfy filter condition. Exiting")
    exit()
shotsData = tr.select('shotsData', where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]',
                                          'pulseId in pulses.index'] )
#Remove pulses with no corresponing shots
pulses = pulses.drop( pulses.index.difference(shotsData.index.levels[0]) )
#Remove unpumped pulses
shotsData = shotsData.query('shotNum % 2 == 0')

#Define CUDA kernel for delay adjustment
@cuda.jit
def shiftBAM(bam, delay):
    bam[cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x ] += delay[cuda.blockIdx.x]
#Copy data to GPU
bam = cp.array(shotsData.BAM.values)
delay = cp.array(pulses.delay.values)
#Shift BAM and add column with new data
shiftBAM[pulses.shape[0], shotsData.index.levels[1].shape[0] // 2](bam,delay)
shotsData['delay'] = bam.get()

shotsData.delay.hist(bins=100)
plt.show()
plt.plot()

print(f"Loading {shotsData.shape[0]} shots")

#Bin data on delay
bins = shotsData.groupby( pd.cut( shotsData.delay, cfg.delaybins ) )

#Gets a TOF dataframe and uses CUDA to calculate pump probe difference specturm
def getDiff(tofTrace):
    #Cuda kernel for pump probe difference calculation
    @cuda.jit
    def tofDiff(tof):
        tof[ 2 * cuda.blockIdx.x    , cuda.blockIdx.y*cuda.blockDim.x + cuda.threadIdx.x ] -= \
        tof[ 2 * cuda.blockIdx.x + 1, cuda.blockIdx.y*cuda.blockDim.x + cuda.threadIdx.x ]
    #Get difference data
    tof = cp.array(tofTrace.to_numpy())
    tofDiff[ (tof.shape[0] // 2 , tof.shape[1] // 250) , 250 ](tof)
    return pd.DataFrame( tof[::2].get(), index = tofTrace.index[::2], columns=tofTrace.columns)

#Read in TOF data and calulate difference
shotsTof  = tr.select('shotsTof',  where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]',
                                          'pulseId in pulses.index'],
                                   iterator=True, chunksize=200000)

#Create empty ouptut image image
img = np.zeros( ( len(bins), 3000 ))
binCount = np.zeros( len(bins) )
#Iterate over data chunks and accumulate them in img
for counter, chunk in enumerate(shotsTof):
    print( f"loading chunk {counter}", end='\r' )
    shotsDiff = getDiff(chunk)
    for binId, bin in enumerate(bins):
        name, group = bin
        try:
            img[binId] += shotsDiff.reindex(group.index).mean()
            binCount[binId] += 1
        except KeyError:
            pass

#average all chunks, counter + 1 is the total number of chunks we loaded
print()
img /= binCount[:,None]
delays = np.array( [name.mid for name, _ in bins] )
#plot resulting image
evConv = mainTofEvConv(pulses.retarder.mean())

plt.pcolor(evConv(shotsDiff.iloc[0].index.to_numpy(dtype=np.float32)), delays ,img, cmap='bwr')

plt.figure()
plt.plot(delays, img.T[830:860].sum(axis=0))

plt.show()


idx.close()
tr.close()
