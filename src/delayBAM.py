#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict

from numba import cuda
import cupy as cp

from utils import evConverter
from utils import filterPulses

cfg = {    'data'     : { 'path'     : '/media/Data/Beamtime/processed/',
                          'index'    : 'index_BAM.h5',
                          'trace'    : 'first_block_BAM.h5' #'71001915-71071776.h5' #'first block.h5'
                        },
           'time'     : { 'start' : datetime(2019,3,26,23,45,0).timestamp(),
                          'stop'  : datetime(2019,3,26,23,51,0).timestamp(),
                        },
           'filters'  : { 'undulatorEV' : (270,271),
                          'retarder'    : (-81,-79),
                          'waveplate'   : (39,41)
                        },
           'delaybins': 1177.5 + np.arange(-5, 5, 0.05),
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
shotsData = tr.select('shotsData', where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]', 'pulseId in pulses.index'] )
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
shiftBAM[pulses.shape[0], shotsData.index.levels[1].shape[0]](bam,delay)
shotsData['delay'] = bam.get()

#Read in TOF data and calulate difference
shotsTof  = tr.select('shotsTof',  where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]', 'pulseId in pulses.index'] )
shotsUvOn   = shotsTof.query('shotNum % 2 == 0')
shotsUvOff  = shotsTof.query('shotNum % 2 == 1')
del shotsTof
shotsDiff = pd.DataFrame( shotsUvOn.to_numpy() - shotsUvOff.to_numpy(), index = shotsUvOn.index )
del shotsUvOff

print("shotsData", shotsData.shape)
print("shotsDiff", shotsDiff.shape)

#Bin data on delay
bins = shotsData.groupby( pd.cut( shotsData.delay, cfg.delaybins ) )

#Create output image
img = []
delays = []
for name, group in bins:
    if len(group):
        img.append   ( shotsDiff.query('index in @group.index').mean() )
        delays.append( name.mid )

img = np.array(img)
plt.pcolor(shotsUvOn.iloc[0].index + 80, delays ,img, cmap='bwr')

plt.figure()
plt.plot(delays, img.T[830:860].sum(axis=0))

plt.show()


idx.close()
tr.close()
