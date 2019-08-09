#!/usr/bin/python3
import pandas as pd
import numpy as np
from datetime import datetime
from attrdict import AttrDict
import matplotlib.pyplot as plt

import utils

import pickle

cfg = {    'data'     : { 'path'     : '/media/Fast1/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'third_block.h5'
                        },
           'time'     : { 'start' : datetime(2019,4,6,5,28,0).timestamp(),
                          'stop'  : datetime(2019,4,6,10,8,0).timestamp(),
                        },
           'filters'  : {
                          'waveplate'   : (10,15),
                          'retarder'    : (-15,-5)
                        },
           'delayBinStep'  : 0.05,
           'energyBinStep' : 1,
           'ioChunkSize'   : 200000,
           'gmdNormalize'  : False,
           'useBAM'        : True,
           'electronROIeV' : (100,105), #Integration region bounds in eV (electron kinetic energy)

           'outFname'    : 'trnexafs'
      }

cfg = AttrDict(cfg)

idx = pd.HDFStore(cfg.data.path + cfg.data.index, mode = 'r')
tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, mode = 'r')

#Get all pulses within time limits
pulses = idx.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')
pulsesLims = (pulses.index[0], pulses.index[-1])
#Filter only pulses with parameters in range
pulses = utils.filterPulses(pulses, cfg.filters)

#Get corresponing shots
assert len(pulses) != 0, "No pulses satisfy filter condition"

#Load Data
shotsData = tr.select('shotsData', where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]',
                                          'pulseId in pulses.index'] )

#Remove pulses with no corresponing shots
pulses = pulses.drop( pulses.index.difference(shotsData.index.levels[0]) )
assert not pulses.opisEV.isnull().any(), "Some opisEV values are NaN"

#Plot relevant parameters as sanity check
utils.plotParams(shotsData)

#Add Bam info
shotsData = shotsData.query('shotNum % 2 == 0')

if cfg.useBAM:
    shotsData['delay'] = utils.shotsDelay(pulses.delay.to_numpy(), shotsData.BAM.to_numpy())
else:
    shotsData['delay'] = utils.shotsDelay(pulses.delay.to_numpy(), None)

binStart, binEnd = utils.getROI(shotsData)

print(f"Loading {shotsData.shape[0]} shots")
print(f"Binning interval {binStart} : {binEnd}")

#load gmd data if needed
gmdData = shotsData.GMD if cfg.gmdNormalize else None

#Bin data on delay and opis energy
delayBins  = shotsData.groupby( pd.cut( shotsData.delay,
                                        np.arange(binStart, binEnd,
                                        cfg.delayBinStep) ) )
energyBins = pulses.groupby   ( pd.cut( pulses.opisEV,
                                        np.arange(pulses.opisEV.min(),
                                                  pulses.opisEV.max(),
                                                  cfg.energyBinStep) ) )

#Read in TOF data and calulate difference, in chunks
shotsTof  = tr.select('shotsTof',  where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]',
                                          'pulseId in pulses.index'],
                                   iterator=True, chunksize=cfg.ioChunkSize)

#Create empty ouptut image image
img      = np.zeros( ( len(energyBins),  len(delayBins)))
binCount = np.zeros( ( len(energyBins),  len(delayBins)))

evConv = utils.mainTofEvConv(pulses.retarder.mean())

#Iterate over data chunks and accumulate them in img
for counter, chunk in enumerate(shotsTof):
    #print( f"loading chunk {counter}", end='\r' )
    evs = evConv(chunk.iloc[0].index)
    photoline = slice( np.abs(evs - cfg.electronROIeV[1]).argmin() ,
                       np.abs(evs - cfg.electronROIeV[0]).argmin() )
    shotsDiff = utils.getDiff(chunk, gmdData, integSlice=photoline)
    print(shotsDiff)
    continue
    for binId, delayBin in enumerate(delayBins):
        name, group = delayBin
        group = group.query("GMD > 2.")
        binTrace = shotsDiff.reindex(group.index).mean()
        if not binTrace.isnull().to_numpy().any():
            img[binId] += binTrace
            binCount[binId] += 1

idx.close()
tr.close()

exit()
#save output for plotting
with open(cgf.outFname + '.pickle', 'wb') as fout:
    pickle.dump(cfg, fout)

np.savetxt(cgf.outFname + '.txt', results)
