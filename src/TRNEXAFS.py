#!/usr/bin/python3
import pandas as pd
import numpy as np
from datetime import datetime
from attrdict import AttrDict
import matplotlib.pyplot as plt
import sys
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
           'delayBinStep'  : 0.1,
           'energyBinStep' : 3,
           'ioChunkSize'   : 200000,
           'gmdNormalize'  : True,
           'useBAM'        : True,
           'electronROIeV' : (102.5,107), #Integration region bounds in eV (electron kinetic energy)

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
energyBins = pulses.groupby   ( pd.cut( pulses.undulatorEV,
                                        np.arange(pulses.undulatorEV.min(),
                                                  pulses.undulatorEV.max(),
                                                  cfg.energyBinStep) ) )

assert len(delayBins) > 0, "No delay bins"

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
    print( f"loading chunk {counter}", end='\r' )

    #gets bounds for photoline integration
    evs = evConv(chunk.iloc[0].index)
    photoline = slice( np.abs(evs - cfg.electronROIeV[1]).argmin() ,
                       np.abs(evs - cfg.electronROIeV[0]).argmin() )

    #calculate difference spectra and integrate over photoline
    shotsDiff = utils.getDiff(chunk, gmdData, integSlice=photoline)
    #iterate over delay bins
    for delayIdx, delayBin in enumerate(delayBins):
        _, group = delayBin
        group = group.query("GMD > 2.")

        #iterate over energy bins (note taht energy bins are just pulseIds)
        for energyIdx, energyBin in enumerate(energyBins):
            _, energyPulses = energyBin
            energyGroup = group.query("pulseId in @energyPulses.index")
            binVal = shotsDiff.reindex(energyGroup.index).mean()
            if not binVal.isnull().to_numpy().any():
                img[energyIdx, delayIdx] += binVal
                binCount[energyIdx, delayIdx] += 1

idx.close()
tr.close()

img /= binCount

#plot resulting image
delays = np.array( [name.mid for name, _ in delayBins] )
energy = np.array( [name.mid for name, _ in energyBins] )

cmax = np.abs(img[np.logical_not(np.isnan(img))]).max()*0.1
plt.pcolor(delays, energy, img, cmap='bwr', vmax=cmax, vmin=-cmax)
plt.show()


exit()
#save output for plotting
with open(cgf.outFname + '.pickle', 'wb') as fout:
    pickle.dump(cfg, fout)

np.savetxt(cgf.outFname + '.txt', results)
