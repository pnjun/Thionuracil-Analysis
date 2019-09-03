#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict

import utils

import pickle

cfg = {    'data'     : { 'path'     : '/media/Fast1/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'first_block.h5'
                        },
           #'time'     : { 'start' : datetime(2019,3,26,16,4,0).timestamp(),
           #              'stop'  : datetime(2019,3,26,23,59,0).timestamp(),
           'time'     : { 'start' : datetime(2019,3,26,16,25,0).timestamp(),
                          'stop'  : datetime(2019,3,26,20,40,0).timestamp(),
                        },
           'filters'  : { 'undulatorEV' : (269,274),
                          'retarder'    : (-81,-79),
                          'waveplate'   : (37,45)
                        },
           #'delayBinStep': 0.05,
           'delayBinNum' : 100,
           'ioChunkSize' : 200000,
           'gmdNormalize': False,
           'useBAM'      : True,

           'outFname'    : 'results_bam=True'
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
if(len(pulses) == 0):
    print("No pulses satisfy filter condition. Exiting\n")
    exit()
shotsData = tr.select('shotsData', where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]',
                                          'pulseId in pulses.index'] )
#Remove pulses with no corresponing shots
pulses = pulses.drop( pulses.index.difference(shotsData.index.levels[0]) )

#Plot relevant parameters
utils.plotParams(shotsData)

if cfg.gmdNormalize:
    gmdData = shotsData.GMD
else:
    gmdData = None

#Add Bam info
shotsNum = len(shotsData.index.levels[1]) / 2
shotsData = shotsData.query('shotNum % 2 == 0') #Remove unpumped pulses

if cfg.useBAM:
    shotsData['delay'] = utils.shotsDelay(pulses.delay.to_numpy(), shotsData.BAM.to_numpy())
else:
    shotsData['delay'] = utils.shotsDelay(pulses.delay.to_numpy(), shotsNum = shotsNum)

binStart, binEnd = utils.getROI(shotsData)

print(f"Loading {shotsData.shape[0]} shots")
print(f"Binning interval {binStart} : {binEnd}")

#Bin data on delay
if 'delayBinStep' in cfg.keys():
    bins = shotsData.groupby( pd.cut( shotsData.delay,
                                      np.arange(binStart, binEnd, cfg.delayBinStep) ) )
else:
    bins = shotsData.groupby( pd.qcut( shotsData.delay, cfg.delayBinNum ) )

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
    shotsDiff = utils.getDiff(chunk, gmdData)
    for binId, bin in enumerate(bins):
        name, group = bin
        group = group.query("GMD > 2.")
        binTrace = shotsDiff.reindex(group.index).mean()
        if not binTrace.isnull().to_numpy().any():
            img[binId] += binTrace
            binCount[binId] += 1

idx.close()
tr.close()

#average all chunks, counter + 1 is the total number of chunks we loaded
img /= binCount[:,None]
#get axis labels
delays = np.array( [name.mid for name, _ in bins] )
evConv = utils.mainTofEvConv(pulses.retarder.mean())
evs = evConv(shotsDiff.iloc[0].index.to_numpy(dtype=np.float32))


#plot resulting image
cmax = np.abs(img[np.logical_not(np.isnan(img))]).max()*0.1
plt.pcolor(evs, delays ,img, cmap='bwr', vmax=cmax, vmin=-cmax)
plt.savefig(cfg.outFname)
#plt.savefig(f'output-{cfg.time.start}-{cfg.time.stop}')
#plot liner graph of integral over photoline
plt.figure()
#photoline = slice(820,877)
photoline = slice( np.abs(evs - 107).argmin() , np.abs(evs - 102.5).argmin() )
plt.plot(delays, img.T[photoline].sum(axis=0))
valence = slice( np.abs(evs - 145).argmin() , np.abs(evs - 140).argmin() )
plt.plot(delays, img.T[valence].sum(axis=0))
plt.show()

'''
results = np.full((img.shape[0]+1,img.shape[1]+1), np.nan)
results[1:,1:] = img
results[1:,0] = delays
results[0,1:] = evs

with open(cfg.outFname + '.pickle', 'wb') as fout:
    pickle.dump(cfg, fout)

np.savetxt(cfg.outFname + '.txt', results, header='first row: kinetic energies, first column: delays')#img)'''
