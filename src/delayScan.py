#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict

import utils

import pickle

cfg = {    'data'     : { 'path'     : '/media/Fast2/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'fisrt_block.h5'
                        },
           'output'   : { 'path'     : './data/',
                          'fname'    : 'testBAM+'
                        },
           'time'     : { 'start' : datetime(2019,3,26,20,45,0).timestamp(),
                          'stop'  : datetime(2019,3,26,20,59,0).timestamp(),
                        },
           'filters'  : { 'opisEV'      : (270,275),
                          'retarder'    : (-81,-79),
#                          'delay'       : (1170, 1180.0),
                          'waveplate'   : (9.,11.)
                        },
           'delayBins'   : True, # if you want to set costumized (non equidistant) binning intervals set to True
           'delayBinStep': 0.01, # relevant if delayBins is False, choose between Step or Num
           #'delayBinNum' : 100,
           'ioChunkSize' : 50000,
           'gmdNormalize': True,
           'useBAM'      : True,
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
    averageBamShift = shotsData.query("BAM != 0").BAM.mean()
    print(f"Correcting delays with BAM data. Average shift is {averageBamShift:.3f} ps")
    shotsData['delay'] = utils.shotsDelay(pulses.delay.to_numpy(), shotsData.BAM.to_numpy())
else:
    shotsData['delay'] = utils.shotsDelay(pulses.delay.to_numpy(), shotsNum = shotsNum)
    averageBamShift = np.float32(0.)


if cfg.delayBins: # insert your binning intervals here
    print(f"Setting up customized bins")
    bin_right = [1180,1179.8,1179.6,1179.4,1179.2,1179,1178.975,1178.95,1178.925,1178.9,
                 1178.875,1178.85,1178.825,1178.8,1178.775,1178.75,1178.725,1178.7,1178.675,1178.65,
                 1178.625,1178.6,1178.575,1178.55,1178.525,1178.5,1178.475,1178.45,1178.425,1178.4,
                 1178.375,1178.35,1178.325,1178.3,1178.275,1178.25,1178.225,1178.2,1178.175,1178.15,
                 1178.125,1178.1,1178.075,1178.05,1178.025,1178,1177.975,1177.95,1177.925,1177.9,
                 1177.65,1177.4,1177.15,1176.9,1176.65,1176.4,1176.15,1175.9,1175.65,1175.4,
                 1175.15,1174.9,1173.9,1172.9,1171.9,1170.9,1169.9,1168.9,1160,1130,1080
                ] # right bounds
    bin_left = [1179.8,1179.6,1179.4,1179.2,1179,1178.975,1178.95,1178.925,1178.9,1178.875,
                1178.85,1178.825,1178.8,1178.775,1178.75,1178.725,1178.7,1178.675,1178.65,1178.625,
                1178.6,1178.575,1178.55,1178.525,1178.5,1178.475,1178.45,1178.425,1178.4,1178.375,
                1178.35,1178.325,1178.3,1178.275,1178.25,1178.225,1178.2,1178.175,1178.15,1178.125,
                1178.1,1178.075,1178.05,1178.025,1178,1177.975,1177.95,1177.925,1177.9,1177.65,
                1177.4,1177.15,1176.9,1176.65,1176.4,1176.15,1175.9,1175.65,1175.4,1175.15,1174.9,
                1173.9,1172.9,1171.9,1170.9,1169.9,1168.9,1167.9,1155,1125,1075
               ] # left bounds
    bin_left -= averageBamShift
    bin_right -= averageBamShift
    interval = pd.IntervalIndex.from_arrays(bin_left, bin_right)
#    for i in interval:
#        print(i)
    bins = shotsData.groupby( pd.cut(shotsData.delay, interval) )
    print(f"Loading {shotsData.shape[0]*2} shots")
else:
    binStart, binEnd = utils.getROI(shotsData)

    print(f"Loading {shotsData.shape[0]*2} shots")
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
img = np.zeros( ( len(bins), 3009 ))
binCount = np.zeros( len(bins) )
#Iterate over data chunks and accumulate them in img
for counter, chunk in enumerate(shotsTof):
    print( f"loading chunk {counter}", end='\r' )
    shotsDiff = utils.getDiff(chunk, gmdData)
    for binId, bin in enumerate(bins):
        name, group = bin
        group = group.query("GMD > 0.5 & BAM != 0") # GMD < 4.5 & uvPow > 320 & uvPow < 380 &
        binTrace = shotsDiff.reindex(group.index).mean()
        if not binTrace.isnull().to_numpy().any():
            img[binId] += binTrace
            binCount[binId] += 1

idx.close()
tr.close()

img /= binCount[:,None]
if uvEven > uvOdd: img *= -1

#plt.plot(binCount)
#plt.show()


#average all chunks, counter + 1 is the total number of chunks we loaded
#print("binCount[:,None]:", binCount[:,None])

#plt.plot(binCount[:,None])
#plt.show()

#get axis labels
delays = np.array( [name.mid for name, _ in bins] )
evConv = utils.mainTofEvConv(pulses.retarder.mean())
evs = evConv(shotsDiff.iloc[0].index.to_numpy(dtype=np.float32))

img = pd.DataFrame(data = img, columns=evs, index=delays).fillna(0)
img.to_hdf(cfg.output.path + cfg.output.fname + ".csv", "data",mode="w", format="table")


#plot resulting image
plt.figure()
cmax = np.max(np.abs(img.values))*0.1 #np.abs(img.values[np.logical_not(np.isnan(img))]).max()*0.1
plt.pcolormesh(evs, delays ,img.values, cmap='bwr', vmax=cmax, vmin=-cmax)
plt.xlabel("Kinetic energy (eV)")
plt.ylabel("Averaged Signal (counts)")
plt.tight_layout()
#plt.savefig(cfg.output.path + cfg.output.fname)
#plt.savefig(f'output-{cfg.time.start}-{cfg.time.stop}')
print("\nDone")
plt.show()

'''
#plot liner graph of integral over photoline
plt.figure()
#photoline = slice(820,877)
photoline = slice( np.abs(evs - 107).argmin() , np.abs(evs - 102.5).argmin() )
plt.plot(delays, img.T[photoline].sum(axis=0))
valence = slice( np.abs(evs - 145).argmin() , np.abs(evs - 140).argmin() )
plt.plot(delays, img.T[valence].sum(axis=0))
plt.show()


results = np.full((img.shape[0]+1,img.shape[1]+1), np.nan)
results[1:,1:] = img
results[1:,0] = delays
results[0,1:] = evs

with open(cfg.outFname + '.pickle', 'wb') as fout:
    pickle.dump(cfg, fout)
np.savetxt(cfg.outFname + '.txt', results, header='first row: kinetic energies, first column: delays')#img)

np.savetxt(cfg.outFname + '.txt', results, header='first row: kinetic energies, first column: delays')#img)'''
