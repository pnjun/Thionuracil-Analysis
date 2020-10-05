#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict

import utils

from scipy import interpolate

cfg = {    'data'     : { 'path'     : '/media/Fast2/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'fisrt_block.h5' #'71001915-71071776.h5' #'first block.h5'
                        },
           'output'   : { 'path'     : './data/',
                          'fname'    : 'kriptonPlot'
                        },
           'time'     : { 'start' : datetime(2019,3,25,11,0,0).timestamp(),
                          'stop'  : datetime(2019,3,25,12,10,0).timestamp(),
                        },
           'filters'  : { 'undulatorEV' : (260.,275.),
                          'retarder'    : (-22,1),
                          'waveplate'   : (6,11)
                        },
           'ioChunkSize' : 50000,
           'writeOutput' : True,  #Set to true to write out data in csv
           'onlyplot'    : True, #Set to true to load data form 'output' file and
                                  #plot only.
           'plotlims'    : (26.7,45)
      }
cfg = AttrDict(cfg)

if not cfg.onlyplot:
    idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
    tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')

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
    shotsCount = shotsData.shape[0]
    print(f"Loading {shotsData.shape[0]} shots")

    #Remove pulses with no corresponing shots
    pulses = pulses.drop( pulses.index.difference(shotsData.index.levels[0]) )

    interval  = pd.IntervalIndex.from_arrays([-21,-1],[-20,0])
    bins      = pulses.groupby( pd.cut(pulses.retarder, interval) )
    #bins = pulses.groupby( 'retarder' )
    shotsTof  = utils.h5load('shotsTof', tr, pulses, chunk=cfg.ioChunkSize)

    traceAcc = np.zeros( ( len(bins), 3009 ))
    binCount = np.zeros( len(bins) )

    #Iterate over data chunks and accumulate them in diffAcc
    for counter, chunk in enumerate(shotsTof):
        print( f"loading chunk {counter} of {shotsCount//cfg.ioChunkSize}",
               end='\r' )

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

    retarder = np.array( [name.mid for name, _ in bins] )
    traceAcc /= binCount[:,None]
    tofs = np.array(chunk.columns.to_numpy())
    print(retarder)
    print(traceAcc)

    # make dataframe and save data
    if cfg.writeOutput:
        np.savez(cfg.output.path + cfg.output.fname,
                 traceAcc = traceAcc, retarder = retarder, tofs=tofs)

if cfg.onlyplot:
    print("Reading data...")
    dataZ = np.load(cfg.output.path + cfg.output.fname + ".npz", allow_pickle=True)
    traceAcc = dataZ['traceAcc']
    retarder = dataZ['retarder']
    tofs     = dataZ['tofs']

plt.figure()

zero1 = 27.54
zero2 = 44.44

class customEvConverter:
    ''' Converts between tof and Ev for main chamber TOF spectrometer
        Usage:
        converter = mainTofEvConv(retarder)
        energy = converter(tof)
    '''
    def __init__(self, retarder):
        self.r = retarder

        evMin = -retarder + 1
        evMax = -retarder + 350

        evRange = np.arange(evMin, evMax, 0.01)
        tofVals = self.ev2tof( evRange )
        self.interpolator = interpolate.interp1d(tofVals, evRange, kind='linear')

    def __call__(self, tof):
        if isinstance(tof, pd.Index):
            tof = tof.to_numpy(dtype=np.float32)
        return self.interpolator(tof)

    def ev2tof(self, e):
        #Parameters for evConversion
        l1 = 0.09      #meters
        l2 = 1.695     #meters
        l3 = 0.002
        m_over_2e = 5.69 / 2
        evOffset = 0.28 #eV

        new_e = e - evOffset
        return np.sqrt(m_over_2e) * ( l1 / np.sqrt(new_e) +
                                      l2 / np.sqrt(new_e + self.r) +
                                      l3 / np.sqrt(new_e + 300) )

colors = ['orange', 'red']

for n, trace in enumerate(traceAcc):
    evConv = customEvConverter(retarder[n])
    evs  = evConv(tofs)
    ROI = slice(np.abs(evs - cfg.plotlims[1]).argmin() , np.abs(evs - cfg.plotlims[0]).argmin())
    trace = trace[ROI]
    evs = evs[ROI]

    p1 = np.abs(evs - zero1).argmin()
    p2 = np.abs(evs - zero2).argmin()
    def bg(x):
        return ((trace[p2] - trace[p1])*x + (trace[p1]*evs[p2] - trace[p2]*evs[p1]) ) / (evs[p2] - evs[p1])

    trace -= bg(evs)

    trace = utils.jacobianCorrect(trace, evs)

    trace /= np.linalg.norm(trace)
    plt.plot(evs, trace, label=f'retarder {-retarder[n]:.0f} V', color=colors[n])

kr_data = np.loadtxt("data/Kr_Werme_M45NN_ref.dat")
ROI = slice(np.abs(kr_data[:,0] - cfg.plotlims[0]).argmin() , np.abs(kr_data[:,0] - cfg.plotlims[1]).argmin())
kr_data[:,1] /= np.linalg.norm(kr_data[ROI,1])
kr_data[:,1] -= kr_data[0,1]
plt.plot(kr_data[:,0], kr_data[:,1], ':',label='reference', color='black')

plt.gca().set_xlim([cfg.plotlims[0],cfg.plotlims[1]])
plt.gca().set_ylabel('intensity [a.u.]')
plt.gca().set_xlabel('electron energy [eV]')
plt.legend()

plt.show()

#Kripton lines
exit()
plt.axvline(x=23.98)
plt.axvline(x=25.23)
plt.axvline(x=30.89)
plt.axvline(x=32.14)
plt.axvline(x=37.67)
plt.axvline(x=38.71)
plt.axvline(x=53.47)
plt.axvline(x=54.70)
plt.axvline(x=177.0)
