#!/usr/bin/python3
import pandas as pd
import numpy as np
from datetime import datetime
from attrdict import AttrDict

from numba import cuda
import cupy as cp

import utils

import pickle

cfg = {    'data'     : { 'path'     : '/media/Fast1/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'second_block.h5'
                        },
           'time'     : { 'start' : datetime(2019,4,1,4,40,0).timestamp(),
                          'stop'  : datetime(2019,4,1,7,3,0).timestamp(),
                        },
           'filters'  : {
                          'waveplate'   : (10,15),
                          'retarder'    : (-6,-4)
                        },
           'delayBinStep'  : 0.05,
           'energyBinStep' : 1,
           'ioChunkSize' : 200000,
           'gmdNormalize': False,
           'useBAM'      : False,
           'boundsROI'   : (100,105), #Integration region bounds in eV

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

#Plot relevant parameters
utils.plotParams(shotsData)

idx.close()
tr.close()

exit()
#save output for plotting
with open(cgf.outFname + '.pickle', 'wb') as fout:
    pickle.dump(cfg, fout)

np.savetxt(cgf.outFname + '.txt', results)
