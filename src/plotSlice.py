#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict

import utils

cfg = {    'data'     : { 'path'     : '/media/Fast2/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'fisrt_block.h5'
                        },
           'time'     : { 'start' : datetime(2019,3,26,2,39,0).timestamp(),
                          'stop'  : datetime(2019,3,26,2,44,0).timestamp(),
                        }
      }
cfg = AttrDict(cfg)


idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')

pulse = idx.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')
data = utils.h5load('shotsTof', tr, pulse)

idx.close()
tr.close()

retarder = pulse.retarder.mean()
maxEv    = pulse.undulatorEV.mean()
print(f"avg ret {retarder:.2f} | avg delay {pulse.delay.mean():.2f} | avg undulator {maxEv:.2f} ")

even = data.query('shotNum % 2 == 0').mean()
odd  = data.query('shotNum % 2 == 1').mean()

evConv = utils.mainTofEvConv(retarder)
evs = evConv(data.columns)

#plt.suptitle("Static FEL only spectrum")
plt.plot(evs, -odd,  label='unpumped')
plt.plot(evs, -even, label='UV pumped')
plt.legend()
plt.gca().set_xlim([-retarder-10, maxEv+20])
plt.show()
