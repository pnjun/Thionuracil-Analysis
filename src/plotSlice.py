#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict

import utils

cfg = {    'data'     : { 'path'     : '/media/Fast2/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'third_block.h5'
                        },
           'time'     : { 'start' : datetime(2019,4,6,2,13,0).timestamp(),
                          'stop'  : datetime(2019,4,6,2,15,0).timestamp(),
                        }
      }
cfg = AttrDict(cfg)


idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')

pulse = idx.select('pulses',
                    where='time >= cfg.time.start and time < cfg.time.stop')

print(pulse[['retarder','delay', 'undulatorEV']])


data = tr.select('shotsTof', where='pulseId >= pulse.index[0] and pulseId < pulse.index[-1]')
even = data.query('shotNum % 2 == 0').mean()
odd  = data.query('shotNum % 2 == 1').mean()

evConv = utils.mainTofEvConv(pulse.retarder.mean())
evs = evConv(data.columns)

plt.plot(evs, even, label='uv on')
plt.plot(evs, odd,  label='uv off')
plt.legend()
plt.show()


idx.close()
tr.close()
