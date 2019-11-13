#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict

from utils import evConverter

cfg = {    'data'     : { 'path'     : '/media/Fast2/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'second_block.h5' #'71001915-71071776.h5' #'first block.h5'
                        },
           'time'     : { 'start' : datetime(2019,4,1,2,52,0).timestamp(),
                          'stop'  : datetime(2019,4,1,2,52,1).timestamp()
                        }
      }
cfg = AttrDict(cfg)


idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')

pulse = idx.select('pulses',
                    where='time >= cfg.time.start and time < cfg.time.stop')

print(pulse[['retarder','opisEV']])

metad = tr.select('shotsData', where='pulseId >= pulse.index[0] and pulseId < pulse.index[-1]')
metad.GMD.plot()
plt.show()

data = tr.select('shotsTof', where='pulseId >= pulse.index[0] and pulseId < pulse.index[-1]')
data = data.mean()

plt.plot(data.index, data)
plt.show()

idx.close()
tr.close()
