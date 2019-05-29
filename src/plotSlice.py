#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict

from utils import evConverter

cfg = {    'data'     : { 'path'     : '/media/Data/Beamtime/processed/',     
                          'index'    : 'index.h5',
                          'trace'    : 'first_block.h5' #'71001915-71071776.h5' #'first block.h5' 
                        }
      }
cfg = AttrDict(cfg)


idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')

start = datetime(2019,3,25,12,11,0).timestamp()
stop  = datetime(2019,3,25,12,13,0).timestamp()

pulse = idx.select('pulses', where='time >= start and time < stop')

print(pulse[['retarder','opisEV']])

data = tr.select('shotsTof', where='pulseId >= pulse.index[0] and pulseId < pulse.index[-1]')
data = data.mean()


        
offset = 7
conv = evConverter.mainTof(-0.18)
tof = ( np.arange(350,3350) + offset ) * 0.0005
evNew = conv(tof)

plt.figure()
plt.plot(evNew, data)
plt.plot(data.index, data)

plt.figure()
plt.plot(tof, data.index)
plt.plot(tof, evNew)

plt.show()

idx.close()
tr.close()
