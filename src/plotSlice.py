#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict

cfg = {    'data'     : { 'path'     : '/media/Data/Beamtime/processed/',     
                          'index'    : 'index_calibtest.h5',
                          'trace'    : 'calibtest.h5' #'71001915-71071776.h5' #'first block.h5' 
                        }
      }
cfg = AttrDict(cfg)


idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')

start = datetime(2019,3,25,15,24,50).timestamp()
stop  = datetime(2019,3,25,15,24,51).timestamp()

pulse = idx.select('pulses', where='time >= start and time < stop')

print(pulse.shape)
print(pulse['retarder'])

data = tr.select('shotsTof', where='pulseId >= pulse.index[0] and pulseId < pulse.index[-1]')


#data.mean().plot()
print(data.iloc[0].index[0], data.iloc[0].index[-1])
#plt.plot(np.arange(0,2700)*0.0005,data.mean().values)
#plt.show()
