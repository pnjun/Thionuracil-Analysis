#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict

from utils import evConverter, mainTofEvConv

cfg = {    'data'     : { 'path'     : '/media/Fast2/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'fisrt_block.h5' #'71001915-71071776.h5' #'first block.h5'
                        }
      }
cfg = AttrDict(cfg)


idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')

kr_start = datetime(2019,3,25,12,2,0).timestamp()
kr_stop  = datetime(2019,3,25,12,10,0).timestamp()

#kr_start = datetime(2019,3,25,12,11,0).timestamp()
#kr_stop = datetime(2019,3,25,12,13,0).timestamp()

pulse  = idx.select('pulses', where='time >= kr_start and time < kr_stop')
kr_retarder = pulse.retarder.mean()
print(kr_retarder)

data    = tr.select('shotsTof', where='pulseId >= pulse.index[0] and pulseId < pulse.index[-1]')
kr_data = data.mean()

kr_conv = mainTofEvConv(kr_retarder)
kr_evs = kr_conv(kr_data.index)

plt.figure()
plt.title('Krypton')
plt.axvline(x=23.98)
plt.axvline(x=25.23)
plt.axvline(x=30.89)
plt.axvline(x=32.14)
plt.axvline(x=37.67)
plt.axvline(x=38.71)
plt.axvline(x=53.47)
plt.axvline(x=54.70)
plt.axvline(x=177.0)
plt.gca().set_xlim([15,60])
plt.gca().set_ylim([-50,0])
plt.gca().set_ylabel('intensity [a.u.]')
plt.gca().set_xlabel('electron energy [eV]')
plt.plot(kr_evs, kr_data, label = 'new')
plt.legend()

plt.show()
idx.close()
tr.close()
