#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict

from utils import evConverter
from utils import filterPulses

cfg = {    'data'     : { 'path'     : '/media/Data/Beamtime/processed/',
                          'index'    : 'index_BAM.h5',
                          'trace'    : 'first_block_BAM.h5' #'71001915-71071776.h5' #'first block.h5'
                        },
           'time'     : { 'start' : datetime(2019,3,27,0,30,0).timestamp(),
                          'stop'  : datetime(2019,3,27,1,0,0).timestamp(),
                        },
           'filters'  : { 'undulatorEV' : (270,271),
                          'retarder'    : (-81,-79),
                          'waveplate'   : (6,9)
                        },
           'delaybins': 1178.5 + np.arange(-2, 2, 0.01),
      }
cfg = AttrDict(cfg)

idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')

pulses = idx.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')
pulsesLims = (pulses.index[0], pulses.index[-1])
pulses = filterPulses(pulses, cfg.filters)

shotsData = tr.select('shotsData', where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]', 'pulseId in pulses.index'] )


shotsTof  = tr.select('shotsTof',  where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]', 'pulseId in pulses.index'] )

shotsUvOn   = shotsTof.query('shotNum % 2 == 0')
shotsUvOff  = shotsTof.query('shotNum % 2 == 1')
del shotsTof

diff = pd.DataFrame( shotsUvOn.to_numpy() - shotsUvOff.to_numpy(), index = shotsUvOn.index )

pulses.reset_index(inplace=True)
bins = pulses.groupby( pd.cut( pulses.delay, cfg.delaybins ) )

img = []
delays = []
for name, group in bins:
    if len(group):
        img.append   ( diff.query('pulseId in @group.pulseId').mean() )
        delays.append( name.mid )

img = np.array(img)
plt.pcolor(shotsUvOn.iloc[0].index + 80, delays ,img, cmap='bwr')

plt.figure()
plt.plot(delays, img.T[830:860].sum(axis=0))

plt.show()

idx.close()
tr.close()
