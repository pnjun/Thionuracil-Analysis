#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict

from utils import evConverter, mainTofEvConv

cfg = {    'data'     : { 'path'     : '/media/Data/Beamtime/processed/',
                          'index'    : 'index_BAM.h5',
                          'trace'    : 'first_block_BAM.h5' #'71001915-71071776.h5' #'first block.h5'
                        }
      }
cfg = AttrDict(cfg)


idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')

kr_start = datetime(2019,3,25,12,11,0).timestamp()
kr_stop  = datetime(2019,3,25,12,13,0).timestamp()

th1_start = datetime(2019,3,26,2,3,0).timestamp()
th1_stop  = datetime(2019,3,26,2,5,0).timestamp()
th2_start = datetime(2019,3,26,2,20,0).timestamp()
th2_stop  = datetime(2019,3,26,2,22,0).timestamp()

pulse  = idx.select('pulses', where='time >= kr_start and time < kr_stop')
kr_retarder = pulse.retarder.mean()
data    = tr.select('shotsTof', where='pulseId >= pulse.index[0] and pulseId < pulse.index[-1]')
kr_data = data.mean()

pulse   = idx.select('pulses', where='time >= th1_start and time < th1_stop')
th1_retarder = pulse.retarder.mean()
data     = tr.select('shotsTof', where='pulseId >= pulse.index[0] and pulseId < pulse.index[-1]')
th1_data = data.mean()

pulse = idx.select('pulses', where='time >= th2_start and time < th2_stop')
th2_retarder = pulse.retarder.mean()
data    = tr.select('shotsTof', where='pulseId >= pulse.index[0] and pulseId < pulse.index[-1]')
th2_data = data.mean()

kr_conv  = evConverter.mainTof(kr_retarder, mcp = 300)
th1_conv = evConverter.mainTof(th1_retarder, mcp = 300)
th2_conv = evConverter.mainTof(th2_retarder, mcp = 300)

offset = 6
tof = ( np.arange(350,3350) + offset ) * 0.0005
kr_evs = kr_conv(tof)
th1_evs = th1_conv(tof)
th2_evs = th2_conv(tof)

kr_conv2 = mainTofEvConv(kr_retarder)
kr_evs2 = kr_conv2(tof)
th1_conv2 = mainTofEvConv(th1_retarder)
th1_evs2 = th1_conv2(tof)

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
plt.plot(kr_evs2, kr_data, label = 'new2')
plt.plot(kr_data.index - kr_retarder, kr_data, label='old')
plt.legend()
'''
plt.figure()
plt.gca().set_xlim([0,300])
plt.gca().set_ylabel('intensity [a.u.]')
plt.gca().set_xlabel('electron energy [eV]')
plt.gca().set_title('Thiouracil 0V retardation')
plt.plot(th1_evs, th1_data)

plt.figure()
plt.subplot(211)
plt.gca().set_xlim([0,300])
plt.gca().set_ylabel('intensity [a.u.]')
plt.gca().set_xlabel('electron energy [eV]')
plt.axvline(x=105.6)
plt.axvline(x=140.8)
plt.axvline(x=252.0)
plt.gca().set_title('0V retardation')
plt.plot(th1_evs, th1_data, label = 'new')
plt.plot(th1_data.index - th1_retarder, th1_data, label='old')
plt.legend()

plt.subplot(212)
plt.gca().set_xlim([0,300])
plt.gca().set_ylabel('intensity [a.u.]')
plt.gca().set_xlabel('electron energy [eV]')
plt.axvline(x=105.6)
plt.axvline(x=140.8)
plt.axvline(x=252.0)
plt.gca().set_title('80V retardation')
plt.plot(th2_evs, th2_data, label = 'new')
plt.plot(th2_data.index - th2_retarder, th2_data, label='old')
plt.legend()'''

plt.figure()
plt.title('Tof to eV conversion')
plt.gca().set_xlabel('time of flight [us]')
plt.gca().set_ylabel('electron energy [eV]')
plt.plot(tof, th1_evs, label = 'new')
plt.plot(tof, th1_evs2, label = 'new2')
plt.plot(tof, th1_data.index - th1_retarder, label = 'old')
plt.legend()

plt.figure()
ev = np.arange(2, 350, 1)
plt.plot(ev, th1_conv.ev2tof(ev), label = 'new')
plt.plot(ev, th1_conv2.ev2tof(ev), label = 'new')
plt.legend()

plt.show()
idx.close()
tr.close()
