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
           'time'     : { 'start' : datetime(2019,3,25,12,0,0).timestamp(),
                          'stop'  : datetime(2019,3,25,12,0,1).timestamp(),
                          #'start' : datetime(2019,4,5,17,30,0).timestamp(),
                          #'stop'  : datetime(2019,4,5,17,35,0).timestamp(),
                        },
           'singleShot'   : False,
           'plotFraction' : False
      }
cfg = AttrDict(cfg)


idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')

pulse = idx.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')
gmd   = utils.h5load('shotsData', tr, pulse).GMD
data  = utils.h5load('shotsTof', tr, pulse)

print(f'loading {pulse.shape[0]} pulses, {data.shape[0]} shots')

idx.close()
tr.close()

retarder = pulse.retarder.mean()
maxEv    = pulse.undulatorEV.mean()
opisEv   = pulse.opisEV.mean()
print(f"avg ret {retarder:.2f} | avg delay {pulse.delay.mean():.2f} | avg undulator {maxEv:.2f} \navg GMD {gmd.mean():.2f} uJ | avg OPIS {opisEv:.2f}")


evConv = utils.mainTofEvConv(retarder)
evs = evConv(data.columns)

ROI = slice( np.abs(evs - 60).argmin(), np.abs(evs - 20).argmin() )
data -= data.to_numpy()[:,ROI].mean(axis=1)[:,None]
data *= -1
data = utils.jacobianCorrect(data, evs)

even = data.query('shotNum % 2 == 0')
odd  = data.query('shotNum % 2 == 1')

if cfg.singleShot:
    even = even.iloc[3]
    odd =  odd.iloc[3]
else:
    even = utils.traceAverage(even)
    odd =  utils.traceAverage(odd)

plt.figure()
#plt.suptitle("Static FEL only spectrum")
ax = plt.gca()
ax.set_ylabel('Intensity [au]')
ax.set_xlabel('Kinetic Energy [eV]')
plt.plot(evs, odd,  label='unpumped')
#plt.plot(evs, even, label='UV pumped')

#plt.legend()
plt.gca().set_xlim([-retarder-10, maxEv+20])

if cfg.plotFraction:
    from matplotlib.widgets import Slider
    f = plt.figure()
    frac = 0.1

    diffPlot, = plt.plot(evs, odd, label='ground state')
    evenPlot, = plt.plot(evs, (even-(1-frac)*odd)/frac, label='excited state')

    f.suptitle(0)
    def update(frac):
        evenPlot.set_ydata( (even-(1-frac)*odd)/frac )

        f.suptitle(frac)
        f.canvas.draw_idle()

    slax = plt.axes([0.1, 0.03, 0.65, 0.04])
    slider = Slider(slax, 'excited fraction', 0, 1, valinit=frac)
    slider.on_changed(update)

plt.show()
