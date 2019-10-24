#!/usr/bin/python3
"""
Script plots several parameters for given time window and filters as time series or histogram.

Parameters plotted:
- OPIS vs undulator
- waveplate
- retarder
- UV power even vs odd shots
- GMD
- BAM and corrected delays

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict

from utils import filterPulses, shotsDelay

import matplotlib
matplotlib.use("GTK3Agg")

cfg = {    'data'        : { 'path'     : '/media/Fast2/ThioUr/processed/',
                             'index'    : 'index.h5',
                             'trace'    : 'fisrt_block.h5'
                           },
           'time'        : { 'start'  : datetime(2019,3,26,16,25,0).timestamp(),
                             'stop'   : datetime(2019,3,27,7,7,0).timestamp()
                          },
           'filters'     : { 'opisEV' : (271,274),
                             'waveplate'   : (7.0, 8.0),
                             'retarder'    : (-81,-79),
                             'delay' : (1160.0,1180.0)
                           },
           'BAM'         : True,
           'ioChunkSize' : 200000
      }

cfg = AttrDict(cfg)

idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
pulses = idx.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')
if cfg.filters:
    pulses = filterPulses(pulses, cfg.filters)
pulsesLims = (pulses.index[0], pulses.index[-1])
idx.close()

print("Retardation value:")
print(f"mean: {pulses.retarder.mean()} +- {pulses.retarder.std()}")
print(f"min: {pulses.retarder.min()}, max: {pulses.retarder.max()}")


diff = pulses.opisEV - pulses.undulatorEV
print("Difference between OPIS readout and undulator setpoint:")
print(f"{diff.mean()} +- {diff.std()}")
if diff.std() > 1.:
    print("-> There are significant, inhomogeneous differences!")
else:
    print("-> There might only be an offset.")

tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')
shotsData = tr.select('shotsData', where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]',
                                          'pulseId in pulses.index'] )
tr.close()

pulses = pulses.drop( pulses.index.difference(shotsData.index.levels[0]) )

if cfg.BAM:
    bamNaN = shotsData.BAM.isna().sum()
    shotsData.BAM = shotsData.BAM.fillna(0)
    shotsData['delay'] = shotsDelay(pulses.delay.to_numpy(), shotsData.BAM.to_numpy())
    print(f"BAM column with {shotsData.shape[0]} entries includes {bamNaN} NaN values.")
    shotsData = shotsData.query("GMD > 0.5 & uvPow < 1000 & BAM != 0") # comment out if you want all data (including possible outlayer)
else:
    shotsData = shotsData.query("GMD > 0.5 & uvPow < 1000") # comment out if you want all data (including possible outlayer)


sdEven = shotsData.query("shotNum % 2 == 0")
sdOdd  = shotsData.query("shotNum % 2 == 1")

uvEven = sdEven["uvPow"]
uvOdd  = sdOdd["uvPow"]

print("UV diode signal analysis:")
print(f"Even shots: {uvEven.mean()} +- {uvEven.std()}")
print(f"Odd shots: {uvOdd.mean()} +- {uvOdd.std()}")

# make some plots
f1 = plt.figure()
plt.plot(pulses.time, pulses.retarder)
plt.xlabel("time")
plt.ylabel("retardation voltage (V)")
plt.tight_layout()

f2 = plt.figure()
plt.plot(pulses.time, pulses.opisEV, label="opis")
plt.plot(pulses.time, pulses.undulatorEV, label="undulator")
plt.xlabel("time")
plt.ylabel("photon energy (eV)")
plt.legend()
plt.tight_layout()

f3, ax3 = plt.subplots(1, 2, figsize=(9,4))
uvEven.hist(ax=ax3[0], bins=1000)
uvOdd.hist(ax=ax3[1], bins=1000)
ax3[0].set_xlabel("uv power")
ax3[1].set_xlabel("uv power")
ax3[0].set_ylabel("counts")
ax3[1].set_ylabel("counts")
ax3[0].set_title("Even shots")
ax3[1].set_title("Odd shots")
plt.tight_layout()

f4, ax4 = plt.subplots(1, 2, figsize=(9,4))
shotsData.hist("GMD", ax=ax4[0], bins=250)
ax4[0].set_xlabel("GMD value")
ax4[0].set_ylabel("Shots")
ax4[0].set_xlim(-1.5, 11.5)
plt.tight_layout()

gmdData = shotsData.pivot_table(index="pulseId", columns="shotNum", values="GMD")
ax4[1].errorbar(gmdData.columns.values, gmdData.mean(), fmt=".", yerr=gmdData.std(), label="Average")
ax4[1].plot(gmdData.columns.values, gmdData.median(), ".", label="Median")
#shotsData.boxplot(ax=ax4[1])
ax4[1].set_xlabel("Pulse number")
ax4[1].set_ylabel("Average energy (µJ)")
plt.tight_layout()

f5, ax5 = plt.subplots(1, 2, figsize=(9,4))
#shotsData.BAM.plot.kde(ax=ax5[0], legend=False)
shotsData.hist("BAM", ax=ax5[0], bins=100)
ax5[0].set_xlabel("BAM data (ps)")
ax5[0].set_ylabel("Counts")

shotsData.hist("delay", ax=ax5[1], bins=200 )
ax5[1].set_xlabel("Corrected Delays (ps)")
plt.tight_layout()
plt.show()
