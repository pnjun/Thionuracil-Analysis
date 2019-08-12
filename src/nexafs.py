#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from attrdict import AttrDict

from utils import mainTofEvConv
from utils import filterPulses

from numba import cuda
import cupy as cp

import matplotlib
matplotlib.use("GTK3Agg")

from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

cfg = {    'data'     : { 'path'     : '/media/Fast1/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'third_block.h5'
                        },
           'time'     : { 'start' : datetime(2019,4,5,19,16,0).timestamp(),
                          'stop'   : datetime(2019,4,5,20,32,0).timestamp()
                        },
           'filters'  : { 'opisEV' : (150,185), # replace with opisEV if opis column contains reasonable values
                          'delay'       : (1263,1264), # comment out if there is only non-UV data in the block
                          'retarder'    : (-6,-4)
                        },
           'output'   : { 'img'  : './Plots/',  # where output images should be stored
                          'data' : './data/'    # where processed data should be stored
                        },
           'ROI'      : (130., 150.),  # eV region to be summed
           'evStep'   : 0.75,  # frequency for binning
           'ioChunkSize' : 50000
      }

cfg = AttrDict(cfg)

timetag = '{0}-{1}'.format(datetime.fromtimestamp(cfg.time.start).isoformat(),
                           datetime.fromtimestamp(cfg.time.stop).isoformat())

idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')

pulses = idx.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')
pulsesLims = (pulses.index[0], pulses.index[-1])
#Filter only pulses with parameters in range
pulses = filterPulses(pulses, cfg.filters)
#Get corresponing shots
assert len(pulses) != 0, "No pulses satisfy filter condition"

idx.close()

shotsData = tr.select('shotsData', where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]',
                                          'pulseId in pulses.index'] )

shotsData = shotsData.pivot_table(index="pulseId", columns="shotNum", values="GMD")
shotsData = shotsData.where(shotsData >= .01).dropna() # remove data where fel was off

#Remove pulses with no corresponing shots
pulses = pulses.drop( pulses.index.difference(shotsData.index) )
pulsesLims = (pulses.index[0], pulses.index[-1])

evLim = None
photEn = None
try:
    evLim = cfg.filters.opisEV
    photEn = pulses.opisEV
except AttributeError:
    evLim = cfg.filters.undulatorEV
    photEn = pulses.undulatorEV

# bin pulse energies to photon energy
intervals = pd.interval_range(start=evLim[0], end=evLim[1], freq=cfg.evStep)
evBin = pd.cut(photEn, intervals)
gmdBin = shotsData.mean(axis=1).groupby(evBin).mean()
opisBin = photEn.groupby(evBin).mean()


# look at actual tof data
shotsTof  = tr.select('shotsTof',
                      where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]', 'pulseId in pulses.index'],
                      iterator=True,
                      chunksize=cfg.ioChunkSize)
img = np.zeros( ( len(intervals), 3000 ))
gmd = np.zeros( len(intervals) )
binCount = np.zeros( len(intervals) )
for counter, chunk in enumerate(shotsTof):
    c = 0
    print( f"processing chunk {counter}", end='\r' )
    idx = chunk.index.levels[0].values
    #chunk = normToGmd(chunk, shotsData.loc[idx])
    chunk = chunk.groupby(level=[0]).mean()
    gmds = shotsData.loc[idx].mean(axis=1)
    shotsBins = pd.cut(photEn.loc[idx], intervals)
    tofs = pd.DataFrame(chunk.groupby(shotsBins).mean().to_numpy())
    gmds = pd.DataFrame(gmds.groupby(shotsBins).mean().to_numpy())
    #print(gmds)
    for group in range(tofs.index.size):
        if not tofs.loc[group].isnull().to_numpy().any():
            img[c] += tofs.loc[group].to_numpy()
            gmd[c] += gmds.loc[group].to_numpy()
            binCount[c] += 1
        c += 1

tofs = tr.select("shotsTof", where=["pulseId == pulsesLims[0]", "pulseId in pulses.index"]).columns
tr.close()

img /= binCount[:,None]
gmd /= binCount
img /= gmd[:,None]

bg = np.array([ j[:100].mean() for j in img])
img -= bg[:,None]

evConv = mainTofEvConv(pulses.retarder.mean())
evs = evConv(tofs.to_numpy(dtype=np.float64))

# make proper pandas dataframe
img = pd.DataFrame(data=img, index=opisBin.values, columns=evs, dtype="float64").dropna()

# save this dataframe to text file
img.to_csv(cfg.output.data + "nexafs_{0}.csv".format(timetag), header=True, index=True , mode="w")

# make a plot
f3 = plt.figure()
cm = plt.pcolormesh(img.columns.values, img.index.values, img.values)#, cmap="bwr", vmin=-cmax, vmax=cmax)
cb = plt.colorbar(cm)
plt.xlabel("kinetic energy (ev)")
plt.ylabel("photon energy (ev)")
plt.tight_layout()
plt.savefig(cfg.output.img + "nexafs_{0}.png".format(timetag))

# look at region of interest
#img = pd.read_csv(cfg.output.data + "nexafs_{0}.csv".format(timetag), header=0, index_col=0)
#img.columns = img.columns.astype('float64')
mask = (img == img) & (img.columns >= cfg.ROI[0]) & (img.columns <= cfg.ROI[1])
roi = img[mask].T.dropna().T

f4 = plt.figure()
plt.plot(roi.index, roi.T.sum(), ".")
plt.xlabel('Photon energy (eV)')
plt.ylabel('Summed signal (arb. units)')
plt.title("Binned: {0} - {1} eV".format(cfg.ROI[0], cfg.ROI[1]))
plt.tight_layout()
plt.savefig(cfg.output.img + "nexafs_binned_{0}-{1}ev_{2}.png".format(cfg.ROI[0], cfg.ROI[1], timetag))
plt.show()

