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

cfg = {    'data'     : { 'path'     : '/media/Fast1/ThioUr/processed/',
                          'index'    : 'index.h5',
                          'trace'    : 'second_block.h5'
                        },
           'time'     : { 'start' : datetime(2019,3,30,20,16,0).timestamp(),
                          'stop'   : datetime(2019,3,31,1,4,0).timestamp()
                        },
           'filters'  : { 'opisEV' : (195,250), # replace with opisEV if opis column contains reasonable values, else use undulatorEV
                          #'delay'       : (1260.6,1260.8), # comment out if there is only non-UV data in the block
                          'retarder'    : (-1,1)
                        },
           'output'   : { 'img'  : './Plots/',  # where output images should be stored
                          'data' : './data/'    # where processed data should be stored
                        },
           'UV'       : False,
           'ROI'      : (30., 85.),  # eV region to be summed
           'evStep'   : 1,  # frequency for binning
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

if cfg.UV:
    shotsData = shotsData.query('shotNum % 2 == 0')

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
#left = [161, 162.5, 163.5, 164.5, 165.25, 166, 166.75, 167.5, 169.0, 170.5, 172]
#right = [162, 163.5, 164.5, 165, 165.75, 166.5, 167.25, 168.5, 170, 171.5, 173]
#intervals = pd.IntervalIndex.from_arrays(left, right)
evBin = pd.cut(photEn, intervals)
gmdBin = shotsData.mean(axis=1).groupby(evBin)
gmdStat = gmdBin.agg(["mean", "std", "count"])
gmdBin = gmdBin.mean()
opisBin = photEn.groupby(evBin).mean()

# plot number of bunches per bin and average gmd per bin
f1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(opisBin, gmdStat["count"] / 1000, ".")
ax1.set_ylabel("Number of MBs (x1000)")
ax2.errorbar(opisBin.dropna().values, gmdStat["mean"].dropna().values, fmt=".", yerr=gmdStat["std"].dropna().values)
ax2.set_xlabel("Photon Energy (eV)")
ax2.set_ylabel("Mean GMD value")
plt.tight_layout()
plt.savefig(cfg.output.img + "nexafs_{0}_stats.png".format(timetag), dpi=300)

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

#img = pd.read_csv(cfg.output.data + "nexafs_{0}.csv".format(timetag), header=0, index_col=0)
#img.columns = img.columns.astype('float64')
d = np.array([i - i.max() for i in img.values])
img = pd.DataFrame(data=d, columns=img.columns, index=img.index)

# make a plot
f2, ax = plt.subplots(1, 2, sharey=True, figsize=(10,4), gridspec_kw={'width_ratios': [2, 3]})
cm = ax[1].pcolormesh(img.columns.values, img.index.values, img.values, cmap="cividis")
cb = plt.colorbar(cm)
ax[1].set_xlabel("Kinetic Energy (eV)")
cb.set_label("GMD normalised average Counts")

# look at region of interest
mask = (img == img) & (img.columns >= cfg.ROI[0]) & (img.columns <= cfg.ROI[1])
roi = img[mask]
roi_max = [ i.argmin() for i in img.values]
roi_pl = [ img.values[i][roi_max[i]-25:roi_max[i]+25].sum() for i in range(len(img.values))]
roi = roi.T.dropna().T

ax[0].plot(roi.T.sum()/1000, roi.index, ".", label="Raw sum")
ax[0].plot((roi.T.sum() - roi_pl)/1000, roi.index, ".", label="PL corrected")
ax[0].set_ylabel('Photon Energy (eV)')
ax[0].set_xlabel('Summed Counts over ROI {0}-{1} eV (x1000)'.format(cfg.ROI[0], cfg.ROI[1]))
ax[0].set_ylim(202,246)
ax[0].legend(loc=9)
plt.tight_layout()
plt.savefig(cfg.output.img + "nexafs_{0}.png".format(timetag), dpi=300)
#plt.show()
plt.close("all")
