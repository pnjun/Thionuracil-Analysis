#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
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
           'time'     : { 'start' : datetime(2019,4,1,4,40,0).timestamp(),
                          'stop'  : datetime(2019,4,1,7,3,0).timestamp()
                        },
           'filters'  : { 'opisEV' : (216,228), # replace with opisEV if opis column contains reasonable values, else use undulatorEV
                          'delay'       : (1256.95,1257.05), # comment out if there is only non-UV data in the block
                          'waveplate'   : (13.0, 13.5),
                          'retarder'    : (-6,-4)
                        },
           'output'   : { 'img'  : './Plots/',  # where output images should be stored
                          'data' : './data/'    # where processed data should be stored
                        },
           'ROI'      : (30, 80),  # eV region to be summed, either tuple or all
           'PLC'      : False,  # photoline correction in region of interest
           'evStep'   : 1.5,  # frequency for binning
           'ioChunkSize' : 50000
      }

cfg = AttrDict(cfg)

# change tag if you change parameters above!!!!
tag = "S2s_01_3080_-300fs"

idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')

pulses = idx.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')
pulsesLims = (pulses.index[0], pulses.index[-1])

pulses = filterPulses(pulses, cfg.filters)
#Get corresponing shots
assert len(pulses) != 0, "No pulses satisfy filter condition"

idx.close()

# Get GMD data for shots
shotsData = tr.select('shotsData', where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]',
                                          'pulseId in pulses.index'] )

shotsData = shotsData.query("GMD > .01")
gmdWUV = shotsData.query("shotNum % 2 == 0").pivot_table(index="pulseId", columns="shotNum", values="GMD")
gmdNUV = shotsData.query("shotNum % 2 == 1").pivot_table(index="pulseId", columns="shotNum", values="GMD")

# Remove pulses with no corresponing shots
pulses = pulses.drop( pulses.index.difference(gmdNUV.index) )
pulsesLims = (pulses.index[0], pulses.index[-1])


# check whether OPIS or Undulator values are used
evLim = None
photEn = None
try:
    evLim = cfg.filters.opisEV
    photEn = pulses.opisEV
except AttributeError:
    evLim = cfg.filters.undulatorEV
    photEn = pulses.undulatorEV

# bin pulse energies to photon energy
# first, create intervals
# for equally spaced energy steps use the first intervals variable and comment out the following 3 lines
# for weirdly spaced energy steps use the other three lines and define interval borders with left and right
intervals = pd.interval_range(start=evLim[0], end=evLim[1], freq=cfg.evStep)
#left = [161, 162.5, 163.5, 164.5, 165.25, 166, 166.75, 167.5, 169.0, 170.5, 172]
#right = [162, 163.5, 164.5, 165, 165.75, 166.5, 167.25, 168.5, 170, 171.5, 173]
#intervals = pd.IntervalIndex.from_arrays(left, right)
evBin = pd.cut(photEn, intervals)
gmdBinWUV = gmdWUV.mean(axis=1).groupby(evBin).agg(["sum", "mean", "std", "count"])
gmdBinNUV = gmdNUV.mean(axis=1).groupby(evBin).agg(["sum", "mean", "std", "count"])
opisBin = photEn.groupby(evBin).mean()

# look at actual tof data
shotsTof  = tr.select('shotsTof',
                      where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]', 'pulseId in pulses.index'],
                      iterator=True,
                      chunksize=cfg.ioChunkSize)

imgWUV = np.zeros( ( len(intervals), 3000 ))
imgNUV = np.zeros( ( len(intervals), 3000 ))
binCount = np.zeros( len(intervals) )
for counter, chunk in enumerate(shotsTof):
    c = 0
    print( f"processing chunk {counter}", end='\r' )
    idx = chunk.index.levels[0].values
    chunkWUV = chunk.query("shotNum % 2 == 0").groupby(level=[0]).mean()
    chunkNUV = chunk.query("shotNum % 2 == 1").groupby(level=[0]).mean()
    shotsBins = pd.cut(photEn.loc[idx], intervals)
    tofsWUV = pd.DataFrame(chunkWUV.groupby(shotsBins).mean().to_numpy())
    tofsNUV = pd.DataFrame(chunkNUV.groupby(shotsBins).mean().to_numpy())
    for group in range(tofsWUV.index.size):
        if not tofsWUV.loc[group].isnull().to_numpy().any():
            imgWUV[c] += tofsWUV.loc[group].to_numpy()
            imgNUV[c] += tofsNUV.loc[group].to_numpy()
            binCount[c] += 1
        c += 1

tofs = tr.select("shotsTof", where=["pulseId == pulsesLims[0]", "pulseId in pulses.index"]).columns
tr.close()

imgWUV /= binCount[:,None] # average energy binned tof traces with number of used chunks
imgWUV /= gmdBinWUV["mean"][:,None]  # normalise average tof traces on average gmd
imgNUV /= binCount[:,None]
imgNUV /= gmdBinNUV["mean"][:,None]

# calulate kinetic energy
evConv = mainTofEvConv(pulses.retarder.mean())
evs = evConv(tofs.to_numpy(dtype=np.float64))

# make pandas dataframe
imgWUV = pd.DataFrame(data=imgWUV, index=opisBin.values, columns=evs, dtype="float64").dropna()
imgNUV = pd.DataFrame(data=imgNUV, index=opisBin.values, columns=evs, dtype="float64").dropna()

# changing the offset so that data has same sign everywhere

d = np.array([i - i.max() for i in imgWUV.values])
imgWUV = pd.DataFrame(data=d, columns=imgWUV.columns, index=imgWUV.index)
d = np.array([i - i.max() for i in imgNUV.values])
imgNUV = pd.DataFrame(data=d, columns=imgNUV.columns, index=imgNUV.index)
del d

# calculate average, difference and projections of ROIs

aver = pd.DataFrame((imgWUV.values + imgNUV.values) / 2, index=imgNUV.index, columns=evs, dtype="float64")
diff = pd.DataFrame(imgWUV.values - imgNUV.values, index=imgNUV.index, columns=evs, dtype="float64")

def roi_sum(dataframe, region_of_interest, plc=False):

    if region_of_interest == "all":
        df_sum = dataframe.T.sum()
    else:
        mask = (dataframe == dataframe) & (dataframe.columns >= region_of_interest[0]) & (dataframe.columns <= region_of_interest[1])
        if plc:
            print("Photoline correction not implemented yet.")
        df_sum = dataframe[mask].T.dropna().sum()

    return df_sum

inuv_sum = roi_sum(imgNUV, cfg.ROI, cfg.PLC)
iwuv_sum = roi_sum(imgWUV, cfg.ROI, cfg.PLC)
aver_sum = roi_sum(aver, cfg.ROI, cfg.PLC)
diff_sum = roi_sum(diff, cfg.ROI, cfg.PLC)

# store data in new hdf file

hdf = h5py.File(cfg.output.data+tag+".h5", "w")
group = hdf.create_group("processed_data")
group.create_dataset("Ekin", data=evs, dtype=float)
group.create_dataset("Ephot", data=imgNUV.index.values, dtype=float)
group.create_dataset("unpumped_nexafs", data=imgNUV.values, dtype=float)
group.create_dataset("pumped_nexafs", data=imgWUV.values, dtype=float)
group.create_dataset("averaged_nexafs", data=aver.values, dtype=float)
group.create_dataset("difference_nexafs", data=diff.values, dtype=float)
group.create_dataset("unpumped_roi_sum", data=inuv_sum.values, dtype=float)
group.create_dataset("pumped_roi_sum", data=iwuv_sum.values, dtype=float)
group.create_dataset("average_roi_sum", data=aver_sum.values, dtype=float)
group.create_dataset("difference_roi_sum", data=diff_sum.values, dtype=float)
hdf.close()


# plot pumped and unpumed nexafs maps in one figure for comparison
f1, ax1 = plt.subplots(1, 2, sharey=True, figsize=(10,4))
cm11 = ax1[0].pcolormesh(imgNUV.columns.values, imgNUV.index.values, imgNUV.values, cmap="cividis", vmax=0, vmin=-300)
cm12 = ax1[1].pcolormesh(imgWUV.columns.values, imgWUV.index.values, imgWUV.values, cmap="cividis", vmax=0, vmin=-300)
cb1 = plt.colorbar(cm11)
ax1[0].set_xlabel("Kinetic Energy (eV)")
ax1[0].set_ylabel("Photon Energy (eV)")
ax1[0].set_title("unpumped")
ax1[1].set_xlabel("Kinetic Energy (eV)")
ax1[1].set_title("UV pumped")
cb1.set_label("GMD Normalised Signal)")
plt.tight_layout()
plt.savefig(cfg.output.img+tag+"_map_comp.png", dpi=150)
plt.close(f1)

# plot average nexafs map
f2, ax2 = plt.subplots(1, 2, sharey=True, figsize=(10,4), gridspec_kw={"width_ratios":[2, 3]})
ax2[0].plot(aver_sum.values, aver.index.values, ".")
ax2[0].set_xlabel("Summed Signal")
ax2[0].set_ylabel("Photon Energy (eV)")
cm2 = ax2[1].pcolormesh(aver.columns.values, aver.index.values, aver.values, cmap="cividis")
cb2 = plt.colorbar(cm2)
ax2[1].set_xlabel("Kinetic Energy (eV)")
cb2.set_label("GMD Normalised Signal")
plt.tight_layout()
plt.savefig(cfg.output.img+tag+"_average.png", dpi=150)
plt.close(f2)

# plot difference nexafs_map
f3, ax3 = plt.subplots(1, 2, sharey=True, figsize=(10,4), gridspec_kw={"width_ratios":[2, 3]})
ax3[0].plot(diff_sum.values, diff.index.values, ".")
ax3[0].set_xlabel("Summed Signal")
ax3[0].set_ylabel("Photon Energy (eV)")
cm3 = ax3[1].pcolormesh(diff.columns.values, diff.index.values, diff.values, cmap="cividis")
cb3 = plt.colorbar(cm3)
ax3[1].set_xlabel("Kinetic Energy (eV)")
cb3.set_label("GMD Normalised Signal")
plt.tight_layout()
plt.savefig(cfg.output.img+tag+"_diff.png", dpi=150)
plt.close(f3)

# plot roi sums of unpumped and pumped
f4 = plt.figure()
plt.plot(inuv_sum.index.values, inuv_sum.values, ".", label="unpumped")
plt.plot(iwuv_sum.index.values, iwuv_sum.values, ".", label="pumped")
plt.xlabel("Photon Energy (eV)")
plt.ylabel("Summed Signal (ROI)")
plt.legend()
plt.tight_layout()
plt.savefig(cfg.output.img+tag+"_roi_comp.png", dpi=150)
plt.close(f4)
