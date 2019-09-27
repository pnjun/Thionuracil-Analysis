#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datetime import datetime
from attrdict import AttrDict

from utils import mainTofEvConv, filterPulses, shotsDelay

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
           'filters'  : { 'opisEV' : (211,223), # replace with opisEV if opis column contains reasonable values, else use undulatorEV
                          'waveplate'   : (13.0, 13.5),
                          'retarder'    : (-6,4)
                        },
           'delay'     : (1256.15,1256.67), # comment out if there is only non-UV data in the block
           'output'   : { 'img'  : './Plots/',  # where output images should be stored
                          'data' : './data/'    # where processed data should be stored
                        },
           'ROI'      : (30, 80),  # eV region to be summed, either tuple or all
           'PLC'      : False,  # photoline correction in region of interest
           'evStep'   : 1.,  # frequency for binning
           'ioChunkSize' : 50000
      }

cfg = AttrDict(cfg)

# change tag if you change parameters above!!!!
tag = "S2s_01_+050-500fs"

idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')

print("Selecting time intervall ...")
pulses = idx.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')
pulsesLims = (pulses.index[0], pulses.index[-1])

print("Filter macrobunches by given filters ...")
pulses = filterPulses(pulses, cfg.filters)
assert len(pulses) != 0, "No pulses satisfy filter condition"

idx.close()

# Get UV and GMD data for shots
print("Load shotsData ...")
shotsData = tr.select('shotsData', where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]', 'pulseId in pulses.index'] )
pulses = pulses.drop( pulses.index.difference(shotsData.index.levels[0]) )

# BAM correction
try:
    cfg.delay
    print("Making BAM correction for delays ...")
    BAM_mean = shotsData.BAM.mean()
    BAM_std = shotsData.BAM.std()
    print(f"Average shift will be: {BAM_mean:.3f} +- {BAM_std:.3f}")
    delayfilter = {"delay" : (cfg.delay[0] + BAM_mean, cfg.delay[1] + BAM_mean)}
    delayfilter = AttrDict(delayfilter)
    shotsData['delay'] = shotsDelay(pulses.delay.to_numpy(), shotsData.BAM.to_numpy())
    print(f"New delay filter is: ({delayfilter.delay[0]:.3f}, {delayfilter.delay[1]:.3f}). Filtering shotsData accordingly ...")
    shotsData = filterPulses(shotsData, delayfilter)
    assert len(shotsData) != 0, "No data satisfies filter condition in shotsData."
except AttributeError:
    print("Skipping BAM correction because there is no delay filter ...")

print("Filter outlayers in GMD and uvPow ...")
shotsData = shotsData.query("GMD > 1 & uvPow < 3000")
print("Separate even and odd shots ...")
sdEven = shotsData.query("shotNum % 2 == 0")
sdOdd = shotsData.query("shotNum % 2 == 1")

#print(sdEven.index.levels[0].size, sdOdd.index.levels[0].size)

# plot histograms
print("Plot histograms for GMD and uvPow ...")
fig1, ax1 = plt.subplots(1, 2, figsize=(9,4))
sdEven["GMD"].hist(ax=ax1[0], bins=100)
sdOdd["GMD"].hist(ax=ax1[1], bins=100)
ax1[0].set_xlabel("GMD")
ax1[1].set_xlabel("GMD")
ax1[0].set_ylabel("Counts")
ax1[0].set_title("Even")
ax1[1].set_title("Odd")
plt.tight_layout()
plt.savefig(cfg.output.img + tag + "_gmd_hist.png")
plt.close(fig1)

fig2, ax2 = plt.subplots(1, 2, figsize=(9,4))
sdEven["uvPow"].hist(ax=ax2[0], bins=100)
sdOdd["uvPow"].hist(ax=ax2[1], bins=100)
ax2[0].set_xlabel("uv power")
ax2[1].set_xlabel("uv power")
ax2[0].set_ylabel("Shots")
ax2[0].set_title("Even")
ax2[1].set_title("Odd")
plt.tight_layout()
plt.savefig(cfg.output.img + tag + "_uv_hist.png")
plt.close(fig2)

# rearrange tables and separate between uv and gmd
print("Separate even and odd dataframes in UV and GMD ...")
gmdEven = sdEven.pivot_table(index="pulseId", columns="shotNum", values="GMD")
gmdOdd = sdOdd.pivot_table(index="pulseId", columns="shotNum", values="GMD")

uvEven = sdEven.pivot_table(index="pulseId", columns="shotNum", values="uvPow")
uvOdd = sdOdd.pivot_table(index="pulseId", columns="shotNum", values="uvPow")

#print(gmdEven.index.size, gmdOdd.index.size)
#print(uvEven.index.size, uvOdd.index.size)

print("Check and correct dataframe sizes ...")
if gmdEven.index.size > gmdOdd.index.size:
    gmdEven = gmdEven.drop(gmdEven.index.difference(gmdOdd.index))
    gmdOdd = gmdOdd.drop(gmdOdd.index.difference(gmdEven.index))
elif gmdEven.index.size < gmdOdd.index.size:
    gmdEven = gmdEven.drop(gmdEven.index.difference(gmdOdd.index))
    gmdOdd = gmdOdd.drop(gmdOdd.index.difference(gmdEven.index))

if uvEven.index.size > uvOdd.index.size:
    uvEven = uvEven.drop(uvEven.index.difference(uvOdd.index))
    uvOdd = uvOdd.drop(uvOdd.index.difference(uvEven.index))
elif uvEven.index.size < uvOdd.index.size:
    uvEven = uvEven.drop(uvEven.index.difference(uvOdd.index))
    uvOdd = uvOdd.drop(uvOdd.index.difference(uvEven.index))

#print(gmdEven.index.size, gmdOdd.index.size)
#print(uvEven.index.size, uvOdd.index.size)

# Remove pulses with no corresponing shots
pulses = pulses.drop( pulses.index.difference(gmdEven.index) )
pulsesLims = (pulses.index[0], pulses.index[-1])

# check whether OPIS or Undulator values are used
print("Check if OPIS or Undulator values are used ...")
evLim = None
photEn = None
try:
    evLim = cfg.filters.opisEV
    photEn = pulses.opisEV
except AttributeError:
    evLim = cfg.filters.undulatorEV
    photEn = pulses.undulatorEV

# bin pulse energies to photon energy
print("Bin stuff ...")
intervals = pd.interval_range(start=evLim[0], end=evLim[1], freq=cfg.evStep)
evBin = pd.cut(photEn, intervals)
opisBin = photEn.groupby(evBin)
gmdBinEven = gmdEven.mean(axis=1).groupby(evBin)
gmdBinOdd = gmdOdd.mean(axis=1).groupby(evBin)
uvBinEven = uvEven.mean(axis=1).groupby(evBin)
uvBinOdd = uvOdd.mean(axis=1).groupby(evBin)

# do some plots

try:
    print("Plot Ephot histogram ...")
    fig3, ax3 = plt.subplots(1, 1, figsize=(5,4))
    opisBin.plot(kind="hist", ax=ax3)
    ax3.set_xlabel("Photon energy (eV)")
    plt.title("Photon energy bin histogram")
    plt.tight_layout()
    plt.savefig(cfg.output.img + tag + "_ephot_hist.png")
    plt.close(fig3)
except TypeError:
    print("TypeError occured. Could not plot Ephot histogram due to missing data in bin. Continue anyways.")

print("Plot mean values for binned GMD and uvPow ...")
fig4, ax4 = plt.subplots(2, 2, figsize=(9, 9), sharex=True)
ax4[0,0].errorbar(opisBin.mean().values, gmdBinEven.mean().values, fmt=".", yerr=gmdBinEven.std().values, label="Mean & Std.")
ax4[0,0].plot(opisBin.mean().values, gmdBinEven.median().values, ".", label="Median")
ax4[0,0].legend()
ax4[0,0].set_title("GMD even")
ax4[0,1].errorbar(opisBin.mean().values, gmdBinOdd.mean().values, fmt=".", yerr=gmdBinOdd.std().values, label="Mean")
ax4[0,1].plot(opisBin.mean().values, gmdBinOdd.median().values, ".", label="Median")
ax4[0,1].set_title("GMD odd")
ax4[1,0].errorbar(opisBin.mean().values, uvBinEven.mean().values, fmt=".", yerr=uvBinEven.std().values, label="Mean & Std.")
ax4[1,0].plot(opisBin.mean().values, uvBinEven.median().values, ".", label="Median")
ax4[1,0].set_title("UV even")
ax4[1,1].errorbar(opisBin.mean().values, uvBinOdd.mean().values, fmt=".", yerr=uvBinOdd.std().values, label="Mean & Std.")
ax4[1,1].plot(opisBin.mean().values, uvBinOdd.median().values, ".", label="Median")
ax4[1,1].set_title("UV Odd")
ax4[1,0].set_xlabel("Photon energy (eV)")
ax4[1,1].set_xlabel("Photon energy (eV)")
ax4[0,0].set_ylabel("Gmd value")
ax4[1,0].set_ylabel("Uv power")
plt.tight_layout()
plt.savefig(cfg.output.img + tag + "_eo_stat.png")
plt.close(fig4)

# now look at actual tof data
print("Go through tof data now ...")
shotsTof  = tr.select('shotsTof',
                      where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]', 'pulseId in pulses.index'],
                      iterator=True,
                      chunksize=cfg.ioChunkSize)

imgEven = np.zeros( ( len(intervals), 3000 ))
imgOdd = np.zeros( ( len(intervals), 3000 ))
binCount = np.zeros( len(intervals) )
for counter, chunk in enumerate(shotsTof):
    c = 0
    print( f"processing chunk {counter}", end='\r' )
    idx = chunk.index.levels[0].values
    chunkEven = chunk.query("shotNum % 2 == 0").groupby(level=[0]).mean()
    chunkOdd = chunk.query("shotNum % 2 == 1").groupby(level=[0]).mean()
    shotsBins = pd.cut(photEn.loc[idx], intervals)
    tofsEven = pd.DataFrame(chunkEven.groupby(shotsBins).mean().to_numpy())
    tofsOdd = pd.DataFrame(chunkOdd.groupby(shotsBins).mean().to_numpy())
    for group in range(tofsEven.index.size):
        if not tofsEven.loc[group].isnull().to_numpy().any():
            imgEven[c] += tofsEven.loc[group].to_numpy()
            imgOdd[c] += tofsOdd.loc[group].to_numpy()
            binCount[c] += 1
        c += 1

tofs = tr.select("shotsTof", where=["pulseId == pulsesLims[0]", "pulseId in pulses.index"]).columns
tr.close()

print("Final average and GMD normalisation for even and odd data ...")
imgEven /= binCount[:,None] # average energy binned tof traces with number of used chunks
imgEven /= gmdBinEven.mean()[:,None]  # normalise average tof traces on average gmd
imgOdd /= binCount[:,None]
imgOdd /= gmdBinOdd.mean()[:,None]

print("Prepare dataframes and calculate other stuff ...")
# calulate kinetic energy
evConv = mainTofEvConv(pulses.retarder.mean())
evs = evConv(tofs.to_numpy(dtype=np.float64))

# make pandas dataframe
imgEven = pd.DataFrame(data=imgEven, index=opisBin.mean().values, columns=evs, dtype="float64").dropna()
imgOdd = pd.DataFrame(data=imgOdd, index=opisBin.mean().values, columns=evs, dtype="float64").dropna()

# changing the offset so that data has same sign everywhere

d = np.array([i - i.max() for i in imgEven.values])
imgEven = pd.DataFrame(data=d, columns=imgEven.columns, index=imgEven.index)
d = np.array([i - i.max() for i in imgOdd.values])
imgOdd = pd.DataFrame(data=d, columns=imgOdd.columns, index=imgOdd.index)
del d

# calculate average, difference and projections of ROIs

aver = pd.DataFrame((imgEven.values + imgOdd.values) / 2, index=imgOdd.index, columns=evs, dtype="float64")
diff = pd.DataFrame(imgEven.values - imgOdd.values, index=imgOdd.index, columns=evs, dtype="float64")

def roi_sum(dataframe, region_of_interest, plc=False):

    if region_of_interest == "all":
        df_sum = dataframe.T.sum()
    else:
        mask = (dataframe == dataframe) & (dataframe.columns >= region_of_interest[0]) & (dataframe.columns <= region_of_interest[1])
        if plc:
            print("Photoline correction not implemented yet.")
        df_sum = dataframe[mask].T.dropna().sum()

    return df_sum

iOdd_sum = roi_sum(imgOdd, cfg.ROI, cfg.PLC)
iEven_sum = roi_sum(imgEven, cfg.ROI, cfg.PLC)
aver_sum = roi_sum(aver, cfg.ROI, cfg.PLC)
diff_sum = roi_sum(diff, cfg.ROI, cfg.PLC)

# store data in new hdf file
print("Store processed data in hdf file ...")
hdf = h5py.File(cfg.output.data+tag+".h5", "w")
group = hdf.create_group("processed_data")
group.create_dataset("Ekin", data=evs, dtype=float)
group.create_dataset("Ephot", data=imgOdd.index.values, dtype=float)
group.create_dataset("unpumped_nexafs", data=imgOdd.values, dtype=float)
group.create_dataset("pumped_nexafs", data=imgEven.values, dtype=float)
group.create_dataset("averaged_nexafs", data=aver.values, dtype=float)
group.create_dataset("difference_nexafs", data=diff.values, dtype=float)
group.create_dataset("unpumped_roi_sum", data=iOdd_sum.values, dtype=float)
group.create_dataset("pumped_roi_sum", data=iEven_sum.values, dtype=float)
group.create_dataset("average_roi_sum", data=aver_sum.values, dtype=float)
group.create_dataset("difference_roi_sum", data=diff_sum.values, dtype=float)
group.create_dataset("Gmd_even_mean", data=gmdBinEven.mean().values, dtype=float)
group.create_dataset("Gmd_even_std", data=gmdBinEven.std().values, dtype=float)
group.create_dataset("Gmd_odd_mean", data=gmdBinOdd.mean().values, dtype=float)
group.create_dataset("Gmd_odd_std", data=gmdBinOdd.std().values, dtype=float)
group.create_dataset("UV_even_mean", data=uvBinEven.mean().values, dtype=float)
group.create_dataset("UV_even_std", data=uvBinEven.std().values, dtype=float)
group.create_dataset("UV_odd_mean", data=uvBinOdd.mean().values, dtype=float)
group.create_dataset("UV_odd_std", data=uvBinOdd.std().values, dtype=float)
hdf.close()


# plot pumped and unpumed nexafs maps in one figure for comparison
print("Plot stuff ...")
f1, ax1 = plt.subplots(1, 2, sharey=True, figsize=(10,4))
cm11 = ax1[0].pcolormesh(imgOdd.columns.values, imgOdd.index.values, imgOdd.values, cmap="cividis", vmax=0, vmin=-300)
cm12 = ax1[1].pcolormesh(imgEven.columns.values, imgEven.index.values, imgEven.values, cmap="cividis", vmax=0, vmin=-300)
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
cm3 = ax3[1].pcolormesh(diff.columns.values, diff.index.values, diff.values, cmap="bwr", vmin=-diff.values.max(), vmax=diff.values.max())
cb3 = plt.colorbar(cm3)
ax3[1].set_xlabel("Kinetic Energy (eV)")
cb3.set_label("GMD Normalised Signal")
plt.tight_layout()
plt.savefig(cfg.output.img+tag+"_diff.png", dpi=150)
plt.close(f3)

# plot roi sums of unpumped and pumped
f4 = plt.figure()
plt.plot(iOdd_sum.index.values, iOdd_sum.values, ".", label="unpumped")
plt.plot(iEven_sum.index.values, iEven_sum.values, ".", label="pumped")
plt.xlabel("Photon Energy (eV)")
plt.ylabel("Summed Signal (ROI)")
plt.legend()
plt.tight_layout()
plt.savefig(cfg.output.img+tag+"_roi_comp.png", dpi=150)
plt.close(f4)
print("Done :-)")

