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
                          'trace'    : 'third_block.h5'
                        },
           'time'     : { 'start' : datetime(2019,4,6,16,2,0).timestamp(),
                          'stop'  : datetime(2019,4,6,18,50,0).timestamp()
                        },
           'filters'  : { 'undulatorEV' : (160,180), # replace with opisEV if opis column contains reasonable values, else use undulatorEV
                          'delay'       : (1261.35,1261.45), # comment out if there is only non-UV data in the block
                          'retarder'    : (-11,-9)
                        },
           'output'   : { 'img'  : './Plots/',  # where output images should be stored
                          'data' : './data/'    # where processed data should be stored
                        },
           'UV'       : True,  # True, if you want to distinguish between even (pumped) and odd (unpumped) shots in a macrobunch
           'ROI'      : "all",  # eV region to be summed, either tuple or all
           'PLC'      : False,  # photoline correction in region of interest
           'evStep'   : 1.5,  # frequency for binning
           'ioChunkSize' : 50000
      }

cfg = AttrDict(cfg)

timetag = "S2p_+200fs_02"#'{0}-{1}'.format(datetime.fromtimestamp(cfg.time.start).isoformat(),
          #                 datetime.fromtimestamp(cfg.time.stop).isoformat())

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
#intervals = pd.interval_range(start=evLim[0], end=evLim[1], freq=cfg.evStep)
left = [161, 162.5, 163.5, 164.5, 165.25, 166, 166.75, 167.5, 169.0, 170.5, 172]
right = [162, 163.5, 164.5, 165, 165.75, 166.5, 167.25, 168.5, 170, 171.5, 173]
intervals = pd.IntervalIndex.from_arrays(left, right)
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

# calulate eVs
evConv = mainTofEvConv(pulses.retarder.mean())
evs = evConv(tofs.to_numpy(dtype=np.float64))

# make proper pandas dataframe
imgWUV = pd.DataFrame(data=imgWUV, index=opisBin.values, columns=evs, dtype="float64").dropna()
imgNUV = pd.DataFrame(data=imgNUV, index=opisBin.values, columns=evs, dtype="float64").dropna()

# save this dataframe to text file
imgWUV.to_csv(cfg.output.data + "nexafs_{0}_wuv.csv".format(timetag), header=True, index=True , mode="w")
imgNUV.to_csv(cfg.output.data + "nexafs_{0}_nuv.csv".format(timetag), header=True, index=True , mode="w")

# preparing stuff for plotting
# changing the offset so that data has same sign everywhere

d = np.array([i - i.max() for i in imgWUV.values])
imgWUV = pd.DataFrame(data=d, columns=imgWUV.columns, index=imgWUV.index)
d = np.array([i - i.max() for i in imgNUV.values])
imgNUV = pd.DataFrame(data=d, columns=imgNUV.columns, index=imgNUV.index)
del d

def plot_stuff(dataframe, roi, outfolder, tag, plc=False):

    f, ax = plt.subplots(1, 2, sharey=True, figsize=(10,4), gridspec_kw={"width_ratios":[2, 3]})

    cm = ax[1].pcolormesh(dataframe.columns.values, dataframe.index.values, dataframe.values, cmap="cividis")
    cb = plt.colorbar(cm)
    ax[1].set_xlabel("Kinetic Energy (eV)")
    cb.set_label("GMD Normalised Signal")

    if roi == "all":
        ax[0].plot(dataframe.T.sum()/1000, dataframe.index, ".")
        ax[0].set_xlabel('Summed Counts (x1000)')
    else:
        mask = (dataframe == dataframe) & (dataframe.columns >= roi[0]) & (dataframe.columns <= roi[1])
        if plc:
            d = dataframe[mask]
            d_max = [ i.argmin() for i in dataframe.values]
            d_pl = [ dataframe.values[i][d_max[i]-25:d_max[i]+25].sum() for i in range(len(dataframe.values))]
            d = d.T.dropna().T
            ax[0].plot(d.T.sum()/1000, d.index, ".", label="Raw sum")
            ax[0].plot((d.T.sum() - d_pl)/1000, roi.index, ".", label="PL corrected")
            ax[0].legend(loc=9)
        else:
            d = dataframe[mask].T.dropna().T
            ax[0].plot(d.T.sum()/1000, d.index, ".", label="Raw sum")

        ax[0].set_xlabel('Summed Counts over ROI {0}-{1} eV (x1000)'.format(roi[0], roi[1]))

    ax[0].set_ylabel('Photon Energy (eV)')

    plt.tight_layout()
    plt.savefig(outfolder + "nexafs_{0}.png".format(tag), dpi=300)
    #plt.show()
    plt.close("all")


plot_stuff(imgWUV, cfg.ROI, cfg.output.img, timetag+"_wuv")
plot_stuff(imgNUV, cfg.ROI, cfg.output.img, timetag+"_nuv")
aver = pd.DataFrame((imgWUV.values + imgNUV.values) / 2, index=imgWUV.index, columns=imgWUV.columns)
plot_stuff(aver, cfg.ROI, cfg.output.img, timetag+"_aver")
diff = pd.DataFrame(imgWUV.values - imgNUV.values, index=imgWUV.index, columns=imgWUV.columns)
plot_stuff(diff, cfg.ROI, cfg.output.img, timetag+"_diff")

