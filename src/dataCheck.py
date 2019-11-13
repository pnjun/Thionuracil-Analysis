#!/usr/bin/python3
"""
Script plots several parameters for given time window and filters as time series or histogram.

Parameters plotted:
- OPIS vs undulator
- waveplate
- retarder
- UV power for pumped shots
- GMD
- BAM and corrected delays
- binning of GMD and UV data of pumped shots by (BAM corrected) delays

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
                             'stop'   : datetime(2019,3,27,7,8,0).timestamp()
                          },
           'filters'     : { 'opisEV'    : (271,274),
                             'waveplate' : (9.0, 11.0),
                             'retarder'  : (-81,-79),
                             'delay'     : (1170, 1180)
                           },
           'plots'       : { 'retarder' : False,  # check retarder values
                             'opis'     : False,  # check opis & undulator values
                             'uv'       : False,  # check uv values
                             'gmd'      : False,  # check gmd values
                             'bam'      : True,  # check bam values
                             'delayBin' : True  # bin gmd and uv data by delay; if bam is True, the delays will be corrected by bam values
                           },
           't0'          : 1177.0,  # time zero, relevant if delayBin is True
      }

cfg = AttrDict(cfg)

print("Load index ...", end="\r")
idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
pulses = idx.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')
if cfg.filters:
    pulses = filterPulses(pulses, cfg.filters)
pulsesLims = (pulses.index[0], pulses.index[-1])
idx.close()

if cfg.plots.retarder:
    print("Retardation value:")
    print(f"mean: {pulses.retarder.mean()} +- {pulses.retarder.std()}")
    print(f"min: {pulses.retarder.min()}, max: {pulses.retarder.max()}")

    print("Plot retarder values...", end="\r")
    f1 = plt.figure()
    plt.plot(pulses.time, pulses.retarder)
    plt.xlabel("time")
    plt.ylabel("retardation voltage (V)")
    plt.tight_layout()

if cfg.plots.opis:
    diff = pulses.opisEV - pulses.undulatorEV
    print("Difference between OPIS readout and undulator setpoint:")
    print(f"{diff.mean()} +- {diff.std()}")
    if diff.std() > 1.:
        print("-> There are significant, inhomogeneous differences!")
    else:
        print("-> There might only be an offset.")

    print("Plot OPIS and undulator values...", end="\r")
    f2 = plt.figure()
    plt.plot(pulses.time, pulses.opisEV, label="opis")
    plt.plot(pulses.time, pulses.undulatorEV, label="undulator")
    plt.xlabel("time")
    plt.ylabel("photon energy (eV)")
    plt.legend()
    plt.tight_layout()

print("Load shotsData ...", end="\r")
tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')
shotsData = tr.select('shotsData', where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]',
                                          'pulseId in pulses.index'] )
tr.close()

pulses = pulses.drop( pulses.index.difference(shotsData.index.levels[0]) )

if cfg.plots.bam:
    print("Correcting delays ...", end="\r")
    bamNaN = shotsData.BAM.isna().sum()
    shotsData.BAM = shotsData.BAM.fillna(0)
    averageBamShift = shotsData.query("BAM != 0").BAM.mean()
    shotsData['delay'] = shotsDelay(pulses.delay.to_numpy(), shotsData.BAM.to_numpy())
    print(f"BAM column with {shotsData.shape[0]} entries includes {bamNaN} NaN values.")
    print(f"Average shift is {averageBamShift:.3f} ps.")

    print("Filter data ...", end="\r")
    shotsData = shotsData.query("GMD > 0.5 & uvPow < 1000 & BAM != 0")

    print("Plot BAM data ...", end="\r")
    f3, ax3 = plt.subplots(2, 2, figsize=(9,6))
    bam = shotsData.pivot_table(index="pulseId", columns="shotNum", values="BAM")

    ax3[0, 0].plot(bam.mean(axis=1), label="MB mean")
    ax3[0, 0].plot(bam.mean(axis=1).rolling(window=100).mean(), label="100 MB mean")
    ax3[0, 0].set_xlabel("Macrobunch")
    ax3[0, 0].set_ylabel("Average BAM value (ps)")
    ax3[0, 0].legend()

    ax3[0, 1].errorbar(bam.columns.values, bam.mean(axis=0).values, fmt=".", yerr=bam.std(axis=0).values)
    ax3[0, 1].set_xlabel("Pulse number")
    ax3[0, 1].set_ylabel("Average BAM value (ps)")

    shotsData.hist("BAM", ax=ax3[1, 0], bins=200)
    ax3[1, 0].set_xlabel("BAM data (ps)")
    ax3[1, 0].set_ylabel("Counts")
    ax3[1, 0].set_title("")

    shotsData.delay = (shotsData.delay - cfg.t0)*-1
    shotsData.hist("delay", ax=ax3[1, 1], bins=1000 )
    ax3[1, 1].set_xlabel("Corrected Delays (ps)")
    ax3[1, 1].set_title("")

    plt.tight_layout()

else:
    print("Filter data ...", end="\r")
    shotsNum = len(shotsData.index.levels[1])
    shotsData['delay'] = shotsDelay(pulses.delay.to_numpy(), shotsNum = shotsNum)
    averageBamShift = np.float32(0.)
    shotsData = shotsData.query("GMD > 0.5 & uvPow < 1000") # comment out if you want all data (including possible outlayer)


if cfg.plots.uv:
    sdEven = shotsData.query("shotNum % 2 == 0")
    sdOdd  = shotsData.query("shotNum % 2 == 1")

    uvEven = sdEven["uvPow"]
    uvOdd  = sdOdd["uvPow"]

    print("UV diode signal analysis:")
    print(f"Even shots: {uvEven.mean()} +- {uvEven.std()}")
    print(f"Odd shots: {uvOdd.mean()} +- {uvOdd.std()}")

    print("Plot UV data ...", end="\r")
    f4, ax4 = plt.subplots(1, 2, figsize=(9,4))



    if uvEven.mean() > uvOdd.mean():
        uvEven.hist(ax=ax4[0], bins=1000)
        f4.suptitle("Even shots")
        uvEven = sdEven.pivot_table(index="pulseId", columns="shotNum", values="uvPow")
        ax4[1].errorbar(uvEven.columns.values, uvEven.mean().values, fmt=".", yerr=uvEven.std().values)
        ax4[1].plot(uvEven.columns.values, uvEven.median().values, ".")
    else:
        uvOdd.hist(ax=ax4[0], bins=1000)
        f4.suptitle("Odd shots")
        uvOdd = sdOdd.pivot_table(index="pulseId", columns="shotNum", values="uvPow")
        ax4[1].errorbar(uvOdd.columns.values, uvOdd.mean().values, fmt=".", yerr=uvOdd.std().values)
        ax4[1].plot(uvOdd.columns.values, uvOdd.median().values, ".")

    ax4[0].set_xlabel("uv power")
    ax4[0].set_ylabel("counts")
    ax4[1].set_xlabel("pulse number")
    ax4[1].set_ylabel("uv power")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

if cfg.plots.gmd:

    print("Plot GMD data ...", end="\r")
    f5, ax5 = plt.subplots(1, 2, figsize=(9,4))
    shotsData.hist("GMD", ax=ax5[0], bins=250)
    ax5[0].set_xlabel("GMD value")
    ax5[0].set_ylabel("Shots")
    ax5[0].set_xlim(-1.5, 11.5)
    ax5[0].set_title("")
    plt.tight_layout()

    gmdData = shotsData.pivot_table(index="pulseId", columns="shotNum", values="GMD")
    ax5[1].errorbar(gmdData.columns.values, gmdData.mean(), fmt=".", yerr=gmdData.std(), label="Average")
    ax5[1].plot(gmdData.columns.values, gmdData.median(), ".", label="Median")
    ax5[1].set_xlabel("Pulse number")
    ax5[1].set_ylabel("Average energy (µJ)")
    plt.tight_layout()

def stats(dataset, delayBins, param, hist=False, paramBins=None):

    dataBin = dataset[param].groupby( pd.cut(dataset.delay, delayBins) )
    dataStat = pd.DataFrame(data=np.zeros( (delayBins.size, 4) ), index=delayBins, columns=["counts", "median", "mean", "std"])
    dataStat["mean"] += dataBin.mean().values
    dataStat["std"] += dataBin.std().values
    dataStat["median"] += dataBin.median().values
    dataStat["counts"] += dataBin.count().values

    if hist:
        dataHist = np.zeros( (interval.size, paramBins.size-1) )
        for count, group in enumerate(dataBin):
            hist, _ = np.histogram(group[1].values, bins=paramBins)
            dataHist[count] += hist
        return dataStat, dataHist
    else:
        return dataStat


if cfg.plots.delayBin:
  
    print("Set up delay bins ...", end="\r")

    """
    bin_right = [1180,1179.8,1179.6,1179.4,1179.2,1179,1178.975,1178.95,1178.925,1178.9,
                 1178.875,1178.85,1178.825,1178.8,1178.775,1178.75,1178.725,1178.7,1178.675,1178.65,
                 1178.625,1178.6,1178.575,1178.55,1178.525,1178.5,1178.475,1178.45,1178.425,1178.4,
                 1178.375,1178.35,1178.325,1178.3,1178.275,1178.25,1178.225,1178.2,1178.175,1178.15,
                 1178.125,1178.1,1178.075,1178.05,1178.025,1178,1177.975,1177.95,1177.925,1177.9,
                 1177.65,1177.4,1177.15,1176.9,1176.65,1176.4,1176.15,1175.9,1175.65,1175.4,
                 1175.15,1174.9,1173.9,1172.9,1171.9,1170.9,1169.9,1168.9,1160,1130,1080
                ] # right bounds
    bin_left = [1179.8,1179.6,1179.4,1179.2,1179,1178.975,1178.95,1178.925,1178.9,1178.875,
                1178.85,1178.825,1178.8,1178.775,1178.75,1178.725,1178.7,1178.675,1178.65,1178.625,
                1178.6,1178.575,1178.55,1178.525,1178.5,1178.475,1178.45,1178.425,1178.4,1178.375,
                1178.35,1178.325,1178.3,1178.275,1178.25,1178.225,1178.2,1178.175,1178.15,1178.125,
                1178.1,1178.075,1178.05,1178.025,1178,1177.975,1177.95,1177.925,1177.9,1177.65,
                1177.4,1177.15,1176.9,1176.65,1176.4,1176.15,1175.9,1175.65,1175.4,1175.15,1174.9,
                1173.9,1172.9,1171.9,1170.9,1169.9,1168.9,1167.9,1155,1125,1075
               ] # left bounds
    bin_left += averageBamShift
    bin_right += averageBamShift

    interval = pd.IntervalIndex.from_arrays(bin_left, bin_right)
    """
    interval = pd.interval_range(start=-2, end=9, freq=0.01)
    delays = np.array([i.mid for i in interval])#(np.array([i.mid for i in interval]) - cfg.t0) * -1

    try:
        sdEven
    except NameError:
        sdEven = shotsData.query("shotNum % 2 == 0")
        sdOdd = shotsData.query("shotNum % 2 == 1")

    print("Binning GMD data ...", end="\r")
#    gmdBin = np.arange(0, 10.1, 0.1)
    gmdStatsEven = stats(sdEven, interval, "GMD")
    gmdStatsOdd = stats(sdOdd, interval, "GMD")

    print("Binning UV data ...", end="\r")
    uvStatsEven = stats(sdEven, interval, "uvPow")
    uvStatsOdd = stats(sdOdd, interval, "uvPow")

    if (gmdStatsEven["counts"].all() != uvStatsEven["counts"].all()) or (gmdStatsOdd["counts"].all() != uvStatsOdd["counts"].all()):
        print("Something's wrong with binning since gmd and uv counts are not the same. Exiting.")
        exit()

    f6, ax6 = plt.subplots(1, 3, figsize=(10, 4))

    if uvStatsEven["mean"].mean() > uvStatsOdd["mean"].mean():

        f6.suptitle("Even shots")
        ax6[0].bar(delays, gmdStatsEven["counts"].values, width=0.01)
        ax6[1].plot(delays, gmdStatsEven["mean"].values, ".")
        ax6[2].plot(delays, uvStatsEven["mean"].values, ".")

    else:

        f6.suptitle("Odd shots")
        ax6[0].bar(delays, gmdStatsOdd["counts"].values, width=0.01)
        ax6[1].plot(delays, gmdStatsOdd["mean"].values, ".")
        ax6[2].plot(delays, uvStatsOdd["mean"].values, ".")

    ax6[0].set_ylabel("Shots per bin")
    ax6[1].set_ylabel("Average GMD value")
    ax6[2].set_ylabel("Average UV power value")
    ax6[0].set_xlabel("Delay (ps)")
    ax6[1].set_xlabel("Delay (ps)")
    ax6[2].set_xlabel("Delay (ps)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


plt.show()
