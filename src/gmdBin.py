#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime
from attrdict import AttrDict

from utils import mainTofEvConv
from utils import filterPulses

from numba import cuda
import cupy as cp

import matplotlib
matplotlib.use("GTK3Agg")

cfg = { 'data'    : { 'path'   : '/media/Fast1/ThioUr/processed/',
                      'index'  : 'index.h5',
                      'trace'  : 'first_block.h5'},
        'hdf'     : { 'pulses' : '/pulses',
                      'time'   : 'time',
                      'param'  : 'GMD',
                      'tof'    : '/shotsTof',
                      'photon' : '/shotsData'},
        'time'    : { 'start' : datetime(2019,3,25,21,27,0).timestamp(),
                      'stop'  : datetime(2019,3,25,23,38,0).timestamp()},
        'filters' : { 'opisEV' : (268,274),
                      'retarder'    : (-81,-79)},
        'ioChunkSize' : 25000
      }

cfg = AttrDict(cfg)


def binToGmd(tofData, laserData, bins=10):
    """Bins ToF traces by GMD/UV power values.

    Keyword arguments:
    tofData -- standard tof dataframe from hdf5 files
    laserData -- standard GMD or UV laser power dataframe from hdf5 files
    bins -- integer (number of bins) or pandas IntervalIndex object

    Output:
    tofBinned -- pandas Dataframe with bins as index, columns remain the same, tof traces are averaged for each bin
    """
    if type(bins) == int:
        # calculate extrema over given dataframe and create interval range
        laserMax = np.ceil(laserData.max())
        laserMin = np.floor(laserData.min())
        bins = pd.interval_range(start=laserMin, end=laserMax, periods=bins)
    elif type(bins) != pd.core.indexes.interval.IntervalIndex:
        raise TypeError("You should use an integer or pandas IntervalIndex object for bins!")

    ldBinned = pd.cut(laserData, bins)
    tofBinned = pd.DataFrame(data=tofData.groupby(ldBinned).mean().to_numpy(), index=bins, columns=tofData.columns)

    return tofBinned


def normToGmd(tofTrace, gmd):
    """Divide each ToF trace by its xray or uv power value and returns the normalised tof dataframe.

    Keyword arguments:
    tofData -- standard tof dataframe from hdf5 files
    gmd -- standard gmd dataframe from hdf5 files

    Output:
    Normalised tofTraces
    """
    @cuda.jit
    def normTofToGmd(tof, gmd):
        row = cuda.blockIdx.x
        col = cuda.blockIdx.y*cuda.blockDim.x + cuda.threadIdx.x
        tof[ row    , col ]  /= gmd[row]

    #Get difference data
    tof = cp.array(tofTrace.to_numpy())
    cuGmd = cp.array(gmd.reindex(tofTrace.index).to_numpy())
    normTofToGmd[ (tof.shape[0] , tof.shape[1] // 250) , 250 ](tof, cuGmd)

    return pd.DataFrame( tof.get(), index = tofTrace.index, columns=tofTrace.columns)


def do_sum_stuff(data, indices, lim=None, fit=1):

    s = np.array([i.sum() for i in data.to_numpy()[:,indices[0]:indices[1]]]) / (data.columns.values[indices[1]] - data.columns.values[indices[0]])
    r = pd.DataFrame(data=s, index=data.index.values)
    #if lim:
        #ind = r.where(r.index.to_series() < lim).dropna().index
        #p = np.polyfit(ind, r.loc[ind], 1)#, cov=True)
        #c = np.sqrt(np.diag(np.array([[i[0] for i in j] for j in c])))
    #else:
        #p = np.polyfit(r.index, r, 1)#, cov=True)
        #c = np.sqrt(np.diag(np.array([[i[0] for i in j] for j in c])) cuda)

    return r#, p, ind


if __name__ == '__main__':

    idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
    tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')
    pulses = idx.select('pulses', where='time >= cfg.time.start and time < cfg.time.stop')
    pulsesLims = (pulses.index[0], pulses.index[-1])
    #Filter only pulses with parameters in range
    pulses = filterPulses(pulses, cfg.filters)
    idx.close()

    #Get corresponing shots
    if(len(pulses) == 0):
        print("No pulses satisfy filter condition. Exiting\n")
        exit()
    shotsData = tr.select('shotsData', where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]', 'pulseId in pulses.index'] )[cfg.hdf.param]
    #Remove pulses with no corresponing shots
    pulses = pulses.drop( pulses.index.difference(shotsData.index.levels[0]) )

    # some analysis on gmd
    print("do some stuff with GMD data.")
    shotsMean = np.array([shotsData.loc[bunch].mean() for bunch in shotsData.index.levels[0].values])

    f1, ax1 = plt.subplots(1,3, figsize=(13,5))
    #plt.title("GMD data from\n{0} - {1}".format(datetime.fromtimestamp(cfg.time.start).isoformat(), datetime.fromtimestamp(cfg.time.stop).isoformat()))
    ax1[0].plot(pulses.index, shotsMean,lw=1)
    ax1[0].set_xlabel("macrobunch number")
    ax1[0].set_ylabel("x-ray intensity (mean over macrobunch) (µJ)")
    #ax11 = ax1[0].twinx()
    #ax11.plot(pulses.index, pulses.opisEV, "C1", lw=1)
    #ax11.set_ylabel("OPIS (eV)")
    shotsData.hist(bins=125, ax=ax1[1])
    ax1[1].set_xlabel("X-ray intensity (µJ)")
    ax1[1].set_ylabel("# of pulses")
    ax1[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    shotsBunch = np.array([shotsData.loc[:, pulse].mean() for pulse in range(50)])
    ax1[2].plot(np.linspace(1,51, 50), shotsBunch)
    ax1[2].set_xlabel("pulse number")
    ax1[2].set_ylabel("mean x-ray intensity (µJ)")
    plt.tight_layout()
    print("done with that stuff.")

    # go for tof data
    intervals = pd.interval_range(start=0, end=125, freq=1.) # bins for gmd
    shotsTof  = tr.select('shotsTof', where=['pulseId >= pulsesLims[0] and pulseId < pulsesLims[1]',
                                             'pulseId in pulses.index'],
                                      iterator=True, chunksize=cfg.ioChunkSize)

    img = np.zeros( ( len(intervals), 3000 ))
    binCount = np.zeros( len(intervals) )
    for counter, chunk in enumerate(shotsTof):
        c = 0
        print( f"processing chunk {counter}", end='\r' )
        idx = chunk.index.levels[0].values
        shotsnorm = normToGmd(chunk, shotsData.loc[idx]) # normalise chunk on corresponding gmd values
        shotsBins = pd.cut(shotsData.loc[idx], intervals) # sort gmd data according to intervals
        tofs = pd.DataFrame(shotsnorm.groupby(shotsBins).mean().to_numpy()) # bin tof data based on sorted gmd data
        for group in range(tofs.index.size):
            if not tofs.loc[group].isnull().to_numpy().any():
                img[c] += tofs.loc[group].to_numpy()
                binCount[c] += 1
            c += 1

    img /= binCount[:,None]
    bg = np.array([np.mean(i[-50:-1]) for i in img])
    img -= bg[:,None]

    gmdBins = shotsData.groupby(pd.cut(shotsData, intervals)).mean()
    tofs = tr.select("shotsTof", where=["pulseId == pulsesLims[0]", "pulseId in pulses.index"]).columns
    tr.close()

    evConv = mainTofEvConv(pulses.retarder.mean())
    evs = evConv(tofs)

    img = pd.DataFrame(data=img, index=gmdBins.values, columns=evs, dtype="float64").dropna()
    cmax = np.max(abs(img.values))*0.5

    r1 = do_sum_stuff(img, [70,350], lim=60.)
    r2 = do_sum_stuff(img, [360,780], lim=50.)
    r3 = do_sum_stuff(img, [790,1100], lim=40.)
    f2, ax2 = plt.subplots(1,2, figsize=(9,4))

    ax2[0].pcolormesh(img.columns.values, img.index.values,img, cmap="bwr", vmax=cmax, vmin=-cmax)
    ax2[0].set_xlabel("kinetic energy (ev)")
    ax2[0].set_ylabel("GMD (µJ)")
    ax2[1].plot(r1.index.values, r1*15, "b.", label="1 (x15)")
    ax2[1].plot(r2.index.values, r2*2.5, "r.", label="2 (x2.5)")
    ax2[1].plot(r3.index.values, r3, "g.", label="3")
    #ax2[1].plot(ind1, np.polyval(p1, ind1), "b-")
    #ax2[1].plot(ind2, np.polyval(p2, ind2), "r-")
    #ax2[1].plot(ind3, np.polyval(p3, ind3), "g-")

    ax2[1].set_xlabel("GMD (µJ)")
    ax2[1].set_ylabel("Summed signal (arb.units)")
    ax2[1].legend()

    plt.tight_layout()
    plt.show()
