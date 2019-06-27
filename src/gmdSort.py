#!/usr/bin/3.6
'''
Contains functions to (sort,) bin and normalise TOF signal to FEL pulse energy or UV laser power.
'''
from attrdict import AttrDict
import time
import datetime
import pandas as pd
import numpy as np

cfg = { 'data'   : { 'path'   : '/media/Data/Beamtime/processed/',
                     'index'  : 'index.h5',
                     'trace' : 'first_block.h5'},
        'hdf'    : { 'pulses' : '/pulses',
                     'time'   : 'time',
                     'param'    : 'uvPow',
                     'tof'    : '/shotsTof',
                     'photon' : '/shotsData'},
        'time'   : { 'start'  : '26.03.19 04:42:00',
                     'stop'   : '26.03.19 05:00:00'}}

cfg = AttrDict(cfg)


def laserIntBin(tofData, laserData, bins=10):
    """Bins ToF traces by GMD values within a macrobunch.

    Keyword arguments:
    tofData -- standard tof dataframe from hdf5 files
    laserData -- standard GMD or UV laser power dataframe from hdf5 files
    bins -- integer (number of bins) or pandas IntervalIndex object

    Output:
    binnedTof -- pandas Dataframe with multindex (pulseId, bins), ToF traces are summed for each bin
    binnedLaser -- pandas Dataframe with multindex (pulseId, bins), includes mean and sum for each bin
    """
    if type(bins) == int:
        # calculate extrema over given dataframe and create interval range
        laserMax = np.ceil(laserData.max())
        laserMin = np.floor(laserData.min())
        bins = pd.interval_range(start=laserMin, end=laserMax, periods=bins)
    elif type(bins) != pd.core.indexes.interval.IntervalIndex:
        raise TypeError("You should use an integer or pandas IntervalIndex object for bins!")

    print("Binning ToF traces:")
    i = 0
    binnedTof = [] # list for storing binned tofs
    binnedLaserSum = [] # list for storing binned gmds
    binnedLaserMean = []
    for bunch in tofData.index.levels[0].values:
        i += 1
        print("Bunch: {0:02d}/{1}".format(i, tofData.index.levels[0].values.size), end="\r")
        # define local variables with data (turns out this is faster than doing it in one line...)
        pulseTofData = tofData.loc[bunch]
        pulseLaserData = laserData.loc[bunch]
        # sort indices from GMD data according to intervals
        laserBinned = pd.cut(pulseLaserData, bins)
        # group actual data using sorted indices
        pulseLaserData = pulseLaserData.groupby(laserBinned)
        pulseTofData = pulseTofData.groupby(laserBinned)
        # append to lists
        for a in pulseTofData.sum().values:
            binnedTof.append(a)
        for b in pulseLaserData.sum().values:
            binnedLaserSum.append(b)
        for c in pulseLaserData.mean().values:
            binnedLaserMean.append(c)

    # create new MultiIndex including bunch
    index = []
    for i in tofData.index.levels[0].values:
        for j in bins:
            index.append([i, j])
    index = pd.MultiIndex.from_arrays(np.array(index).transpose(), names=["pulseId", "bins"])

    # create new dataframes with binned data
    binnedTof = pd.DataFrame(data=np.array(binnedTof), index=index)
    binnedLaser = pd.DataFrame(data=np.array([binnedLaserSum, binnedLaserMean]).transpose(), index=index, columns=["Sum", "Mean"])
    print("")
    print("Done. Returning binned data.")

    return binnedTof, binnedLaser

def laserIntNorm(tofData, laserData):
    """Divide each ToF trace of a pulse within a macrobunch by its GMD value and returns the normalised tof dataframe.

    Keyword arguments:
    tofData -- standard tof dataframe from hdf5 files
    laserData -- standard gmd dataframe from hdf5 files

    Output:
    normedTof -- dataframe similar to tofData but normalised on GMD data and no column names
    """
    print("Normalize Tof traces:")
    i = 0
    normedTof = []
    for bunch in tofData.index.levels[0].values: # loop over bunches
        i += 1
        for pulse in tofData.index.levels[1].values: # loop over pulses
            print("Bunch: {0:02d}/{1}, Pulse: {2:02d}/50".format(i, tofData.index.levels[0].values.size, pulse+1), end="\r")
            # define local variables with data (turns out this is faster than doing it in one line...)
            pulseTofData = tofData.loc[bunch].loc[pulse]
            pulselaserData = laserData.loc[bunch].loc[pulse]
            # divide ToF data by GMD
            pulseTofData = pulseTofData / pulselaserData
            # store new ToF in list
            normedTof.append(pulseTofData.values)

    print("")
    # Create new dataframe
    tofData = pd.DataFrame(data = np.array(normedTof), index=tofData.index)
    print("Done. Returning normalized dataframe.")
    # return stuff
    return tofData


def laserIntSort(tofData, laserData):
    """Sorts ToF traces by GMD values within a macrobunch.

    Keyword arguments:
    tofData -- standard tof dataframe from hdf5 files
    laserData -- standard gmd dataframe from hdf5 files
    """
    pass


if __name__ == '__main__':

    start = int(time.mktime(datetime.datetime.strptime(cfg.time.start, '%d.%m.%y %H:%M:%S').timetuple()))
    stop  = int(time.mktime(datetime.datetime.strptime(cfg.time.stop, '%d.%m.%y %H:%M:%S').timetuple()))

    idx = pd.HDFStore(cfg.data.path + cfg.data.index, 'r')
    tr  = pd.HDFStore(cfg.data.path + cfg.data.trace, 'r')
    pulse = idx.select('pulses', where='time >= start and time < stop')
    param = tr.select('shotsData', where='pulseId >= pulse.index[0] and pulseId <= pulse.index[-1]')[cfg.hdf.param]
    data = tr.select('shotsTof', where='pulseId >= pulse.index[0] and pulseId <= pulse.index[-1]')
    idx.close()
    tr.close()

    #data = laserIntNorm(data,param)
    intervals = pd.interval_range(start=10, end=60, periods=10)
    data, param = laserIntBin(data, param, bins=intervals)
