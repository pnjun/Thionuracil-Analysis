#!/usr/bin/3.6
'''
Contains functions to sort, bin and normalise TOF signal to FEL pulse energy

Technically, it should also be applicable to UV power and maybe other parameters.
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
                     'gmd'    : 'Uv laser power',
                     'tof'    : '/shotsTof',
                     'photon' : '/shotsData'},
        'time'   : { 'start'  : '25.03.19 15:08:00',
                     'stop'   : '25.03.19 15:08:10'}}

cfg = AttrDict(cfg)


def laserIntBin(tofData, laserData, bins=10):
    """Bins ToF traces by GMD values within a macrobunch.

    Keyword arguments:
    tofData -- standard tof dataframe from hdf5 files
    laserData -- standard GMD or UV laser power dataframe from hdf5 files
    bins -- integer, number of bins

    Output:
    binnedTof -- pandas Dataframe with multindex (pulseId, bins), ToF traces are summed for each bin
    binnedLaser -- pandas Dataframe with multindex (pulseId, bins), includes mean and sum for each bin
    """
    # calculate extrema over given dataframe and create interval range
    laserMax = np.ceil(laserData.max())
    laserMin = np.floor(laserData.min())
    intervals = pd.interval_range(start=laserMin, end=laserMax, periods=bins)

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
        laserBinned = pd.cut(pulseLaserData, intervals)
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
        for j in intervals:
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
    print("Normalize Tof data on GMD:")
    i = 0
    normedTof = [] # list for storing results
    for bunch in tofData.index.levels[0].values: # loop over bunches
        i += 1
        for pulse in tofData.index.levels[1].values: # loop over pulses
            print("Bunch: {0:02d}/{1}, Pulse: {2:02d}/50".format(i, tofData.index.levels[0].values.size, pulse+1), end="\r")
            # define local variables with data (turns out this is faster than doing it in one line...)
            pulseTofData = tofData.loc[bunch].loc[pulse]
            pulselaserData = laserData.loc[bunch].loc[pulse]
            # divide ToF data by GMD
            pulseTofData = pulseTofData.div(pulselaserData)
            # store new ToF in list
            normedTof.append(pulseTofData.values)
    
    print("")
    # Create new dataframe
    normedTof = pd.DataFrame(data = normedTof, index=tofData.index)
    print("Done. Returning normalized dataframe.")
    # return stuff
    return normedTof


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
    gmd = tr.select('shotsData', where='pulseId >= pulse.index[0] and pulseId <= pulse.index[-1]')['GMD']
    data = tr.select('shotsTof', where='pulseId >= pulse.index[0] and pulseId <= pulse.index[-1]')
    idx.close()
    tr.close()

    #dataNormed = laserNorm(data,gmd)
    dataBinned, gmdBinned = laserIntBin(data, gmd, bins=10)
