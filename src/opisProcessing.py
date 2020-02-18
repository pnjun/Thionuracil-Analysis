#!/usr/bin/python3
'''
Simlar to dataPreprocessing but for OPIS data. Slices OPIS data.

In the code macrobunch are 'pulses' and shots are 'shots'

Requires: - pandas
          - h5py
          - attrdict

For info contact Fabiano

'''

import h5py
import pandas as pd
import numpy as np
import os, glob
import uuid
from time import time
from attrdict import AttrDict
from utils import Slicer
import opisUtils as ou
from datetime import datetime
import numba

PREFIX = '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/'

#cfguration parameters:
cfg = {    'data'     : { 'path'     : '/media/Data/ThioUr/raw/',
                          'files': 'FLASH2_USER1-2019-0?-[30][0123789]*.h5'
                          #['FLASH2_USER1-2019-03-31T0700.h5'] #['FLASH2_USER1-2019-04-01T0238.h5]
                          #'FLASH2_USER1-2019-0?-[30][0123789]*.h5'
                          #['FLASH2_USER1-2019-03-31T0500.h5'] ,
                          # List of files to process. All files must have the same number of shots per macrobunch
                        },
           'output'   : {
                          'folder' : '/media/Fast2/ThioUr/processed/',
                          'fname'  :  'second_block.h5',
                          'start' : datetime(2019,3,30,20,16,0).timestamp(),
                          'stop'  : datetime(2019,3,30,21,1,0).timestamp(),
                        },
           'hdf'      : { 'opisTr0'    : PREFIX + 'Raw data/CH00',
                          'opisTr1'    : PREFIX + 'Raw data/CH01',
                          'opisTr2'    : PREFIX + 'Raw data/CH02',
                          'opisTr3'    : PREFIX + 'Raw data/CH03',
                          'ret0'       : PREFIX + 'Expert stuff/eTOF1 voltages/Ret nominal set',
                          'ret1'       : PREFIX + 'Expert stuff/eTOF2 voltages/Ret nominal set',
                          'ret2'       : PREFIX + 'Expert stuff/eTOF3 voltages/Ret nominal set',
                          'ret3'       : PREFIX + 'Expert stuff/eTOF4 voltages/Ret nominal set',
                          'gasID'      : PREFIX + 'Expert stuff/XGM.GAS_SUPPLY/FL2.TUNNEL.OPIS/GAS.TYPE.ID',
                          'undulSetWL' : '/FL2/Electron Diagnostic/Undulator setting/set wavelength',
                          'times'      : '/Timing/time stamp/fl2user1',
                        },
           'slicing0' : { 'offset'   : 263,   #Offset of first slice in samples (time zero)
                          'period'   : 3500.290,  #Rep period of FEL in samples
                          'window'   : 1700,  #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 250,     #Skip fist samples of each slice (cuts off high energy electrons)
                          'dt'       : 0.00014,     #Time between samples [us]
                          'shotsNum' : 49,    #Number of shots per macrobunch
                        },
           'slicing1' : { 'offset' : 225, 'period' : 3500.302,
                          'window' : 1700,'skipNum' : 250,
                          'dt' : 0.00014, 'shotsNum' : 49,
                        },
           'slicing2' : { 'offset'   : 220,'period'   : 3500.300,
                          'window'   : 1700, 'skipNum'  : 250,
                          'dt'       : 0.00014, 'shotsNum' : 49,
                        },
           'slicing3' : { 'offset'   : 219, 'period'   : 3500.297,
                          'window'   : 1700, 'skipNum'  : 250,
                          'dt'       : 0.00014, 'shotsNum' : 49,
                        },
           'chunkSize': 2400, #How many macrobunches to read/write at a time. Increasing increases RAM usage

           'ampliRange': np.linspace(5, 80, 12),
           'enerRange' : np.linspace(-3, 7, 32)
         }
cfg = AttrDict(cfg)

#Input file list
flist = [ cfg.data.path + fname for fname in cfg.data.files ] if isinstance(cfg.data.files, tuple) else glob.glob(cfg.data.path + cfg.data.files)
flist = sorted(flist)

#Output file
fout = pd.HDFStore(cfg.output.folder + cfg.output.fname)
try:
    del fout['opisFit'] #delete previously stored data to avoid duplicates
except (KeyError):
    pass

print(f"processing {len(flist)} files")
for fname in flist:
    with h5py.File( fname ) as dataf:
        #Check if OPIS raw data is present in file:
        try:
            dataf[cfg.hdf.opisTr0]
        except KeyError:
            print(f"No OPIS traces in {fname[-18:-3]}, continuing")
            continue
        else:
            print(f"_____________________\nFile: {fname[-18:-3]}\n¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯")

        #Initialize slicers for the 4 tofs spectrometers
        opisSlicer0 = Slicer(cfg.slicing0, removeBg = True)
        opisSlicer1 = Slicer(cfg.slicing1, removeBg = True)
        opisSlicer2 = Slicer(cfg.slicing2, removeBg = True)
        opisSlicer3 = Slicer(cfg.slicing3, removeBg = True)

        #Load macrobunch info for gas and retarder setting
        pulses = pd.DataFrame( { 'pulseId' : dataf[cfg.hdf.times][:, 2].astype('int64'),
                                 'time'    : dataf[cfg.hdf.times][:, 0],
                                 'undulEV' : 1239.84193 / dataf[cfg.hdf.undulSetWL][:, 0], #nanometers to eV
                                 'gasID'     : dataf[cfg.hdf.gasID][:, 0],
                                 'ret0'      : dataf[cfg.hdf.ret0][:, 0],
                                 'ret1'      : dataf[cfg.hdf.ret1][:, 0],
                                 'ret2'      : dataf[cfg.hdf.ret2][:, 0],
                                 'ret3'      : dataf[cfg.hdf.ret3][:, 0],
                               } , columns = [ 'pulseId', 'time', 'undulEV', 'gasID', 'ret0', 'ret1', 'ret2', 'ret3' ])
        pulses = pulses.set_index('pulseId')

        #Split up computation in cunks
        chunks = np.arange(0, dataf[cfg.hdf.opisTr1].shape[0], cfg.chunkSize)
        endtime = time()
        for i, start in enumerate(chunks):
            starttime = endtime

            sl = slice(start,start+cfg.chunkSize)

            #Time bounds check
            if 'start' in cfg.output.keys():
                if pulses[sl].time.iloc[-1] < cfg.output.start:
                    print("** skipping chunk, time is too early")
                    continue
                if pulses[sl].time.iloc[0] > cfg.output.stop:
                    print("** skipping chunk, time is too late")
                    continue

            #Sanity checks
            if pulses[sl].gasID.nunique() != 1:
                print(f"** skipping chunk, more than one gasID setting")
                continue

            if pulses[sl].gasID.isnull().values.any():
                print(f"** skipping chunk, gasID contains Nan")
                continue

            #Ev converter only handles 170V retardation for now
            if np.any( np.abs(pulses[sl].ret0 - 170) > 1 ):
                print(f"skipping chunk, Retarder {pulses[sl].ret0.mean()}")
                continue
            #TODO: ADAPT evConv to retarder setting for now only 170V
            evConv = ou.geometricEvConv(170)

            shots0 = opisSlicer0( dataf[cfg.hdf.opisTr0][sl], pulses[sl])
            shots1 = opisSlicer1( dataf[cfg.hdf.opisTr1][sl], pulses[sl])
            shots2 = opisSlicer2( dataf[cfg.hdf.opisTr2][sl], pulses[sl])
            shots3 = opisSlicer3( dataf[cfg.hdf.opisTr3][sl], pulses[sl])
            traces = [shots0, shots1, shots2, shots3]

            fitter = ou.evFitter2ElectricBoogaloo(pulses[sl].gasID.iloc[0],
                                                  evConv,traces[0].columns,
                                                  pulses.undulEV.mean())

            def fitTraces(traces, evGuess):
                fitter.loadTraces(traces)
                fit = fitter.leastSquare(evGuess)

                fitDf = pd.DataFrame(fit, index = traces[0].index,
                                         columns=['ev', 'ampli', 'fwhm'])
                fitDf['ignoreMask'] = fitter.ignoreMask()
                return fitDf

            #check if all shots have the same undulator setting
            evGuesses = pulses[sl].undulEV.unique()
            try:
                if len(evGuesses) != 1:
                    #Need a separate conversion block for each different undulator setting
                    #since we use the unulator setting as a starting guess for the ev fit
                    for evGuess in evGuesses:
                        pulseIds = pulses[sl].query('undulEV == @evGuess').index
                        newTraces = [tr.query('pulseId in @pulseIds') for tr in traces]
                        fitted = fitTraces(newTraces, evGuess)
                        fout.append('opisFit', fitted)
                else:
                    evGuess = pulses[sl].undulEV.iloc[0]
                    fitted = fitTraces(traces, evGuess)
                    fout.append('opisFit', fitted)
            except numba.cuda.cudadrv.driver.CudaAPIError:
                print(f"** skipping chunk, exception raised during fit")
                continue

            endtime = time()
            print(f"chunk {i+1:02d} of {len(chunks)} | {datetime.fromtimestamp(pulses[sl].time.iloc[0])} | avg ev {fitted.ev.mean():.1f} | speed {cfg.chunkSize / (endtime-starttime):.1f} bunch/sec ")
fout.close()

if cfg.output.fname == 'AUTO':
    newname = "OPIS-%d-%d.h5" % (pulses.index[0], pulses.index[-1])
    os.rename(cfg.output.folder + outfname, cfg.output.folder + newname  )
    print("Output renamed to: %s" % newname)
