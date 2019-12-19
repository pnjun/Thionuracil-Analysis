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
from attrdict import AttrDict
from utils import Slicer

#cfguration parameters:
cfg = {    'data'     : { 'path'     : '/media/Data/ThioUr/raw/',
                          'files': ['FLASH2_USER1-2019-03-31T1613.h5']
                          #['FLASH2_USER1-2019-03-31T0700.h5'] #['FLASH2_USER1-2019-04-01T0238.h5]
                          #'FLASH2_USER1-2019-0?-[30][0123789]*.h5'
                          #['FLASH2_USER1-2019-03-31T0500.h5'] ,
                          # List of files to process. All files must have the same number of shots per macrobunch
                        },
           'output'   : {
                          'folder' : '/media/Fast1/ThioUr/processed/',
                          # 'AUTO' for 'OPIS-<firstPulseId>-<lastPulseId.h5>'. Use only when data.files is a list of subsequent shots.
                          'fname'  :  'opistest3.h5' #second_block_opis.h5',
                        },
           'hdf'      : { 'opisTr0'    : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Raw data/CH00',
                          'opisTr1'    : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Raw data/CH01',
                          'opisTr2'    : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Raw data/CH02',
                          'opisTr3'    : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Raw data/CH03',
                          'ret0'       : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Expert stuff/eTOF1 voltages/Ret nominal set',
                          'ret1'       : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Expert stuff/eTOF2 voltages/Ret nominal set',
                          'ret2'       : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Expert stuff/eTOF3 voltages/Ret nominal set',
                          'ret3'       : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Expert stuff/eTOF4 voltages/Ret nominal set',
                          'gasID'      : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Expert stuff/XGM.GAS_SUPPLY/FL2.TUNNEL.OPIS/GAS.TYPE.ID',
                          'times'      : '/Timing/time stamp/fl2user1',
                        },
           'slicing0' : { 'offset'   : 263,   #Offset of first slice in samples (time zero)
                          'period'   : 3500.290,  #Rep period of FEL in samples
                          'window'   : 1500,  #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 250,     #Skip fist samples of each slice (cuts off high energy electrons)
                          'dt'       : 0.00014,     #Time between samples [us]
                          'shotsNum' : 49,    #Number of shots per macrobunch
                        },
           'slicing1' : { 'offset'   : 225,     #Offset of first slice in samples (time zero)
                          'period'   : 3500.302,  #Rep period of FEL in samples
                          'window'   : 1500,  #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 250,     #Skip fist samples of each slice (cuts off high energy electrons)
                          'dt'       : 0.00014,     #Time between samples [us]
                          'shotsNum' : 49,    #Number of shots per macrobunch
                        },
           'slicing2' : { 'offset'   : 220,     #Offset of first slice in samples (time zero)
                          'period'   : 3500.300,  #Rep period of FEL in samples
                          'window'   : 1500,  #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 250,     #Skip fist samples of each slice (cuts off high energy electrons)
                          'dt'       : 0.00014,     #Time between samples [us]
                          'shotsNum' : 49,    #Number of shots per macrobunch
                        },
           'slicing3' : { 'offset'   : 219,     #Offset of first slice in samples (time zero)
                          'period'   : 3500.297,  #Rep period of FEL in samples
                          'window'   : 1500,  #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 250,     #Skip fist samples of each slice (cuts off high energy electrons)
                          'dt'       : 0.00014,     #Time between samples [us]
                          'shotsNum' : 49,    #Number of shots per macrobunch
                        },
           'chunkSize': 2000 #How many macrobunches to read/write at a time. Increasing increases RAM usage (1 macrobunch is about 6.5 MB)
         }
cfg = AttrDict(cfg)



def main():
    outfname = uuid.uuid4().hex + '.temp' if cfg.output.fname == 'AUTO' else cfg.output.fname

    fout = pd.HDFStore(cfg.output.folder + outfname)#, complib = 'zlib', complevel = 1)

    flist = [ cfg.data.path + fname for fname in cfg.data.files ] if isinstance(cfg.data.files, tuple) else glob.glob(cfg.data.path + cfg.data.files)
    flist = sorted(flist)

    print(f"processing {len(flist)} files")
    for fname in flist:
        with h5py.File( fname ) as dataf:
            #Dataframe for macrobunch info
            pulses = pd.DataFrame( { 'pulseId'     : dataf[cfg.hdf.times][:, 2].astype('int64'),
                                     'time'      : dataf[cfg.hdf.times][:, 0],
                                     'gasID'     : dataf[cfg.hdf.gasID][:, 0],
                                     'ret0'      : dataf[cfg.hdf.ret0][:, 0],
                                     'ret1'      : dataf[cfg.hdf.ret1][:, 0],
                                     'ret2'      : dataf[cfg.hdf.ret2][:, 0],
                                     'ret3'      : dataf[cfg.hdf.ret3][:, 0],
                                   } , columns = [ 'pulseId', 'time', 'gasID', 'ret0', 'ret1', 'ret2', 'ret3' ])
            pulses = pulses.set_index('pulseId')

            #Check if OPIS raw data is present in file:
            try:
                dataf[cfg.hdf.opisTr0]
            except KeyError:
                print(f"No OPIS traces in {fname}, continuing")
                continue

            #Slice shot data and add it to shotsTof
            opisSlicer0 = Slicer(cfg.slicing0, removeBg = True)
            opisSlicer1 = Slicer(cfg.slicing1, removeBg = True)
            opisSlicer2 = Slicer(cfg.slicing2, removeBg = True)
            opisSlicer3 = Slicer(cfg.slicing3, removeBg = True)


            #Split up computation in cunks
            chunks = np.arange(0, dataf[cfg.hdf.opisTr1].shape[0], cfg.chunkSize)
            for i, start in enumerate(chunks):
                print(f"processing {fname}, chunk {i} of {len(chunks)}" , end ='\r')
                sl = slice(start,start+cfg.chunkSize)

                shots0 = opisSlicer0( dataf[cfg.hdf.opisTr0][sl], pulses[sl])
                shots1 = opisSlicer1( dataf[cfg.hdf.opisTr1][sl], pulses[sl])
                shots2 = opisSlicer2( dataf[cfg.hdf.opisTr2][sl], pulses[sl])
                shots3 = opisSlicer3( dataf[cfg.hdf.opisTr3][sl], pulses[sl])

                try:
                    fout.append( 'tof0', shots0, format='t' , append = True )
                    fout.append( 'tof1', shots1, format='t' , append = True )
                    fout.append( 'tof2', shots2, format='t' , append = True )
                    fout.append( 'tof3', shots3, format='t' , append = True )
                except Exception as e:
                    print()
                    print(e)

                print()
                print(fout.keys())

            pulses = pulses.query("index != 0")

            fout.append('pulses' , pulses , format='t' , data_columns=True, append = True )
            print()
    fout.close()

    if cfg.output.fname == 'AUTO':
        newname = "OPIS-%d-%d.h5" % (pulses.index[0], pulses.index[-1])
        os.rename(cfg.output.folder + outfname, cfg.output.folder + newname  )
        print("Output renamed to: %s" % newname)

if __name__ == "__main__":
	main()
