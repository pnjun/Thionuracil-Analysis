#!/bin/python3.4
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
import os
import uuid
from attrdict import AttrDict

from utils import Slicer

#cfguration parameters:
cfg = {    'data'     : { 'path'     : '/media/Data/ThioUr/raw/',
                          'files'    : ['FLASH2_USER1-2019-04-01T0238.h5'] ,   # List of files to process. All files must have the same number of shots per macrobunch
                        },
           'output'   : {
                          'folder' : '',
                          # 'AUTO' for 'OPIS-<firstPulseId>-<lastPulseId.h5>'. Use only when data.files is a list of subsequent shots.
                          'fname'  : 'opistest.h5',
                        },
           'hdf'      : { 'opisTr0'    : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Raw data/CH00',
                          'opisTr1'    : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Raw data/CH01',
                          'opisTr2'    : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Raw data/CH02',
                          'opisTr3'    : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Raw data/CH03',
                          'ret0'       : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Expert stuff/eTOF1 voltages/Ret nominal set',
                          'ret1'       : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Expert stuff/eTOF2 voltages/Ret nominal set',
                          'ret2'       : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Expert stuff/eTOF3 voltages/Ret nominal set',
                          'ret'       : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Expert stuff/eTOF4 voltages/Ret nominal set',
                          'times'      : '/Timing/time stamp/fl2user1',
                        },
           'slicing0' : { 'offset'   : 263,   #Offset of first slice in samples (time zero)
                          'period'   : 3500.290,  #Rep period of FEL in samples
                          'window'   : 1400,  #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 350,     #Skip fist samples of each slice (cuts off high energy electrons)
                          'dt'       : 0.00014,     #Time between samples [us]
                          'shotsNum' : 49,    #Number of shots per macrobunch
                        },
           'slicing1' : { 'offset'   : 225,     #Offset of first slice in samples (time zero)
                          'period'   : 3500.302,  #Rep period of FEL in samples
                          'window'   : 1400,  #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 350,     #Skip fist samples of each slice (cuts off high energy electrons)
                          'dt'       : 0.00014,     #Time between samples [us]
                          'shotsNum' : 49,    #Number of shots per macrobunch
                        },
           'slicing2' : { 'offset'   : 220,     #Offset of first slice in samples (time zero)
                          'period'   : 3500.300,  #Rep period of FEL in samples
                          'window'   : 1400,  #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 350,     #Skip fist samples of each slice (cuts off high energy electrons)
                          'dt'       : 0.00014,     #Time between samples [us]
                          'shotsNum' : 49,    #Number of shots per macrobunch
                        },
           'slicing3' : { 'offset'   : 219,     #Offset of first slice in samples (time zero)
                          'period'   : 3500.297,  #Rep period of FEL in samples
                          'window'   : 1400,  #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 350,     #Skip fist samples of each slice (cuts off high energy electrons)
                          'dt'       : 0.00014,     #Time between samples [us]
                          'shotsNum' : 49,    #Number of shots per macrobunch
                        },
           'chunkSize': 500 #How many macrobunches to read/write at a time. Increasing increases RAM usage (1 macrobunch is about 6.5 MB)
         }
cfg = AttrDict(cfg)



def main():
    outfname = uuid.uuid4().hex + ".temp" if cfg.output.fname == 'AUTO' else cfg.output.fname

    fout = pd.HDFStore(cfg.output.folder + outfname, "w")  # complevel btw 0 and 10
    for fname in cfg.data.files:
        with h5py.File( cfg.data.path + fname ) as dataf:
            #Dataframe for macrobunch info
            pulses = pd.DataFrame( { 'pulseId'     : dataf[cfg.hdf.times][:, 2].astype('int64'),
                                     'time'        : dataf[cfg.hdf.times][:, 0],
                                   } , columns = [ 'pulseId', 'time' ])
            pulses = pulses.set_index('pulseId')

            #Slice shot data and add it to shotsTof
            opisSlicer0 = Slicer(cfg.slicing0, removeBg = True)
            opisSlicer1 = Slicer(cfg.slicing1, removeBg = True)
            opisSlicer2 = Slicer(cfg.slicing2, removeBg = True)
            opisSlicer3 = Slicer(cfg.slicing3, removeBg = True)


            #Split up computation in cunks
            chunks = np.arange(0, dataf[cfg.hdf.opisTr1].shape[0], cfg.chunkSize)
            for i, start in enumerate(chunks):
                print("processing %s, chunk %d of %d        " % (fname, i, len(chunks)) , end ='\r')
                sl = slice(start,start+cfg.chunkSize)

                shots0 = opisSlicer0( dataf[cfg.hdf.opisTr0][sl], pulses[sl])
                shots1 = opisSlicer1( dataf[cfg.hdf.opisTr1][sl], pulses[sl])
                shots2 = opisSlicer2( dataf[cfg.hdf.opisTr2][sl], pulses[sl])
                shots3 = opisSlicer3( dataf[cfg.hdf.opisTr3][sl], pulses[sl])
                '''
                #plot one of the traces\
                import matplotlib.pyplot as plt
                from utils import opisEvConv
                conv= opisEvConv()
                plt.plot(conv[0](shots0.columns),shots0.mean())
                plt.plot(conv[1](shots1.columns),shots1.mean())
                plt.plot(conv[2](shots2.columns),shots2.mean())
                plt.plot(conv[3](shots3.columns),shots3.mean())
                #shots0.mean().plot()
                #shots1.mean().plot()
                #shots2.mean().plot()
                #shots3.mean().plot()
                plt.show()

                exit()'''
                try:
                    fout.append( 'tof0', shots0, format='t' , append = True )
                    fout.append( 'tof1', shots1, format='t' , append = True )
                    fout.append( 'tof2', shots2, format='t' , append = True )
                    fout.append( 'tof3', shots3, format='t' , append = True )
                except Exception as e:
                    print(e)
            print()
    fout.close()

    if cfg.output.fname == 'AUTO':
        newname = "OPIS-%d-%d.h5" % (pulses.index[0], pulses.index[-1])
        os.rename(cfg.output.folder + outfname, cfg.output.folder + newname  )
        print("Output renamed to: %s" % newname)

if __name__ == "__main__":
	main()
