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
cfg = {    'data'     : { 'path'     : '/media/Data/Beamtime/raw/',     
                          'files'    : [ 'FLASH2_USER1-2019-04-06T0400.h5' ] ,   # List of files to process. All files must have the same number of shots per macrobunch
                        },
           'output'   : { 
                          'folder' : '',
                          # 'AUTO' for 'OPIS-<firstPulseId>-<lastPulseId.h5>'. Use only when data.files is a list of subsequent shots.     
                          'fname'  : 'opistest.h5',  
                        },  
           'hdf'      : { 'opisTr1'    : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Raw data/CH00',
                          'opisTr2'    : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Raw data/CH01',
                          'opisTr3'    : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Raw data/CH02',
                          'opisTr4'    : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Raw data/CH03',
                          'ret1'       : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Expert stuff/eTOF1 voltages/Ret nominal set',
                          'ret2'       : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Expert stuff/eTOF2 voltages/Ret nominal set',
                          'ret3'       : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Expert stuff/eTOF3 voltages/Ret nominal set',
                          'ret4'       : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Expert stuff/eTOF4 voltages/Ret nominal set',
                          'times'      : '/Timing/time stamp/fl2user1',
                        },                
           'slicing1' : { 'offset'   : 3478,      #Offset of first slice in samples (time zero)
                          'period'   : 3500,     #Rep period of FEL in samples
                          'window'   : 3500,     #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 300,      #Number of samples to skip at the beginning of each slice (cuts off high energy electrons)
                          'shotsNum' : 20,       #Number of shots per macrobunch
                        },
           'slicing2' : { 'offset'   : 3441,      #Offset of first slice in samples (time zero)
                          'period'   : 3500,     #Rep period of FEL in samples
                          'window'   : 3500,     #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 300,      #Number of samples to skip at the beginning of each slice (cuts off high energy electrons)
                          'shotsNum' : 20,       #Number of shots per macrobunch
                        },
           'slicing3' : { 'offset'   : 3435,      #Offset of first slice in samples (time zero)
                          'period'   : 3500,     #Rep period of FEL in samples
                          'window'   : 3500,     #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 300,      #Number of samples to skip at the beginning of each slice (cuts off high energy electrons)
                          'shotsNum' : 20,       #Number of shots per macrobunch
                        },
           'slicing4' : { 'offset'   : 3433,      #Offset of first slice in samples (time zero)
                          'period'   : 3500,     #Rep period of FEL in samples
                          'window'   : 3500,     #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 300,      #Number of samples to skip at the beginning of each slice (cuts off high energy electrons)
                          'shotsNum' : 20,       #Number of shots per macrobunch
                        },
           'tof2ev'   : { 'len'      : 0.309,      #lenght of flight tube in meters
                          'dt'       : 0.001429    #interval between tof samples in us
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
            opisSlicer1 = Slicer(cfg.slicing1, tof2ev = cfg.tof2ev)
            opisSlicer2 = Slicer(cfg.slicing2, tof2ev = cfg.tof2ev)
            opisSlicer3 = Slicer(cfg.slicing3, tof2ev = cfg.tof2ev)
            opisSlicer4 = Slicer(cfg.slicing4, tof2ev = cfg.tof2ev)
                    
            #Split up computation in cunks
            chunks = np.arange(0, dataf[cfg.hdf.opisTr1].shape[0], cfg.chunkSize)
            for i, start in enumerate(chunks):
                print("processing %s, chunk %d of %d        " % (fname, i, len(chunks)) , end ='\r')
                sl = slice(start,start+cfg.chunkSize)
                
                shots1 = opisSlicer1( dataf[cfg.hdf.opisTr1][sl], pulses[sl])
                shots2 = opisSlicer2( dataf[cfg.hdf.opisTr2][sl], pulses[sl])
                shots3 = opisSlicer3( dataf[cfg.hdf.opisTr3][sl], pulses[sl])
                shots4 = opisSlicer4( dataf[cfg.hdf.opisTr4][sl], pulses[sl])

                #plot one of the traces\
                energy = False
                import matplotlib.pyplot as plt
                shots1.query("pulseId > 81034488 and pulseId < 81034688").mean().plot(use_index=energy)
                shots2.query("pulseId > 81034488 and pulseId < 81034688").mean().plot(use_index=energy)
                shots3.query("pulseId > 81034488 and pulseId < 81034688").mean().plot(use_index=energy)
                shots4.query("pulseId > 81034488 and pulseId < 81034688").mean().plot(use_index=energy)
                plt.show()
                
                exit()
                if shots1 is not None: 
                    fout.put( 'tof1', shots1, format='t', append = True )
                    fout.put( 'tof2', shots2, format='t', append = True )
                    fout.put( 'tof3', shots3, format='t', append = True )
                    fout.put( 'tof4', shots4, format='t', append = True )
                                        
            print()       
    fout.close()
    
    if cfg.output.fname == 'AUTO':
        newname = "OPIS-%d-%d.h5" % (pulses.index[0], pulses.index[-1]) 
        os.rename(cfg.output.folder + outfname, cfg.output.folder + newname  )
        print("Output renamed to: %s" % newname)
        
if __name__ == "__main__":
	main()
	
