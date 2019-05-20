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
from dataPreprocessing import Slicer
from attrdict import AttrDict

#cfguration parameters:
cfg = {    'data'     : { 'path'     : '/asap3/flash/gpfs/fl24/2019/data/11005582/raw/hdf/by-start/',     
                          'files'    : [ 'FLASH2_USER1-2019-03-30T2100.h5' ] ,   # List of files to process. All files must have the same number of shots per macrobunch
                        },
           'output'   : { 
                          'folder' : '',#'/asap3/flash/gpfs/fl24/2019/data/11005582/shared/Analysis/',
                          'fname'  : 'AUTO',  # use 'AUTO' for 'OPIS-<firstPulseId>-<lastPulseId.h5>'. Use this only when data.files is a list of subsequent shots.        
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
           'slicing'  : { 'offset'   : 0,        #Offset of first slice in samples (time zero)
                          'period'   : 3500,     #Rep period of FEL in samples
                          'window'   : 2000,     #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 550,      #Number of samples to skip at the beginning of each slice (cuts off high energy electrons)
                          'shotsNum' : 50,       #Number of shots per macrobunch
                        },
           'tof2ev'   : { 'len'      : 2,        #lenght of flight tube in meters
                          'dt'       : 0.0014    #interval between tof samples in s
                        },                                                                          
           'chunkSize': 500 #How many macrobunches to read/write at a time. Increasing increases RAM usage (1 macrobunch is about 6.5 MB)
         }
cfg = AttrDict(cfg)


        
def main():
    outfname = uuid.uuid4().hex + ".temp" if cfg.output.fname == 'AUTO' else cfg.output.fname
    
    fout = pd.HDFStore(cfg.output.folder + outfname, "w", complevel=6)  # complevel btw 0 and 10; default lib for pandas is zlib, change with complib=''
    for fname in cfg.data.files:
        with h5py.File( cfg.data.path + fname ) as dataf:
            #Dataframe for macrobunch info
            pulses = pd.DataFrame( { 'pulseId'     : dataf[cfg.hdf.times][:, 2].astype('int64'), #using int64 instead of uint64 since the latter is not always supported by pytables
                                     'time'        : dataf[cfg.hdf.times][:, 0],
                                   } , columns = [ 'pulseId', 'time' ]) 
            pulses = pulses.set_index('pulseId')   
            
            
            #Slice shot data and add it to shotsTof
            opisSlicer = Slicer(cfg.slicing, tof2ev = cfg.tof2ev)
                    
            #NOTE : Slicer will drop all traces with macrobunch id = 0. We will need to remove them from the other dataframes as well.
            #       The removal must be done after the slicing, otherwise slicer will get arrays with non-matching number of macrobunches
            chunks = np.arange(0, dataf[cfg.hdf.opisTr1].shape[0], cfg.chunkSize) #We do stuff in cuncks to avoid loading too much data in memory at once
            for i, start in enumerate(chunks):
                print("processing %s, chunk %d of %d        " % (fname, i, len(chunks)) , end ='\r')
                sl = slice(start,start+cfg.chunkSize)
                
                shots1 = opisSlicer( dataf[cfg.hdf.opisTr1], pulses, sl )
                shots2 = opisSlicer( dataf[cfg.hdf.opisTr2], pulses, sl )
                shots3 = opisSlicer( dataf[cfg.hdf.opisTr3], pulses, sl )
                shots4 = opisSlicer( dataf[cfg.hdf.opisTr4], pulses, sl )

                #plot one of the traces
                import matplotlib.pyplot as plt
                shots1.loc[[75634529]].mean(axis=0).plot()
                #shots2.loc[[75634529]].mean(axis=0).plot()
                #shots3.loc[[75634529]].mean(axis=0).plot()
                #shots4.loc[[75634529]].mean(axis=0).plot()
                plt.show()
                                 
                if shotsTof is not None: 
                    fout.put( 'opisShots', shotsTof, format='t', append = True )

            print()       
    fout.close()
    
    if cfg.output.fname == 'AUTO':
        newname = "OPIS-%d-%d.h5" % (pulses.index[0], pulses.index[-1]) 
        os.rename(cfg.output.folder + outfname, cfg.output.folder + newname  )
        print("Output renamed to: %s" % newname)
        
if __name__ == "__main__":
	main()
	
