#!/usr/bin/python3
'''
This script reads data from the hdf5 files provided by flash and compresses 
and reorganizes the data structure. Each macrobunch is split up shot by shot. 
TOF and OPIS traces are saved by shot togeter with shot data (GMD, UVON), 
with additional datatables for macrobunch params (timestam, pressure, delay, etc..)
In the code macrobunch are 'pulses' and shots are 'shots'
Requires: - pandas
          - h5py
          - attrdict
For info contact Fabiano
**** NOTES ****
tof slicing offset might be slightly off -> eV conversion is wrong -> we need to check this
laser slicing offset might be off by one shot -> swaps even and odd shots -> we need to check this
'''

import h5py
import pandas as pd
import numpy as np
import uuid
import os, glob
from time import time
from contextlib import suppress
from attrdict import AttrDict

from utils import Slicer

#cfguration parameters:
cfg = {    'data'     : { 'path'     : '/media/Data/Beamtime/raw/',     
                          'files'    : 'FLASH2_USER1-2019-0?-[^2][^456]*.h5', #'FLASH2_USER1-2019-03-2*.h5', #['FLASH2_USER1-2019-03-27T0630.h5'] # List of files to process or globbable string. All files must have the same number of shots
                        },
           'output'   : { 
                          'folder'      : '/media/Data/Beamtime/processed/',
                          'pulsefname'  : 'index_test.h5',
                          'shotsfname'  : 'first_block_test.h5',  # use 'AUTO' for '<firstPulseId>-<lastPulseId.h5>'. Use this only when data.files is a list of subsequent shots.        
                        },  
           'hdf'      : { 'tofTrace'   : '/FL2/Experiment/MTCA-EXP1/ADQ412 GHz ADC/CH00/TD',
                          'retarder'   : '/FL2/Experiment/URSA-PQ/TOF/HV retarder',
                          'delay'      : '/FL2/Experiment/Pump probe laser/laser delay readback',
                          'waveplate'  : '/FL2/Experiment/Pump probe laser/FL24/attenuator position',
                          'times'      : '/Timing/time stamp/fl2user1',
                          'opisEV'     : '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean phtoton energy',
                          'undulSetWL' : '/FL2/Electron Diagnostic/Undulator setting/set wavelength',
                          'shotGmd'    : '/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall',
                          'laserTrace' : '/FL2/Experiment/MTCA-EXP1/SIS8300 100MHz ADC/CH4/TD'
                        },                
           'slicing'  : { 'offset'   : 21732,       #Offset of first slice in samples (time zero)
                          'period'   : 9969.67,     #Rep period of FEL in samples
                          'window'   : 3000 ,       #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 350,         #Number of samples to skip at the beginning of each slice (cuts off high energy electrons
                          'shotsNum' : 50,          #Number of shots per macrobunch
                        },
           'tof2ev'   : { 'len'      : 1.7088,      #lenght of flight tube in meters
                          'dt'       : 0.0005       #interval between tof samples in us
                        },
           'laser'    : { 'slicing'  :  { 'offset'   : 855,        #Same as above but for 100MHz laser trace slicing
                                          'period'   : 540,    
                                          'window'   : 30,       
                                          'skipNum'  : 0,         
                                          'shotsNum' : 50,          
                                        },
                        },
                                                   
           'chunkSize': 500 #How many macrobunches to read/write at a time. Increasing increases RAM usage (1 macrobunch is about 6.5 MB)
         }
cfg = AttrDict(cfg)


def main():
    outfname = uuid.uuid4().hex + '.temp' if cfg.output.shotsfname == 'AUTO' else cfg.output.shotsfname
    
    shotsfout = pd.HDFStore(cfg.output.folder + outfname)  # complevel btw 0 and 10
    pulsefout = pd.HDFStore(cfg.output.folder + cfg.output.pulsefname)
    
    flist = [ cfg.data.path + fname for fname in cfg.data.files ] if isinstance(cfg.data.files, tuple) else glob.glob(cfg.data.path + cfg.data.files)
    print("processing %d files" % len(flist))
    for fname in flist:
        print("Opening %s" % fname)
        with h5py.File( fname ) as dataf:
            #Dataframe for macrobunch info
            pulses = pd.DataFrame( { 'pulseId'     : dataf[cfg.hdf.times][:, 2].astype('int64'),
                                     'time'        : dataf[cfg.hdf.times][:, 0],
                                     'undulatorEV' : 1239.84193 / dataf[cfg.hdf.undulSetWL][:, 0], #nanometers to eV
                                     'opisEV'      : dataf[cfg.hdf.opisEV][:, 0],
                                     'delay'       : dataf[cfg.hdf.delay][:, 0],
                                     'waveplate'   : dataf[cfg.hdf.waveplate][:, 0],
                                     'retarder'    : dataf[cfg.hdf.retarder][:, 0]
                                   } , columns = [ 'pulseId', 'time', 'undulatorEV', 'opisEV', 'delay', 'waveplate', 'retarder' ]) 
            pulses = pulses.set_index('pulseId')   
            
            #Slice shot data and add it to shotsTof
            tofSlicer = Slicer(cfg.slicing, tof2ev = cfg.tof2ev)
            laserSlicer = Slicer(cfg.laser.slicing, removeBg=True)
            
            #NOTE : Slicer will drop all traces with macrobunch id = 0.
            # We will need to remove them from the other dataframes as well.
            # The removal must be done after the slicing, otherwise slicer will get arrays with 
            # non-matching number of macrobunches
            
            laserStatusList = []
            #We do stuff in cuncks to avoid loading too much data in memory at once
            chunks = np.arange(0, dataf[cfg.hdf.tofTrace].shape[0], cfg.chunkSize) 
            
            #Load index of data already written to file
            #We check against this to avoid writing duplicates
            try:
                shotsIdx = shotsfout['shotsData'].index.levels[0]
            except (KeyError):
                shotsIdx = []
                
            endtime = 0
            for i, start in enumerate(chunks):
                starttime = endtime
                endtime = time()
                print("chunk %d of %d (%.1f bunches/sec )        " % (i+1, len(chunks), cfg.chunkSize / (endtime-starttime) ) , end ='\r')
                
                sl = slice(start,start+cfg.chunkSize)
                shotsTof = tofSlicer( dataf[cfg.hdf.tofTrace][sl], pulses[sl])
                laserTr  = laserSlicer( dataf[cfg.hdf.laserTrace][sl], pulses[sl])
                
                #plot one of the traces
                #import matplotlib.pyplot as plt
                #plt.plot(laserTr.iloc[500])
                #plt.plot(laserTr.iloc[501])
                #plt.show()
                                
                if shotsTof is not None:      
                    #Some raw files have overlap. If this is the case, drop bunches we already processed.
                    shotsTof = shotsTof.query('pulseId not in @shotsIdx')           
                    
                    #Write down tof data
                    shotsfout.put( 'shotsTof', shotsTof, format='t', append = True )
                    
                    #Keep laser data for later, we bundle it with GMD before writing it out
                    laserTr = laserTr.sum(axis=1)  #integrate laser power
                    laserStatusList.append(laserTr)
            print()
            
            
            #Dataframe for shot by shot metadata. TOF traces are in shotsTof  
            #Contains GMD and UV laser power shot by shot   
            shotsData = pd.DataFrame( dataf[cfg.hdf.shotGmd], index=pulses.index )
            #purge invalid IDs, stack shots (creates multindex) and rename column
            shotsData = shotsData[shotsData.index != 0].stack(dropna = False).to_frame(name='GMD') 
            shotsData.index.rename( ['pulseId', 'shotNum'], inplace=True )          
            shotsData = shotsData.join(pd.concat(laserStatusList).to_frame(name='uvPow'))

            #Write out the DFs   
 
            #purge invalid marcobunch id 
            pulses    = pulses   [ pulses.index != 0 ] 
            #Remove pulses that have already been processed to avoid inserting duplicates
            try:
                pulsesIdx = pulsefout['pulses'].index
                pulses = pulses.query("index not in @pulsesIdx")
            except (KeyError):
                pass
            shotsData = shotsData.query("pulseId not in @shotsIdx") 

            pulsefout.append('pulses'   , pulses   , format='t' , data_columns=True, append = True )           
            shotsfout.append('shotsData', shotsData, format='t' , data_columns=True, append = True )

    shotsfout.close()
    pulsefout.close()
    
    if cfg.output.shotsfname == 'AUTO' and len(pulses):
        newname = "%d-%d.h5" % (pulses.index[0], pulses.index[-1]) 
        os.rename(cfg.output.folder + outfname, cfg.output.folder + newname  )
        print("Output renamed to: %s" % newname)
        
if __name__ == "__main__":
	main()
	
