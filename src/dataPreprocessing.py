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
from contextlib import suppress
from attrdict import AttrDict

#cfguration parameters:
cfg = { 'data'     : { 'path'     : '/asap3/flash/gpfs/fl24/2019/data/11005582/raw/hdf/by-start/',     
                          'files'    : [ 'FLASH2_USER1-2019-03-26T0500.h5' ] ,   # List of files to process. All files must have the same number of shots per macrobunch
                        },
           'output'   : { 
                          'folder' : '',#'/asap3/flash/gpfs/fl24/2019/data/11005582/shared/Analysis/',
                          'fname'  : 'compressed.h5',       
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
           'slicing'  : { 'offset'   : 22000,       #Offset of first slice in samples (time zero)
                          'period'   : 9969.67,     #Rep period of FEL in samples
                          'window'   : 2700 ,       #Shot lenght in samples (cuts off low energy electrons)
                          'skipNum'  : 300,         #Number of samples to skip at the beginning of each slice (cuts off high energy electrons)
                          'shotsNum' : 50,          #Number of shots per macrobunch
                        },
           'laser'    : { 'slicing'  :  { 'offset'   : 850,        #Same as above but for 100MHz laser trace slicing
                                          'period'   : 540,    
                                          'window'   : 38,       
                                          'skipNum'  : 0,         
                                          'shotsNum' : 50,          
                                        },
                          'bgLvl'    : 32720       # Value for bg correction of Laser trace 
                        },
                                                   
           'chunkSize': 70 #How many macrobunches to read/write at a time. Increasing increases RAM usage (1 macrobunch is about 6.5 MB)
         }
cfg = AttrDict(cfg)

class Slicer:
    def __init__(self, sliceParams, eVnames = False):
        ''' 
        Prepares a list of indexes for tof trace slicing
        self.slices is a list of np.ranges, each one corresponding to a slice
        '''
        shotsCuts = (sliceParams.offset + sliceParams.skipNum + ( sliceParams.period * np.arange(sliceParams.shotsNum) )).astype(int)
        self.slices = shotsCuts[:,None] + np.arange(sliceParams.window)
        
        self.skipNum  = sliceParams.skipNum
        self.eVnames  = eVnames
        self.sampleDT = 0.0005 # Time in us between each sample point (used for ev conversion)
        
    def __call__(self, tofData, pulses, mask = slice(None)):
        '''
        Slices tofData and returns a dataframe of shots. Each shot is indexed by macrobunch and shotnumber
        mask is a slice object indicating which subset of the data to operate on.
        '''
        pulseList = []
        indexList = []
        for trace, pulseId in zip( tofData[mask], pulses.index[mask] ):
            with suppress(AssertionError):
                #Some pulses have no macrobunch number => we drop them (don't waste time on useless data)
                assert(pulseId != 0)
                pulseList.append( pd.DataFrame(  trace[self.slices]  ))
                indexList.append( pulseId )
                
        #Create multindexed series. Top level index is macrobunch number, sub index is shot num.
        try:
            shots = pd.concat(pulseList, keys=indexList)
            shots.index.rename( ['pulseId', 'shotNum'], inplace=True )
            
            #Name columns as tof to eV conversion
            if self.eVnames: shots.columns = self.Tof2eV( ( shots.columns + self.skipNum ) * self.sampleDT ) 
            
            return shots
        except ValueError:
            return None
                        
    def Tof2eV(self, tof):
        ''' converts time of flight into ectronvolts '''
        # Constants for conversion:
        s = 1.7
        m_over_e = 5.69
        # UNITS AND ORDERS OF MAGNITUDE DO CHECK OUT    
        return 0.5 * m_over_e * ( s / ( tof ) )**2
        
def main():
    fout = pd.HDFStore(cfg.output.folder + cfg.output.fname, complevel=6)  # complevel btw 0 and 10; default lib for pandas is zlib, change with complib=''
    
    for fname in cfg.data.files:
        with h5py.File( cfg.data.path + fname ) as dataf:
            #Dataframe for macrobunch info
            pulses = pd.DataFrame( { 'pulseId'     : dataf[cfg.hdf.times][:, 2].astype('int64'), #using int64 instead of uint64 since the latter is not always supported by pytables
                                     'time'        : dataf[cfg.hdf.times][:, 0],
                                     'undulatorEV' : 1239.84193 / dataf[cfg.hdf.undulSetWL][:, 0], #nanometers to eV
                                     'opisEV'      : dataf[cfg.hdf.opisEV][:, 0],
                                     'delay'       : dataf[cfg.hdf.delay][:, 0],
                                     'waveplate'   : dataf[cfg.hdf.waveplate][:, 0],
                                     'retarder'    : dataf[cfg.hdf.retarder][:, 0]
                                   } , columns = [ 'pulseId', 'time', 'undulatorEV', 'opisEV', 'delay', 'waveplate', 'retarder' ]) 
            pulses = pulses.set_index('pulseId')   
            
            
            #Slice shot data and add it to shotsTof
            tofSlicer = Slicer(cfg.slicing, eVnames = True)
            laserSlicer = Slicer(cfg.laser.slicing)
            
            #NOTE : Slicer will drop all traces with macrobunch id = 0. We will need to remove them from the other dataframes as well.
            #       The removal must be done after the slicing, otherwise slicer will get arrays with non-matching number of macrobunches
            
            laserStatusList = []
            chunks = np.arange(0, dataf[cfg.hdf.tofTrace].shape[0], cfg.chunkSize) #We do stuff in cuncks to avoid loading too much data in memory at once
            for i, start in enumerate(chunks):
                print("processing %s, chunk %d of %d        " % (fname, i, len(chunks)) , end ='\r')
                sl = slice(start,start+cfg.chunkSize)
                
                shotsTof = tofSlicer( dataf[cfg.hdf.tofTrace], pulses, sl )
                laserTr  = laserSlicer( dataf[cfg.hdf.laserTrace], pulses, sl )
                
                #plot one of the traces
                #import matplotlib.pyplot as plt
                #plt.plot(dataf[cfg.hdf.laserTrace][10000])
                #plt.show()
                                 
                if shotsTof is not None: 
                    fout.put( 'shotsTof', shotsTof, format='t', append = True )
                    
                    laserTr -= cfg.laser.bgLvl
                    laserTr = laserTr.sum(axis=1)  #integrate laser power
                    laserStatusList.append(laserTr)
            print()
            
            
            #Dataframe for shot by shot metadata. TOF traces are in shotsTof  
            #Contains GMD and UV laser power shot by shot   
            shotsData = pd.DataFrame( dataf[cfg.hdf.shotGmd], index=pulses.index )
            shotsData = shotsData[shotsData.index != 0].stack(dropna = False).to_frame(name='GMD') #purge invalid IDs, stack shots (creates multindex) and rename column
            shotsData.index.rename( ['pulseId', 'shotNum'], inplace=True )          
            shotsData = shotsData.join(pd.concat(laserStatusList).to_frame(name='uvPow'))
            
            #Write out the other DF   
            pulses    = pulses   [ pulses.index != 0 ] #purge invalid marcobunch id            
            fout.put('pulses'   , pulses   , format='t' , data_columns=True, append = True )
            fout.put('shotsData', shotsData, format='t' , data_columns=True, append = True )
        
        
    fout.close()

        
if __name__ == "__main__":
	main()
	
