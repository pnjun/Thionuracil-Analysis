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
'''

import h5py
import pandas as pd
import numpy as np
from contextlib import suppress
from attrdict import AttrDict

#Configuration parameters:
config = { 'data'     : { 'path'     : '/asap3/flash/gpfs/fl24/2019/data/11005582/raw/hdf/by-start/',     
                          'files'    : [ 'FLASH2_USER1-2019-03-25T1548.h5' ] ,   # List of files to process. All files must have the same number of shots per macrobunch
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
           'output'   : { 
                          'folder' : '/asap3/flash/gpfs/fl24/2019/data/11005582/shared/Analysis/',
                          'fname'  : 'compressed.h5',       
                        },           
                                              
           'chunkSize': 51 #How many macrobunches to read/write at a time. Increasing increases RAM usage (1 macrobunch is about 6.5 MB)
         }
config = AttrDict(config)

class Slicer:
    def __init__(self, sliceParams):
        ''' 
        Prepares a list of indexes for tof trace slicing
        self.slices is a list of np.ranges, each one corresponding to a slice
        '''
        shotsCuts = (sliceParams.offset + sliceParams.skipNum + ( sliceParams.period * np.arange(sliceParams.shotsNum) )).astype(int)
        self.slices = shotsCuts[:,None] + np.arange(sliceParams.window)
        
        self.skipNum  = sliceParams.skipNum
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
                #Some pulses have no macrobunch number => we drop them
                assert(pulseId != 0)
                pulseList.append( pd.DataFrame(  trace[self.slices]  ))
                indexList.append( pulseId )
                
        #Create multindexed series. Top level index is macrobunch number, sub index is shot num.
        try:
            shots = pd.concat(pulseList, keys=indexList)
            shots.index.rename( ['pulseId', 'shotNum'], inplace=True )
            shots.columns = self.Tof2eV( ( shots.columns + self.skipNum ) * self.sampleDT ) 
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
    fout = pd.HDFStore(config.output.folder + config.output.fname, complevel=6)  # complevel btw 0 and 10; default lib for pandas is zlib, change with complib=''
    
    for fname in config.data.files:
        with h5py.File( config.data.path + fname ) as dataf:
            #Dataframe for macrobunch info
            pulses = pd.DataFrame( { 'pulseId'     : dataf[config.hdf.times][:, 2].astype('int64'), #using int64 instead of uint64 since the latter is not always supported by pytables
                                     'time'        : dataf[config.hdf.times][:, 0],
                                     'undulatorEV' : 1239.84193 / dataf[config.hdf.undulSetWL][:, 0], #nanometers to eV
                                     'opisEV'      : dataf[config.hdf.opisEV][:, 0],
                                     'delay'       : dataf[config.hdf.delay][:, 0],
                                     'waveplate'   : dataf[config.hdf.waveplate][:, 0],
                                     'retarder'    : dataf[config.hdf.retarder][:, 0]
                                   } , columns = [ 'pulseId', 'time', 'undulatorEV', 'opisEV', 'delay', 'waveplate', 'retarder' ]) 
            pulses = pulses.set_index('pulseId')   
            
            #Dataframe for shot by shot metadata. TOF traces are in shotsTof                       
            shotsData = pd.DataFrame( dataf[config.hdf.shotGmd], index=pulses.index ).stack()
            shotsData.index.rename( ['pulseId', 'shotNum'], inplace=True )
            shotsData.name = 'GMD'
            
            
            #Slice shot data and add it to shotsTof
            getSlices = Slicer(config.slicing)
            
            chunks = np.arange(0, dataf[config.hdf.tofTrace].shape[0], config.chunkSize) #We do stuff in cuncks to avoid loading too much data in memory at once
            for i, start in enumerate(chunks):
                print("processing %s, chunk %d of %d        " % (fname, i, len(chunks)) , end ='\r')
         
                shotsTof = getSlices( dataf[config.hdf.tofTrace], pulses, slice(start,start+config.chunkSize) )
                
                
                
                if shotsTof is not None:
                    fout.put( 'shotsTof', shotsTof, format='t', append = True )
            
            #Write out the other DF
            shotsData = shotsData[ shotsData.index.get_level_values('pulseId') != 0 ] #purge invalid marcobunch id
            pulses    = pulses   [ pulses.index != 0 ] 
            fout.put('pulses'   , pulses   , format='t' , data_columns=True, append = True )
            fout.put('shotsData', shotsData, format='t' , data_columns=True, append = True )
        
        
    fout.close()

        
if __name__ == "__main__":
	main()
	
