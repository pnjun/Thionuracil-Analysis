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
                          'files'    : ['FLASH2_USER1-2019-03-25T1548.h5' ] ,
                        },
           'hdf'      : { 'tofTrace'   : '/FL2/Experiment/MTCA-EXP1/ADQ412 GHz ADC/CH00/TD',
                          'retarder'   : '/FL2/Experiment/URSA-PQ/TOF/HV retarder',
                          'delay'      : '/FL2/Experiment/Pump probe laser/laser delay readback',
                          'shotsCount' : '/FL2/Timing/actual number of bunches',
                          'times'      : '/Timing/time stamp/fl2user1',
                        },                
           'slicing'  : { 'offset'   : 22000,       #Offset of first slice in samples (time zero)
                          'period'   : 9969.67,     #Rep period of FEL in samples
                          'window'   : 2000 ,       #Shot lenght in samples
                          'maxshots' : 79,          #Maxium nuber of shots per macrobunch
                        },
           'output'   : { 'fname' : 'compressed.h5',       
                          'format': 'table'         # must be 'table' or 'fixed'
                        },           
                                              
           'chunkSize': 200 #How many macrobunches to read/write at a time. Increasing increases RAM usage
         }
config = AttrDict(config)

class Slicer:
    def __init__(self, sliceParams):
        ''' 
        Prepares a list of indexes for tof trace slicing
        self.slices is a list of np.ranges, each one corresponding to a slice
        '''
        shotsCuts = (sliceParams.offset + ( sliceParams.period * np.arange(sliceParams.maxshots) )).astype(int)
        self.slices = shotsCuts[:,None] + np.arange(sliceParams.window)
        
    def __call__(self, tofData, pulses, mask = slice(None) ):
        '''
        Slices tofData and returns a dataframe of shots. Each shot is indexed by macrobunch and shotnumber
        mask is a slice object indicating which subset of the data to operate on.
        '''
        pulseList = []
        indexList = []
        for trace, shotCount, pulseId in zip( tofData[mask], pulses['shotsCount'][mask], pulses['pulseid'][mask] ):
            with suppress(IndexError, AssertionError):
                #Some pulses have no shots (NaN in 'actual number of bunches') or no macrobunch number => we drop them
                assert(pulseId != 0)
                pulseList.append( pd.Series(  trace[self.slices[:shotCount]].tolist()  ))
                indexList.append( pulseId )
                
        #Create multindexed series. Top level index is macrobunch number, sub index is shot num.
        return pd.DataFrame( pulseList, index=indexList).stack()
        
def main():
    fout = pd.HDFStore(config.output.fname)
    
    for fname in config.data.files:
        with h5py.File( config.data.path + fname ) as dataf:
            #Add macrobunch information to bunch DF
            pulses = pd.DataFrame( { 'pulseid'    : dataf[config.hdf.times][:, 2], 
                                              'time'       : dataf[config.hdf.times][:, 0],
                                              'shotsCount' : dataf[config.hdf.shotsCount][:, 0],
                                              'delay'      : dataf[config.hdf.delay][:, 0],
                                              'retarder'   : dataf[config.hdf.retarder][:, 0]
                                             } ) 
                                          
            #Slice shot data and add it to shots DF
            getSlices = Slicer(config.slicing)
            chunks = np.arange(0, dataf[config.hdf.tofTrace].shape[0], config.chunkSize)
            
            for start in chunks:
                print("processing %d" % (start))
                #TODO: dataframe is not serializable
                fout.append( 'shots', getSlices( dataf[config.hdf.tofTrace], pulses, slice(start,start+config.chunkSize) ) ,
                                      format=config.output.format, data_columns=True)
           
            fout.append( 'shots', getSlices( dataf[config.hdf.tofTrace], pulses,  slice(dataf[config.hdf.tofTrace].shape[0],None) ) ,
                                  format=config.output.format, data_columns=True)
                                    
        pulses = pulses.set_index('pulseid')  
        pulses = pulses[pulses.index != 0]
        fout.append('pulses', pulses, format=config.output.format, data_columns=True)
        del pulses
    
    fout.close()

        
if __name__ == "__main__":
	main()
	
