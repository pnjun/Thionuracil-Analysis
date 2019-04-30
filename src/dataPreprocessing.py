'''
This script reads data from the hdf5 files provided by flash and compresses 
and reorganizes the data structure. Each macropulse is split up shot by shot. 
TOF and OPIS traces are saved by shot, with additional datatables for macrobunch
(timestam, pressure, delay, etc..) and shot data (GMD, UVON)

Each shot is given a shot id as follows:
    shot_id = macrobunch_id * 100 + shotnumber

Requires: - pandas
          - h5py
          - attrdict

For info contact Fabiano
'''

import h5py
import pandas as pd
from attrdict import AttrDict

#Configuration parameters:
config = { 'data'     : { 'path'     : '/asap3/flash/gpfs/fl24/2019/data/11005582/raw/hdf/by-start/',     
                          'files'    : [ 'FLASH2_USER1-2019-03-25T1219.h5' ] 
                        },
           'hdf'      : { 'tofTrace' : '/FL2/Experiment/MTCA-EXP1/ADQ412 GHz ADC/CH00/TD',
                          'retarder' : '/FL2/Experiment/URSA-PQ/TOF/HV retarder',
                          'delay'    : '/FL2/Experiment/Pump probe laser/laser delay readback',
                          'times'    : '/Timing/time stamp/fl2user1'
                        },                
           'slicing'  : { 'offset'   : 22000, 
                          'period'   : 9969.67,
                          'window'   : 2000 
                        }
         }
config = AttrDict(config)

bunchList = []
shotsList = []

for fname in config.data.files:
    with h5py.File( config.data.path + fname ) as dataf:
    
        bunchList.append( pd.DataFrame( { 'bunchid'  : dataf[config.hdf.times][:, 2], 
                                          'time'     : dataf[config.hdf.times][:, 0],
                                          'delay'    : dataf[config.hdf.delay][:, 0],
                                          'retarder' : dataf[config.hdf.retarder][:, 0]
                                         } ) )
       
bunches = pd.concat(bunchList)       
bunches = bunches.set_index('bunchid')  
bunches = bunches[bunches.index != 0]

#shots = pd.concat(shotsList)    
#shots.set_index('shotid')

   
print(bunches)
        
    
