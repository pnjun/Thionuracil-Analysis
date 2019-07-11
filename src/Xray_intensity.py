import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from datetime import datetime
from attrdict import AttrDict
from utils import mainTofEvConv

#create a dictionary with basic parameters
cfg =	{ 'data' : 	{'path' : '/media/Data/Beamtime/processed/',
			'index' : 'index.h5',
			'trace' : 'first_block.h5'
			}
	}
# change dictionary to support attributes
cfg = AttrDict(cfg)

### load pulse index file
idx = pd.HDFStore(cfg.data.path + cfg.data.index, mode = 'r')
### load electron traces
tr = pd.HDFStore(cfg.data.path + cfg.data.trace, mode = 'r')

#print("These are the keys of the electron traces:", tr.keys())
### keys are shotsData, shotsTof
#print("These are the keys of shotsData:", tr.shotsData.keys())
### keys are GMD, uvPow, BAM

#print("These are the keys of the pulse index file:", idx.keys())
### keys are pulses
#print("These are the keys of idx.pulses:", idx.pulses.keys())
### keys of pulses are time, undulatorEV opisEV delay waveplate retarder


'''	electron traces can only be loaded for certain time intervals to revent memory error
	scheme: choose a start and a stop time with datetime()
	determine the respective linux time stamp with the .timestamp() method
	define a pulseId array
'''

start = [	datetime(2019,3,25,21,27,00).timestamp(),
		datetime(2019,3,25,21,40,00).timestamp(),
		datetime(2019,3,25,21,58,00).timestamp(),
		datetime(2019,3,25,22,20,00).timestamp(),
		datetime(2019,3,25,22,31,00).timestamp(),
		datetime(2019,3,25,22,37,00).timestamp(),
		datetime(2019,3,25,22,49,00).timestamp(),
		datetime(2019,3,25,23,4,00).timestamp(),
		datetime(2019,3,25,23,22,00).timestamp(),
		datetime(2019,3,25,23,35,00).timestamp()
		]

#start = min(idx.pulses.time)

stop = [	datetime(2019,3,25,21,33,00).timestamp(),
		datetime(2019,3,25,21,45,00).timestamp(),
		datetime(2019,3,25,22,3,00).timestamp(),
		datetime(2019,3,25,22,23,00).timestamp(),
		datetime(2019,3,25,22,34,00).timestamp(),
		datetime(2019,3,25,22,43,00).timestamp(),
		datetime(2019,3,25,22,52,00).timestamp(),
		datetime(2019,3,25,23,8,00).timestamp(),
		datetime(2019,3,25,23,26,00).timestamp(),
		datetime(2019,3,25,23,38,00).timestamp()
		]

GMD_set = [0.023, 0.088, 0.6, 1, 5.75, 10, 18, 45, 60, 3.52]

''''	Multi-Gaussian fit function
	sum of 9 Gaussians fiited to data'''

def fit_func(	x, C, A0,x0,d0,A1,x1,d1,A2,x2,d2,A3,x3,d3,A4,x4,d4,A5,x5,d5,A6,x6,d6,
		A7,x7,d7,A8,x8,d8,A9,x9,d9,A10,x10,d10):
	return	(C + A0 * np.exp(-0.5 * np.square((x-x0) / d0))
		+ A1 * np.exp(-0.5 * np.square((x-x1) / d1))
		+ A2 * np.exp(-0.5 * np.square((x-x2) / d2))
		+ A3 * np.exp(-0.5 * np.square((x-x3) / d3))
		+ A4 * np.exp(-0.5 * np.square((x-x4) / d4))
		+ A5 * np.exp(-0.5 * np.square((x-x5) / d5))
		+ A6 * np.exp(-0.5 * np.square((x-x6) / d6))
		+ A7 * np.exp(-0.5 * np.square((x-x7) / d7))
		+ A8 * np.exp(-0.5 * np.square((x-x8) / d8))
		+ A9 * np.exp(-0.5 * np.square((x-x9) / d9))
		)

par =	(	5.6, 0.1, 255.0, 0.2,
		0.1, 240.0, 0.2,
		0.1, 230.0, 0.2,
		0.1, 170.0, 0.2,
		0.1, 145.0, 0.3,
		0.1, 105.0, 0.1
		)


bnd_low =	[5.1, 0.0, 250., 0.10,
		0.0, 225., 0.1,
		0.0, 205., 0.10,
		0.0, 180., 0.10,
		0.0, 160., 0.10,
		0.0, 135., 0.10,
		0.0, 115., 0.10,
		0.0, 100., 0.10,
		0.0, 88., 0.10,
		0.0, 84., 0.1]

bnd_up = 	[5.9, 100.0, 265., 20.,
		100.0, 235., 10.,
		100.0, 220., 8.,
		100.0, 198., 20.,
		100.0, 175., 10.,
		100.0, 145., 10.,
		100.0, 132., 20.,
		100.0, 106., 5.,
		100.0, 96., 10.,
		100.0, 87., 5.]
bnd = (bnd_low, bnd_up)


amplitudes centers = widths = [[] for x in range(9)]

for i in range(6,7):
#for i in range(0,len(start)):
	pulse = idx.select('pulses', where='time >= start[i] and time < stop[i]')
	
	### shotsData
	meta = tr.select('shotsData', where='pulseId >= pulse.index[0] and pulseId < pulse.index[-1]')
	### electron traces
	data = tr.select('shotsTof', where='pulseId >= pulse.index[0] and pulseId < pulse.index[-1]')
	
	e_data = data.mean() * (-1)

	###averaged retarder
	tr_retarder= pulse.retarder.mean()

	ev_conv = mainTofEvConv(tr_retarder)
	
	evs = ev_conv(data.iloc[0].index.to_numpy(dtype=np.float32))
	
	params, paramsconv = optimize.curve_fit(fit_func, evs, e_data, bounds=bnd)
	
	print(params)
	#amplitude0.append(params[1])
	
	plt.figure(i+1)
	plt.plot(evs, e_data)
	plt.plot(evs, fit_func(evs, *params))
	#pulse.retarder.plot()
	#plt.plot([(pulse.index[0],0),(pulse.index[-1],0)],[GMD_set[i],GMD_set[i]], color = 'r')
	#meta.GMD.plot()
	#plt.plot((pulse.index[0],pulse.index[-1]),(GMD_set[i],GMD_set[i]))
	#plt.plot(pulse,meta.GMD)
	#plt.text(pulse.index[0],1,'i')
	#plt.ylabel('Retarder voltage in V')

plt.show()

#stop = max(idx.pulses.time)-5000

#pulse = idx.select('pulses', where='time >=start and time < stop')

#print("BAM stop:", pulse.loc[71344215].time, "BAM restart:", pulse.loc[71797868].time)

#data = tr.select('shotsData', where='pulseId >= 72547540 and pulseId < 72547550')



###plot data
#plt.figure(1)
#data.GMD.plot()
#plt.plot(data.BAM[0:]-72547540data.BAM[1:])
#plt.ylabel('GMD')

#plt.show()
idx.close()
tr.close()

