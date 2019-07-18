import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from datetime import datetime
from attrdict import AttrDict
from utils import mainTofEvConv
import time
start_time = time.time()

#create a dictionary with basic parameters
cfg =	{ 'data' : 	{'path' : '/media/Fast1/ThioUr/processed/',
			#'path' : '/media/Data/ThioUr/processed/',
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

start = [datetime(2019,3,25,21,27,00).timestamp(),
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

length = len(start)

#start = min(idx.pulses.time)

stop = [datetime(2019,3,25,21,33,00).timestamp(),
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

def fit_func(x, C, A0,x0,d0,A1,x1,d1,A2,x2,d2,A3,x3,d3,A4,x4,d4,A5,x5,d5,
			A6,x6,d6,A7,x7,d7,A8,x8,d8,A9,x9,d9):
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
len_fit = 10

par =	(	5.1, 0.02, 255.0, 0.2,
		0.02, 240.0, 0.2,
		0.02, 230.0, 0.2,
		0.1, 170.0, 0.2,
		0.1, 145.0, 0.3,
		0.1, 105.0, 0.1
		)


bnd_low =[4.0, 0.01, 250., 5.0,
		0.01, 225., 2.0,
		0.01, 210., 2.0,
		0.006, 180., 4.0,
		0.006, 160., 1.0,
		0.006, 137., 3.0,
		0.006, 115., 7.0,
		0.006, 103., 1.0,
		0.006, 87., 1.0,
		0.006, 83., 1.0]

bnd_up = [5.65, 100.0, 265., 20.,
		100.0, 245., 15.,
		100.0, 220., 15.,
		100.0, 190., 10.,
		100.0, 175., 15.,
		100.0, 150., 7.,
		100.0, 135., 15.,
		100.0, 110., 5.,
		100.0, 95., 6.,
		100.0, 86., 4.]
bnd = (bnd_low, bnd_up)


### create nested lists for amplitudes, centers and widths
### first dimension is the fit component, second dimension is the spectrum
amplitudes = [[] for x in range(len_fit)]
centers = [[] for x in range(len_fit)]
widths = [[] for x in range(len_fit)]
wavelength = []

time1 = time.time() - start_time
print("time1--- %s seconds ---" % (time1))
###############################################################################

### loop for fitting
#for i in range(0,1):
for i in range(length):

	### list of macrobunch indices
	pulse = idx.select('pulses', where='time >= start[i] and time < stop[i]')
	opis = idx.select('opisEV', where = 'pulseId >= pulse.index[0] and pulseId < pulse.index[-1]')
	#print("pulse:", pulse)
	
	### shotsData, not used here
	#meta = tr.select('shotsData', where='pulseId >= pulse.index[0] and pulseId < pulse.index[-1]')
	
	### electron traces
	data = tr.select('shotsTof', where='pulseId >= pulse.index[0] and pulseId < pulse.index[-1]')
	#print(data)
	e_data = data.mean() * (-1)
	wavelength.appen(opis.mean())

	###averaged retarder
	tr_retarder= pulse.retarder.mean()

	ev_conv = mainTofEvConv(tr_retarder)
	
	evs = ev_conv(data.iloc[0].index.to_numpy(dtype=np.float32))
	
	params, paramsconv = optimize.curve_fit(fit_func, evs, e_data, bounds=bnd)
	
	#print(params)
	for k in range(1,len(params),3):	#for k in range(1,5,3):
		amplitudes[int((k-1)/3)].append(params[k])
		centers[int((k-1)/3)].append(params[k+1])
		widths[int((k-1)/3)].append(params[k+2])
	
	#print("Amplitudes:", amplitudes)
	
	plt.figure(i)
	plt.plot(evs, e_data)
	plt.plot(evs, fit_func(evs, *params))
	#pulse.retarder.plot()
	#plt.plot([(pulse.index[0],0),(pulse.index[-1],0)],[GMD_set[i],GMD_set[i]], color = 'r')
	#meta.GMD.plot()
	#plt.plot((pulse.index[0],pulse.index[-1]),(GMD_set[i],GMD_set[i]))
	#plt.plot(pulse,meta.GMD)
	#plt.text(pulse.index[0],1,'i')
	#plt.ylabel('Retarder voltage in V')

time2 = time.time() - time1 - start_time
print("time2--- %s seconds---" % (time2))
print(i)
fignum = i+1
#print("length of params:", len(params))


plt.figure(fignum)
wavelgth = plt.plot(wavelength, label = 'wavelength')
plt.legend()
fignum +=1

plt.figure(fignum)

plt.plot(amplitudes[0])
plt.plot(amplitudes[1])
plt.plot(amplitudes[2])
plt.plot(amplitudes[3])
plt.plot(amplitudes[4])
plt.plot(amplitudes[5])
plt.plot(amplitudes[6])
plt.plot(amplitudes[7])
plt.plot(amplitudes[8])
plt.plot(amplitudes[9])
plt.title("Amplitudes")
fignum +=1
time3 = time.time() - time2 - start_time
print("time 3--- %s seconds---" % (time3))

plt.figure(fignum)

cent0 = plt.plot(centers[0], label = 'center0')
cent1 = plt.plot(centers[1], label = 'center1')
cent2 = plt.plot(centers[2], label = 'center2')
plt.title("Centers 0 to 2")
plt.legend()
fignum +=1

plt.figure(fignum)
cent3 = plt.plot(centers[3], label = 'center3')
cent4 = plt.plot(centers[4], label = 'center4')
plt.title("Centers 3 and 4")
plt.legend()
fignum +=1

plt.figure(fignum)
cent5 = plt.plot(centers[5], label = 'center5')
cent6 = plt.plot(centers[6], label = 'center6')
plt.title("Centers 5 and 6")
plt.legend()
fignum +=1

plt.figure(fignum)
cent7 = plt.plot(centers[7], label = 'center7')
cent8 = plt.plot(centers[8], label = 'center8')
cent9 = plt.plot(centers[9], label = 'center9')
plt.title("Centers 7 to 9")
plt.legend()
fignum +=1

plt.figure(fignum)
width0 = plt.plot(widths[0], label = 'width0')
width1 = plt.plot(widths[1], label = 'width1')
widht2 = plt.plot(widths[2], label = 'widht2')
plt.title("Widths 0 to 2")
plt.legend()
fignum +=1

plt.figure(fignum)
widht3 = plt.plot(widths[3], label = 'width3')
width4 = plt.plot(widths[4], label = 'width4')
plt.title("Widths 3 and ")
plt.legend()
fignum +=1

plt.figure(fignum)
widht5 = plt.plot(widths[5], label = 'width5')
width6 = plt.plot(widths[6], label = 'width6')
plt.title("Widths 5 and 6")
plt.legend()
fignum +=1

plt.figure(fignum)
widht7 = plt.plot(widths[7], label = 'width7')
width8 = plt.plot(widths[8], label = 'width8')
width9 = plt.plot(widths[9], label = 'width9')
plt.title("Widths 7 to 9")
plt.legend()
fignum +=1

#plt.legend(handles=[width0, width1])
#plt.legend((line1, line2),(width0, width1))

plt.figure(fignum)
#plt.plot(np.subtract(centers[0]-centers[1]-centers[2]))
plt.plot([a-b for a,b in zip(centers[0],centers[1])])
plt.title("Difference peaks 0 and 1")
fignum +=1

plt.figure(fignum)
plt.plot([a-b for a,b in zip(centers[0],centers[2])])
plt.title("Difference peaks 0 and 2")
fignum +=1

plt.figure(fignum)
plt.plot([a-b for a,b in zip(centers[1],centers[2])])
plt.title("Difference peaks 1 and 2")
fignum +=1

#plt.figure(fig+5)
#plt.plot(centers[
plt.figure(fignum)
plt.plot([a-b for a,b in zip(centers[5],centers[6])])
plt.title("Difference peaks 5 and 6")
fignum +=1
#map(lambda x:plt.plot(amplitudes[x]),range(0,len(amplitudes[1])))

time4 = time.time() - time3 - start_time
print("time 4--- %s seconds" % (time4))

#plt.figure(fig+4)
#plt.plot(GMD_set)
#plt.show()
#stop = max(idx.pulses.time)-5000

#pulse = idx.select('pulses', where='time >=start and time < stop')

#print("BAM stop:", pulse.loc[71344215].time, "BAM restart:", pulse.loc[71797868].time)

#data = tr.select('shotsData', where='pulseId >= 72547540 and pulseId < 72547550')



###plot data
#plt.figure(1)
#data.GMD.plot()
#plt.plot(data.BAM[0:]-72547540data.BAM[1:])
#plt.ylabel('GMD')

plt.show()
idx.close()
tr.close()

