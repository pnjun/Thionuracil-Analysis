from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import pickle

filename = argv[1]

inpickle = open(filename, 'rb')
cfg = pickle.load(inpickle)
inpickle.close()

inplot = np.loadtxt(filename+'.txt')#, skiprows=1)

img = inplot[1:,1:]
evs = inplot[0,1:]
delays = inplot[1:,0]
print("These are evs:", evs)
print("These are delays:", delays)
cmax = np.abs(img[np.logical_not(np.isnan(img))]).max()*0.1
plt.pcolor(evs, delays ,img, cmap='bwr', vmax=cmax, vmin=-cmax)
plt.savefig(filename+'2')
plt.show()
