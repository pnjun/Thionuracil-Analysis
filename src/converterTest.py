#!/usr/bin/python3

import utils
import numpy as np
import matplotlib.pyplot as plt

tofs = np.arange(0.1720, 1.6764, 0.0005)

evConv = utils.mainTofEvConv(0)
evs  = evConv(tofs)

plt.plot(tofs, evs)
plt.show()

np.savetxt('energyCal', np.array([tofs, evs]).T)
