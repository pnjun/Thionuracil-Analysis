#!/usr/bin/python3

import utils
import numpy as np
import matplotlib.pyplot as plt

tofs = np.arange(0.25, 1.65, 0.0005)

retarders = [0,20,40,60,80]

plt.rcParams.update({'font.size': 14})

for ret in retarders:
    evConv = utils.mainTofEvConv(-ret)
    evs  = evConv(tofs)
    plt.plot(tofs, evs, label=f"{ret} V")

plt.gca().set_ylabel('Energy [eV]')
plt.gca().set_xlabel('Time-of-flight [μs]')
plt.legend()
plt.show()

#np.savetxt('energyCal', np.array([tofs, evs]).T)
