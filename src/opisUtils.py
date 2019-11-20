from collections import namedtuple

import pandas as pd
import numpy as np
from numba import cuda
import cmath
import cupy as cp

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate

GAS_ID_AR = 0

class evFitter:
    def __init__(self, gasID):
        #Parameters of a peak for fitting: energy, relative amplitude and
        #range to use for fitting at either side of peak position (in ev)
        Peak = namedtuple('Peak', 'en ampl width')
        if gasID == GAS_ID_AR:
            self.peaksData = [ Peak( 15.8, 1.00, 6 ),
                               Peak( 29.3, 0.21, 3 ) ]

        else:
            raise Exception("Gas id not valid")

    def loadPeaks(self, data, energies, undulEv):
        assert(data[0].shape == data[1].shape )
        assert(data[0].shape == data[2].shape )
        assert(data[0].shape == data[2].shape )
        self.shotsNum = data[0].shape[0]
        # Will be filled with tuples of ( energyvals, tofdata )
        # for each peak. energyvals and tofdata are hanles to
        # cuda arrays storing the data
        self.peaks = []

        #Iteration on peaks
        for peak in self.peaksData:
            tofs = []

            #For each peak, we have one energy and tof couple for
            #each OPIS channel.
            for tof, ener in zip(data, energies):
                #start and end indexes of peak data
                s = np.abs( ener - (undulEv - peak.en + peak.width) ).argmin()
                e = np.abs( ener - (undulEv - peak.en - peak.width) ).argmin()

                #create cupy arrays with tof and energy data and store
                #hanled in a tuple
                tofs.append( ( cp.array(ener[s:e], dtype=cp.float32)),
                               cp.array(tof.iloc[:,s:e].to_numpy(),
                                        dtype=cp.float32)) ) )

            self.peaks.append( tofs )

    def leastSquare(self, ampliRange, enRange, FWHM = 2):
        if not hasattr(self, 'peaks'):
            raise Exception("No peak data loaded. Use loadPeak first")

        @cuda.jit(device=True, inline=True)
        def gauss(x, center, fwhm):
            ''' Iniline CUDA gaussian helper function '''
            return cmath.exp( - 4*cmath.log(2)*(x-center)**2/fwhm**2)

        @cuda.jit()
        def getResiduals(energyIn, tracesIn, ampliesIn, offsetsIn,
                         residualsOut):
            ''' Calculate residuals for fitting. Each block takes care of a
            shot. Each thread in a block takes care of a point in parameter
            space (i.e. a couple of amplitude/offset values).
            Inputs:
            The energy index corresponing to the tof data (1d shotlen array)
            The tof data (2d shotnum*shotlen array)
            The array of amplidude values to be tested (1d n array)
            The array of energy offset values to be tested (1d m array)
            Output:
            Residuals for all shots and all combinations of amplidude and
            offset (3d shotnum * n *m array) '''

            len = energyIn.shape[0]
            ampli    = ampliesIn[cuda.threadIdx.x]
            centerEn = offsetsIn[cuda.threadIdx.y] + energyIn[len//2]

            residual = 0
            for i in range(len):
                residual += ( traces[cuda.blockIdx.x, i] -
                              ampli * gauss( energy[i], centerEn, FWHM ))**2

            residuals[cuda.blockIdx.x,
                      cuda.threadIdx.x,
                      cuda.threadIdx.y] = residual

        #Allocate space for output array on GPU
        residuals = cp.zeros( (self.shotsNum,
                               ampliRange.shape[0],
                               enRange.shape[0] ), dtype=cupy.float32) )

        peakId = 0

        #Allocate space for amplitudes and offset arrays
        amplies = cp.array(ampliRange, dtype=cp.float32)
        offsets = cp.array(enRange, dtype=cp.float32)

        en, traces = self.peaks[peakId]
        gridDim = traces.shape[0]
        blockDim =  amplies.shape[0], offsets.shape[0]
        getResiduals [ gridDim , blockDim ] (en, traces,
                                             amplies, offsets, residuals)

        return residuals.get()

    def plotPeaks(self, traceIdx = 0, show=True):
        if not hasattr(self, 'peaks'):
            raise Exception("No peak data loaded. Use loadPeak first")

        for peak in self.peaks:
            for en, trace in peak:
                plt.plot(en.get(), trace[traceIdx].get())
        if show:
            plt.show()

class evConv:
    ''' Converts between tof and Ev for tunnel OPIS TOF spectrometers
        Usage:
        converter = opisEvConv()
        energy = converter[channel](tof)
    '''
    def __init__(self):
        evMax = 350

        params = np.array(
                 [[ 1.87851827E+002, -1.71740614E+002, 1.68133279E+004, -1.23094641E+002, 1.85483300E+001 ],
                  [ 3.03846564E+002, -1.69947313E+002, 1.62788476E+004, -8.80818471E+001, 9.88444848E+000 ],
                  [ 2.13931606E+002, -1.71492500E+002, 1.61927408E+004, -1.18796787E+002, 1.66342468E+001 ],
                  [ 2.90336251E+002, -1.69942322E+002, 1.44589453E+004, -1.00972976E+002, 1.10047737E+001 ]])

        # generate ev ranges for each channel depeding on their retarder setting
        # (retarder is second column of params )
        evRanges = [ np.arange(-evMin+1e-3, evMax, 1) for evMin in params[:,1] ]
        # Calculate correspoiding TOFS for interpolation
        tofVals  = [ self.ev2tof( channel, evRange ) for channel, evRange in zip(params, evRanges) ]
        # Initialize interpolators
        self.interpolators = [ interpolate.interp1d(tof, evRange, kind='linear') \
                               for tof, evRange in zip(tofVals, evRanges) ]

    def __getitem__(self, channel):
        def foo(tof):
            if isinstance(tof, pd.Index):
                tof = tof.to_numpy(dtype=np.float32)
            return self.interpolators[channel](tof)
        return foo

    def ev2tof(self, p, e):
        return (  p[4] + p[0] / np.sqrt(e + p[1]) + p[2] / ( e + p[3] )**1.5 ) / 1000
