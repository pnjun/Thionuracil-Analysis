from collections import namedtuple
import warnings

import pandas as pd
import numpy as np

import numba
from numba import cuda

import math
import cupy as cp

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
from time import time

GAS_ID_AR = 0

class evFitter:
    def __init__(self, gasID):
        self.FWHM = 2  #FWHM of FEL to be used for fitting
        self.ignoreThreshold = 40. #minimum integral for a shot to be considered valid

        #Parameters of a peak for fitting: energy and relative amplitude
        if gasID == GAS_ID_AR:
                        #      energy  ampli
            self.peaksData = [ [ 15.8, 1.00 ],
                               [ 29.3, 0.21 ]]
            #where to cut the spectrum in eV
            self.evCuts = [3, 33]

            #How much samples to use at both ends of spectrum to allow for
            #offset fitting (should be a multiple of 16)
            self.padding = 32

        else:
            raise Exception("Gas id not valid")

    def loadTraces(self, data, evConv, evGuess):
        ''' Loads Tof traces and corresponding energy axis to
            the GPU memory. Each array is sliced so that only
            the relevant peak data is loaded '''
        assert(data[0].shape == data[1].shape )
        assert(data[0].shape == data[2].shape )
        assert(data[0].shape == data[2].shape )

        #Get peak bounds in samples for tof slicing. We use channel 0
        #Other channels might have sligtly different values, but
        #we are interested in a rough cut and doing it this way
        #ensures that all tofs are sliced in the same place
        baseEn = evConv[0](data[0].columns)
        s = np.abs( baseEn - (evGuess - self.evCuts[0] )).argmin()
        e = np.abs( baseEn - (evGuess - self.evCuts[1] )).argmin()

        if s < self.padding or e > data[0].shape[1] - self.padding:
            raise Exception(f"Not enough data in slice or padding too long. s:e = {s}:{e} pad:{self.padding}")
        self.ROIBounds = slice(s-2*self.padding,e+2*self.padding)

        # Will be filled with of energyvals and tofdata
        # Energies are calculated twice: once for normal tof samples
        # and once at half-step positions. This is needed because
        # oppisite spectrometers might have odd offset, resulting in a
        # tof trace that is half step ahead or behind the 'normal' tof
        # samples (since the offset is split in half between the oppisite
        # traces).
        self.traces = []
        self.energies = []
        #Iterate over the 4 TOFs
        for n, tof in enumerate(data):
            self.traces.append( tof.iloc[:,self.ROIBounds].to_numpy() )
            cols = tof.columns.to_numpy(dtype=np.float32)[self.ROIBounds]
            cols2 = cols + (cols[1] - cols[0])/2
            self.energies.append(np.array([ evConv[n](cols),
                                            evConv[n](cols2)]))

        #Put everything in a matrix and send it to GPU memory
        #Shape: (shotsNum, 4, samples). The 4 is for the 4 tofs spectrometers
        self.traces = cp.array(np.stack(self.traces, axis=1), dtype=cp.float32)
        #Shape: (4, 2, samples). 4 spectromenters, 2 for integer/halfinteger steps
        self.energies=cp.array(np.stack(self.energies,axis=0),dtype=cp.float32)

        #Integral of ROI
        self.integs = cp.linalg.norm(self.traces[:,:,self.padding:-self.padding],
                                    axis=2)

        self.shotsNum = self.traces.shape[0]
        self.traceLen = self.traces.shape[2]

    def getOffsets(self, oRange=None, getDiffs=False):
        ''' Scans the traces finding the optimal offsets (i.e. offset
            for which the difference is minimum) between tofs 0 and 2
            and 1 and 3. Orange must be lower than'''

        if not hasattr(self, 'traces'):
            raise Exception("No peak data loaded. Use load first")
        if (oRange == None):
            oRange = 2*self.padding
        if oRange > 2*self.padding:
            raise Exception("oRange must be less than padding")
        if (oRange % 32 != 0):
            warnings.warn("oRange should be a multiple of 32 for full occupancy")

        padding = self.padding
        traceLen = self.traceLen

        @cuda.jit()
        def getOffsetsDiff(tr1, tr2, diffsOut):
            '''Calculates the difference between trA and trB for all
               offsets within -oRange and +oRange. Each block does
               a shot, each thread does one offset. For full occupancy
               32 or 64 threads per block should be used.'''

            #calculate the integral of the difference.
            #Each thread does one offset
            res = 0
            off = -cuda.blockDim.x // 2 + cuda.threadIdx.x
            for i in range(padding, traceLen - padding):
                res += abs(tr1[cuda.blockIdx.x,i] - tr2[cuda.blockIdx.x,i+off])
            diffsOut[cuda.blockIdx.x, cuda.threadIdx.x] = res

        if getDiffs: ret = []
        diffs = cp.zeros( (self.shotsNum, oRange) , dtype=cp.float32)

        getOffsetsDiff[self.shotsNum, oRange]( self.traces[:,0],
                                               self.traces[:,2],
                                               diffs)
        self.xOffsets = diffs.argmin(axis=1) - oRange//2
        if getDiffs: ret.append(diffs.get())

        getOffsetsDiff[self.shotsNum, oRange]( self.traces[:,1],
                                               self.traces[:,3],
                                               diffs)
        self.yOffsets = diffs.argmin(axis=1) - oRange//2
        if getDiffs: ret.append(diffs.get())

        if getDiffs:
            return ret

    @staticmethod
    def gauss(x, center, fwhm):
            ''' Gaussian fitting function (helper)'''
            return np.exp( - 4*np.log(2.)*(x-center)**2/fwhm**2)

    def spectrumFit(self, x, fwhm):
        ret = 0.
        for p in self.peaksData:
            ret += p[1] * self.gauss(x, p[0], fwhm)
        return ret

    def leastSquare(self, ampliRange, enRange):
        if not hasattr(self, 'xOffsets'):
            raise Exception("No offset data loaded. Use getOffsets first")

        #Gaussian fitting function
        @cuda.jit("float32(float32,float32,float32)", device=True, inline=True)
        def gauss(x, center, fwhm):
                ''' Compiled gauss version for CUDA kernel '''
                return math.exp( - 4*math.log(2.)*(x-center)**2/fwhm**2)

        #Generates a fit spectrum by linear combination of gaussians
        @cuda.jit("float32(float32,float32[:,:],float32)", device=True)
        def spectrumFit(x, peaks, fwhm):
            ret = 0.
            for p in range(peaks.shape[0]):
                ret += peaks[p,1] * gauss(x, peaks[p,0], fwhm)
            return ret

        #Returns area of fit from trace integral (there is a 0.92 correlation)
        @cuda.jit("float32(int32)", device=True, inline=True)
        def integ2area(integ):
            return (integ - 20.18)*0.1549

        #Same as above but run locally
        def integ2area2(integ):
            return (integ - 20.18)*0.1549

        padding = self.padding
        traceLen = self.traceLen

        # TODO: **** MODIFY SO THAT FHWM AND AMPLI ARE FITTED
        # TOGHETER AS A FUNCTION OF TRACE INTEGRAL (USE CORRELATIONS)
        # NEED TO WRITE THE CORRELATION FUNCTION INSIDE getResiduals,
        # AND THEN TO MODIFY THE OUTPUT OF leastSquare TO TAKE THAT
        # INTO ACCOUNT

        @cuda.jit()
        def getResiduals(ener, traces, integs,
                         xOffs, yOffs, peaks,
                         ampliRange, enRange,
                         residualsOut):
            ''' Calculate residuals for fitting. Each block takes care of a
            shot. Each thread in a block takes care of a point in parameter
            space (i.e. a couple of amplitude/energy values).
            Inputs:
            The energy index corresponing to the tof data (self.energies)
            The tof data (self.traces)
            The x and y offsets
            Peaks parameters for fiter function
            Range of amplidudes and energies to try
            Output:
            Residuals for all shots and all combinations of amplidude/energy
            (3d shotnum * ampliLen * enLen array) '''

            # Params value of the current thread
            ampli = ampliRange[cuda.threadIdx.x]
            photonEn = enRange[cuda.blockIdx.y]

            #Assume fhwm from integral of trace (use leastSquare3Params to see
            #0.91 correlation between trace integral and fitted area)
            area = integ2area(integs[cuda.blockIdx.x])
            fwhm = area / ampli

            res = 0.
            for i in range(padding, traceLen-padding):
                ##TODO: FIX o1 and o2 calculation to match the one in plotFitted

                # TOFS 0 and 2
                #for even(odd) offset we use normal(halfstep) energy scale
                off = xOffs[cuda.blockIdx.x]
                par = off % 2
                o1 = off//2
                #No branching, all threads have same offset
                o2 = o1 if not par else o1 - 1

                res += ( ampli * spectrumFit( photonEn-ener[0 ,par, i + o1 ],
                                              peaks, fwhm)
                         - traces[cuda.blockIdx.x, 0, i] )**2

                res += ( ampli * spectrumFit( photonEn-ener[2 ,par, i + o2 ],
                                              peaks, fwhm)
                         - traces[cuda.blockIdx.x, 2, i + off] )**2

                # TOFS 1 and 3
                off = yOffs[cuda.blockIdx.x]
                par = off % 2
                o1 = off//2
                #No branching, all threads have same offset
                o2 = o1 if not par else o1 - 1

                res += ( ampli * spectrumFit( photonEn-ener[1 ,par, i + o1 ],
                                              peaks, fwhm)
                         - traces[cuda.blockIdx.x, 1, i] )**2
                res += ( ampli * spectrumFit( photonEn-ener[3 ,par, i + o2 ],
                                              peaks, fwhm)
                         - traces[cuda.blockIdx.x, 3, i + off] )**2

            residualsOut[cuda.blockIdx.x,          #shotnum
                         cuda.blockIdx.y,          #energy
                         cuda.threadIdx.x ] = res  #amplitude


        ampliNum = ampliRange.shape[0]
        enNum = enRange.shape[0]

        if (ampliNum*enNum % 32) != 0:
               raise Exception ("len(ampliRange) * len(enRange) must be a multiple of 32")

        #Allocate space for output array on GPU
        residuals = cp.zeros( (self.shotsNum, enNum, ampliNum ),
                               dtype=cp.float32)


        #Allocate space for amplitudes and offset arrays
        amplies = cp.array(ampliRange, dtype=cp.float32)
        energies = cp.array(enRange, dtype=cp.float32)
        integs = self.integs.sum(axis=1)
        peaks = cp.array(self.peaksData, dtype=cp.float32)

        gridDim = self.shotsNum, enNum
        blockDim =  ampliNum

        print(f'problem size {residuals.shape}')
        t=time()
        getResiduals [ gridDim , blockDim ] (self.energies, self.traces, integs,
                                             self.xOffsets, self.yOffsets, peaks,
                                             amplies, energies,
                                             residuals)
        cuda.synchronize()
        print(f'time to fit {time() - t}\n')
        #print(cp.get_default_memory_pool().used_bytes())

        #find indexes of minimum residuals for every shot
        resShape = residuals.shape
        residuals = residuals.reshape(residuals.shape[0], -1)
        residulas_idx = cp.unravel_index( residuals.argmin(axis=1), resShape[1:] )

        fittedEn    = energies [ residulas_idx[0] ]
        fittedAmpli = amplies  [ residulas_idx[1] ]
        fwhm = integ2area2(integs) / fittedAmpli

        #Extract the parameter values at the indexes found in previous step
        self.fitResults = np.array([fittedEn.get(), fittedAmpli.get(), fwhm.get()]).T

        return self.fitResults

    def ignoreMask(self, threshhold = None, getRaw=False):
        '''Returns a mask for the shots. True me
        ans a shot should be ignored'''
        if threshhold is None: threshhold = self.ignoreThreshold

        mask  = cp.any( self.integs < threshhold, axis=1 )
        if getRaw:
            return mask.get(), self.integs.get()
        else:
            return mask.get()

    def plotFitted(self, traceIdx = 0, show=True):
        if not hasattr(self, 'fitResults'):
            raise Exception("No fit results. Use leastSquare first")

        plt.figure()
        tofs = self.traces[traceIdx]
        fitEn, fitAmpli, fitFWHM  = self.fitResults[traceIdx]
        #fitAmpli, fitEn = -60, 222
        waterfall = 50

        #Tofs 0 and 2
        off = self.xOffsets.get()[traceIdx]
        par = off % 2; o1 = off//2
        o2 = o1 if not par else o1 - 1

        if off - self.padding == 0:
            trSl = slice(self.padding + off, None )
        else:
            trSl = slice(self.padding + off, -self.padding + off )

        t0En = self.energies[0 , par, self.padding+o1 : -self.padding+o1].get()
        t2En = self.energies[2 , par, self.padding+o2 : -self.padding+o2].get()

        plt.plot(t0En, tofs[0, self.padding : -self.padding].get())
        plt.plot(t2En, tofs[2, trSl ].get() )
        plt.plot(t0En, fitAmpli * self.spectrumFit(fitEn-t0En, fitFWHM) )

        #Tofs 1 and 3
        off = self.yOffsets.get()[traceIdx]
        par = off % 2; o1 = off//2
        o2 = o1 if not par else o1 - 1

        if off - self.padding == 0:
            trSl = slice(self.padding + off, None )
        else:
            trSl = slice(self.padding + off, -self.padding + off )

        t1En = self.energies[1 , par, self.padding+o1 : -self.padding+o1].get()
        t3En = self.energies[3 , par, self.padding+o2 : -self.padding+o2].get()

        plt.plot(t1En, tofs[1, self.padding : -self.padding].get() + waterfall)
        plt.plot(t3En, tofs[3, trSl ].get() + waterfall)
        plt.plot(t1En, fitAmpli * self.spectrumFit(fitEn-t1En, fitFWHM) + waterfall)

        if show:
            plt.show()

    def plotTraces(self, traceIdx = 0, shift = False, show=True):
        if not hasattr(self, 'traces'):
            raise Exception("No peak data loaded. Use loadPeak first")

        if not hasattr(self, 'xOffsets') and shift:
            raise Exception("Call getOffsets before using shift = True")
        plt.figure()
        tofs = self.traces[traceIdx]

        for n, trace in enumerate(tofs):
            o = 0
            if n == 2 and shift:
                o = self.xOffsets.get()[traceIdx]
            if n == 3 and shift:
                o = self.yOffsets.get()[traceIdx]

            waterfall = 0
            if n % 2 != 0:
                waterfall = 70

            if o - self.padding == 0:
                trSl = slice(self.padding + o, None )
            else:
                trSl = slice(self.padding + o, -self.padding + o )
            plt.plot(trace.get()[trSl] + waterfall)

        if show:
            plt.show()

    def leastSquare3Params(self, ampliRange, enRange, fwhmRange):
        if not hasattr(self, 'xOffsets'):
            raise Exception("No offset data loaded. Use getOffsets first")

        @cuda.jit("float32(float32,float32,float32)", device=True, inline=True)
        def gauss(x, center, fwhm):
                ''' Compiled gauss version for CUDA kernel '''
                return math.exp( - 4*math.log(2.)*(x-center)**2/fwhm**2)

        @cuda.jit("float32(float32,float32[:,:],float32)", device=True)
        def spectrumFit(x, peaks, fwhm):
            ret = 0.
            for p in range(peaks.shape[0]):
                ret += peaks[p,1] * gauss(x, peaks[p,0], fwhm)
            return ret

        padding = self.padding
        traceLen = self.traceLen

        @cuda.jit()
        def getResiduals(ener, traces,
                         xOffs, yOffs, peaks,
                         ampliRange, enRange, fwhmRange,
                         residualsOut):
            ''' Calculate residuals for fitting. Each block takes care of a
            shot. Each thread in a block takes care of a point in parameter
            space (i.e. a couple of amplitude/energy values).
            Inputs:
            The energy index corresponing to the tof data (self.energies)
            The tof data (self.traces)
            The x and y offsets
            Peaks parameters for fiter function
            Range of amplidudes and energies to try
            Output:
            Residuals for all shots and all combinations of amplidude/energy
            (3d shotnum * ampliLen * enLen array) '''

            # Params value of the current thread
            ampli = ampliRange[cuda.threadIdx.x]
            photonEn = enRange[cuda.threadIdx.y]
            fwhm  = fwhmRange[cuda.blockIdx.y]

            res = 0.
            for i in range(padding, traceLen-padding):
                ##TODO: FIX o1 and o2 calculation to match the one in plotFitted

                # TOFS 0 and 2
                #for even(odd) offset we use normal(halfstep) energy scale
                off = xOffs[cuda.blockIdx.x]
                par = off % 2
                o1 = off//2
                #No branching, all threads have same offset
                o2 = o1 if not par else o1 - 1

                res += ( ampli * spectrumFit( photonEn-ener[0 ,par, i + o1 ],
                                              peaks, fwhm)
                         - traces[cuda.blockIdx.x, 0, i] )**2

                res += ( ampli * spectrumFit( photonEn-ener[2 ,par, i + o2 ],
                                              peaks, fwhm)
                         - traces[cuda.blockIdx.x, 2, i + off] )**2

                # TOFS 1 and 3
                off = yOffs[cuda.blockIdx.x]
                par = off % 2
                o1 = off//2
                #No branching, all threads have same offset
                o2 = o1 if not par else o1 - 1

                res += ( ampli * spectrumFit( photonEn-ener[1 ,par, i + o1 ],
                                              peaks, fwhm)
                         - traces[cuda.blockIdx.x, 1, i] )**2
                res += ( ampli * spectrumFit( photonEn-ener[3 ,par, i + o2 ],
                                              peaks, fwhm)
                         - traces[cuda.blockIdx.x, 3, i + off] )**2

            residualsOut[cuda.blockIdx.x,        #shotnum
                         cuda.threadIdx.y,       #energy
                         cuda.threadIdx.x,       #amplitude
                         cuda.blockIdx.y] = res  #fwhm


        ampliNum = ampliRange.shape[0]
        enNum = enRange.shape[0]
        fwhmNum = fwhmRange.shape[0]

        if (ampliNum*enNum % 32) != 0:
               raise Exception ("len(ampliRange) * len(enRange) must be a multiple of 32")

        #Allocate space for output array on GPU
        residuals = cp.zeros( (self.shotsNum, enNum, ampliNum, fwhmNum ),
                               dtype=cp.float32)


        #Allocate space for amplitudes and offset arrays
        amplies = cp.array(ampliRange, dtype=cp.float32)
        energies = cp.array(enRange, dtype=cp.float32)
        fwhmes = cp.array(fwhmRange, dtype=cp.float32)
        peaks = cp.array(self.peaksData, dtype=cp.float32)

        gridDim = self.shotsNum, fwhmNum
        blockDim =  ampliNum, enNum

        print(f'problem size {residuals.shape}')
        t=time()
        getResiduals [ gridDim , blockDim ] (self.energies, self.traces,
                                             self.xOffsets, self.yOffsets,
                                             peaks, amplies, energies, fwhmes,
                                             residuals)
        cuda.synchronize()
        print(f'time to fit {time() - t}\n')
        #print(cp.get_default_memory_pool().used_bytes())

        #find indexes of minimum residuals for every shot
        resShape = residuals.shape
        residuals = residuals.reshape(residuals.shape[0], -1)
        residulas_idx = cp.unravel_index( residuals.argmin(axis=1), resShape[1:] )

        fittedEn    = energies [ residulas_idx[0] ]
        fittedAmpli = amplies  [ residulas_idx[1] ]
        fittedFwhm  = fwhmes   [ residulas_idx[2] ]

        #Extract the parameter values at the indexes found in previous step
        self.fitResults = np.array([fittedEn.get(), fittedAmpli.get(), fittedFwhm.get()]).T

        return self.fitResults

class geometricEvConv:
    ''' Converts between tof and Ev for main chamber TOF spectrometer
        Usage:
        converter = mainTofEvConv(retarder)
        energy = converter(tof)
        Uses geometric model, retardationis currently fixed to 100V
    '''
    def __init__(self, retarder):
        self.r = retarder

        evMin = self.r + 1
        evMax = self.r + 350

        evRange = np.arange(evMin, evMax, 1)
        tofVals  = [ self.ev2tof( n, evRange ) for n in range(4) ]
        self.interpolators = [ interpolate.interp1d(tof, evRange,
                                                    kind='linear') \
                               for tof in tofVals ]

    def __getitem__(self, channel):
        def foo(tof):
            if isinstance(tof, pd.Index):
                tof = tof.to_numpy(dtype=np.float32)
            return self.interpolators[channel](tof)
        return foo

    def ev2tof(self, n ,e):
        #Parameters for evConversion
        VMPec = [-234.2, -189.4, -323.1, -324.2 ]
        l = [0.0350,  0.0150,  0.0636,  0.1884,  0.0072]
        r = [0.0   ,  self.r*0.64  ,  self.r*0.84  , self.r ,  VMPec[n] ]
        m_over_2e = 5.69 / 2
        evOffset = 0 #eV

        new_e = e - evOffset

        a = [  lt / np.sqrt(new_e - rt) for lt,rt in zip(l,r) ]
        return np.sqrt(m_over_2e) * np.array(a).sum(axis=0)

class calibratedEvConv:
    ''' Converts between tof and Ev for tunnel OPIS TOF spectrometers
        Usage:
        converter = opisEvConv()
        energy = converter[channel](tof)
        Uses fixed calibration, retardation voltage is 170V
    '''
    def __init__(self):
        evMax = 350

        self.params = np.array(
                 [[ 1.87851827E+002, -1.71740614E+002, 1.68133279E+004, -1.23094641E+002, 1.85483300E+001 ],
                  [ 3.03846564E+002, -1.69947313E+002, 1.62788476E+004, -8.80818471E+001, 9.88444848E+000 ],
                  [ 2.13931606E+002, -1.71492500E+002, 1.61927408E+004, -1.18796787E+002, 1.66342468E+001 ],
                  [ 2.90336251E+002, -1.69942322E+002, 1.44589453E+004, -1.00972976E+002, 1.10047737E+001 ]])

        # generate ev ranges for each channel depeding on their retarder setting
        # (retarder is second column of params )
        evRanges = [ np.arange(-evMin+1e-3, evMax, 1) for evMin in self.params[:,1] ]
        # Calculate correspoiding TOFS for interpolation
        tofVals  = [ self.ev2tof( channel, evRange ) for channel, evRange in enumerate(evRanges) ]
        # Initialize interpolators
        self.interpolators = [ interpolate.interp1d(tof,
                                                    evRange,kind='linear') \
                               for tof, evRange in zip(tofVals, evRanges) ]

    def __getitem__(self, channel):
        def foo(tof):
            if isinstance(tof, pd.Index):
                tof = tof.to_numpy(dtype=np.float32)
            return self.interpolators[channel](tof)
        return foo

    def ev2tof(self, n, e):
        p = self.params[n]
        return (  p[4] + p[0] / np.sqrt(e + p[1]) + p[2] / ( e + p[3] )**1.5 ) / 1000
