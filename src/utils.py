''' Moudle with helper classes for data manipulation / analysis '''
import scipy.integrate as integ
from scipy import interpolate
import numpy as np
import pandas as pd

def filterPulses(pulses, filters):
    ''' Filters a df of pulses for all rows where parameters are within the bounds defined in filt object
        Possible filters attribute are coloums of pulses, they must be a tuple of (min, max)
    '''
    queryList = []
    for name in filters.keys():
        queryList.append( "@filters.{0}[0] < {0} < @filters.{0}[1]".format(name) )

    if not queryList:
        raise KeyError("At least one filter parameter must be given")

    queryExpr = " and ".join(queryList)
    return pulses.query(queryExpr)

def correctBAM(delaysData, bamData):
    from numba import cuda
    import cupy as cp

    ''' Takes an array of delays and an array of BAM values and combines them, returing an array with the same shape of bamData offset with the delay value.
    bamData must be 25 times longer than delay. Each delay value is mapped to 25 BAM values. '''
    assert bamData.shape[0] == delaysData.shape[0] * 25, "BAM data must be 25 times longer than delays"

    #Define CUDA kernel for delay adjustment
    @cuda.jit
    def shiftBAM(bam, delay):
        bam[cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x ] += delay[cuda.blockIdx.x]
        #bam[cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x ] = delay[cuda.blockIdx.x]

    #Check BAM data integrity
    if np.isnan(bamData).any():
        raise ValueError("BAM data not complete (NaN)")

    #Copy data to GPU
    bam = cp.array(bamData)
    delay = cp.array(delaysData)
    #Shift BAM and add column with new data
    shiftBAM[delaysData.shape[0], 25](bam,delay)
    return bam.get()

class mainTofEvConv:
    ''' Converts between tof and Ev for main chamber TOF spectrometer
        Usage:
        converter = mainTofEvConv(retarder)
        energy = converter(tof)
    '''
    def __init__(self, retarder):
        self.r = retarder

        evMin = -retarder + 1
        evMax = -retarder + 350

        evRange = np.arange(evMin, evMax, 1)
        tofVals = self.ev2tof( evRange )
        self.interpolator = interpolate.interp1d(tofVals, evRange, kind='linear')

    def __call__(self, tof):
        if isinstance(tof, pd.Index):
            tof = tof.to_numpy(dtype=np.float32)
        return self.interpolator(tof)

    def ev2tof(self, e):
        #Parameters for evConversion
        l1 = 0.05      #meters
        l2 = 1.734     #meters
        l3 = 0.003
        m_over_2e = 5.69 / 2
        evOffset = 0.6 #eV

        new_e = e - evOffset
        return np.sqrt(m_over_2e) * ( l1 / np.sqrt(new_e) +
                                      l2 / np.sqrt(new_e + self.r) +
                                      l3 / np.sqrt(new_e + 300) )

class opisEvConv:
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

class Slicer:
    from contextlib import suppress
    ''' Splits a ADC trace into slices given a set of slicing parameters '''
    def __init__(self, sliceParams, removeBg = False):
        '''
        Prepares a list of indexes for tof trace slicing
        self.slices is a list of np.ranges, each one corresponding to a slice
        If params contais a 'dt' key, results are indexed by time of flight
        '''
        shotsCuts = (sliceParams.offset + sliceParams.skipNum + ( sliceParams.period * np.arange(sliceParams.shotsNum) )).astype(int)

        self.slices = shotsCuts[:,None] + np.arange(sliceParams.window)

        self.removeBg = removeBg
        self.skipNum  = sliceParams.skipNum
        if 'dt' in sliceParams.keys():
            self.dt = sliceParams.dt
        else:
            self.dt = None

    def __call__(self, tofData, pulses):
        '''
        Slices tofData and returns a dataframe of shots. Each shot is indexed by macrobunch and shotnumber
        '''
        pulseList = []
        indexList = []
        for trace, pulseId in zip( tofData, pulses.index ):
            bkg = np.mean(trace) if self.removeBg else 0

            with suppress(AssertionError):
                #Some pulses have no macrobunch number => we drop them (don't waste time on useless data)
                assert(pulseId != 0)
                pulseList.append( pd.DataFrame(  trace[self.slices] - bkg ))
                indexList.append( pulseId )

        #Create multindexed series. Top level index is macrobunch number, sub index is shot num.
        try:
            shots = pd.concat(pulseList, keys=indexList)
            shots.index.rename( ['pulseId', 'shotNum'], inplace=True )
            if self.dt: shots.columns =  ( shots.columns + self.skipNum ) * self.dt

            return shots
        except ValueError:
            return None

class evConverter:
    ''' Converts between tof and ev for a given retardation voltage
        Units are volts, electronvolts and microseconds
        Retarder must be a vectorized callable with the lenght and maxV attributes.
        Use the classmethods to create a converted already calibrated for the main tof
        or the OPIS tofs
    '''
    def __init__(self, retarder, lenght, evMin, evMax, evStep):
        print("# WARNING: evConverter is deprecated and will be removed. Use mainTofEvConv")
        self.r = retarder
        self.l = lenght

        evRange = np.arange(evMin, evMax, evStep)
        tofVals = [ self.ev2tof(ev) for ev in evRange ]
        self.interpolator = interpolate.interp1d(tofVals, evRange, kind='linear')

    @classmethod
    def mainTof(cls, retarder, mcp = None, evOffset = 0.6):
        ''' Defines the potential model for the TOF spectrometer
            offset is a global 'retardation' we apply for absolutely no reason
            except that it makes the calibration better.
        '''
        if mcp is None: mcp = retarder

        def potential(x):
            if   x < 0.05:               #free flight before retarder lens
                return -evOffset
            elif x < 1.784:               #flight inside tube
                return -evOffset + retarder
            else:                         #acceleration before MCP
                return -evOffset + mcp
        #retarder is negative! => we want electron energies above -retarder [eV]
        return cls(potential, 1.787, -retarder+evOffset+2, -retarder+evOffset+350, 1)

    def __call__(self, tof):
        return self.interpolator(tof)

    def ev2tof(self, e):
        return integ.quad( lambda x : self._integrand(x,e), 0, self.l)[0]

    def _integrand(self, x, en):
        m_over_e = 5.69
        return np.sqrt( m_over_e / 2 / ( en + self.r(x) ) ) # energy + retarder because retarder is negative!'''


if __name__ == "__main__":
    conv = opisEvConv()
    tof = np.arange(0.048, 0.244, 0.01)
    print(conv[1](tof))
