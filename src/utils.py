''' Moudle with helper classes for data manipulation / analysis '''
import scipy.integrate as integ
from scipy import interpolate
import numpy as np
import pandas as pd

def movAvg(a, n) :
    'simple moving average'
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

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

def h5load(dfname, h5store, pulses, chunk=None):
    '''
        Loads data from dataframe 'dfname' from h5store,
        only where pulseId matches the Id of pulses.
        If chuck is passed, an iterator is returned with
        the chunklen specified
    '''
    pulsesLims = (pulses.index[0], pulses.index[-1])
    iter = True if chunk is not None else False
    return h5store.select(dfname,
           where= ['pulseId >= pulsesLims[0] and pulseId <= pulsesLims[1]',
                   'pulseId in pulses.index'],
           iterator=iter,
           chunksize= chunk )

def jacobianCorrect(tofTraces, evs, renorm=50):
    '''Corrects tofTraces using jacobian calculated from evs. renorm is used as
       an additional rescale factor'''

    jacobian = np.empty(evs.shape)
    jacobian[:-1] = -np.diff(evs)
    jacobian[-1] = jacobian[-2]

    tofTraces /= jacobian * renorm
    return tofTraces

def shotsDelay(delaysData, bamData=None, shotsNum = None):
    ''' Takes an array of delays and an array of BAM values and combines them, returing an array with the same shape of bamData offset with the delay value.
    If bamData is None no bam correction is performed, an array of delaysData.shape[0] * shotsNum size is created and returned using just delayData'''

    from numba import cuda
    import cupy as cp

    #Define CUDA kernels for delay adjustment
    @cuda.jit
    def shiftBAM(bam, delay):
        bam[cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x ] = delay[cuda.blockIdx.x] - bam[cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x ] #With BAM

    @cuda.jit
    def propagateDelay(bam, delay):
        bam[cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x ] = delay[cuda.blockIdx.x]  #No BAM

    #Check BAM data integrity
    delay = cp.array(delaysData)
    if bamData is not None:
        assert not np.isnan(bamData).any(), "BAM data not complete (NaN)"
        assert bamData.shape[0] % delaysData.shape[0]  == 0, "bamData lenght is not a integer multiple of delatysData"

        shotsNum = bamData.shape[0] // delaysData.shape[0]

        #Copy data to GPU and shift BAM
        bam = cp.array(bamData)
        shiftBAM[delaysData.shape[0], shotsNum](bam,delay)
    else:
        assert isinstance(shotsNum, int), "if bamData is none shotsNum must be a valid int"

        bam = cp.empty(delaysData.shape[0] * shotsNum)
        propagateDelay[delaysData.shape[0], shotsNum](bam,delay)

    return bam.get()


def getExct(tofTrace, frac, gmd = None):
    '''
    Calculates the excited only spectrum given the even and odd traces and an estimate of the
    excited fraction of molecules in the sample, as: (even-(1-frac)*odd)/frac.
    '''
    from numba import cuda
    import cupy as cp

    p = 1-frac

    #Cuda kernel for pump probe difference calculation
    @cuda.jit
    def tofExc(tof):
        row = 2 * cuda.blockIdx.x
        col = cuda.blockIdx.y*cuda.blockDim.x + cuda.threadIdx.x
        tof[ row, col ] = ( tof[ row, col ] - p*tof[ row + 1, col ] ) / frac
    @cuda.jit
    def tofExcGMD(tof, gmd):
        row = 2 * cuda.blockIdx.x
        col = cuda.blockIdx.y*cuda.blockDim.x + cuda.threadIdx.x
        tof[ row    , col ]  /= gmd[row]
        tof[ row + 1, col ]  /= gmd[row + 1]
        tof[ row, col ] = ( tof[ row, col ] - p*tof[ row + 1, col ] ) / frac

    #Get difference data
    tof = cp.array(tofTrace.to_numpy())

    if gmd is not None:
        #move gmd data to gpu, but only the subset corresponing to the data in tofTrace
        cuGmd = cp.array(gmd.reindex(tofTrace.index).to_numpy())
        tofExcGMD[ (tof.shape[0] // 2 , tof.shape[1] // 64) , 64 ](tof, cuGmd)
    else:
        tofExc[ (tof.shape[0] // 2 , tof.shape[1] // 64) , 64 ](tof)

    if tof.shape[1] == 3009:
        #we have an extra point left there for future interpolation, drop it
        tof[::2,-1] = 0

    return pd.DataFrame( tof[::2].get(),
                         index = tofTrace.index[::2], columns=tofTrace.columns)

def getDiff(tofTrace, gmd = None, lowPass = None):
    '''
    Function that gets a TOF dataframe and uses CUDA to calculate pump probe difference specturm
    If GMD is not None traces are normalized on gmd before difference calculation
    If lowpass is given, a brickwall lowpass filter is applied to the data
    before subtracting the traces. The lowpass value given will be the number
    of harmonics removed from the signal
    '''
    from numba import cuda
    import cupy as cp

    #Cuda kernel for pump probe difference calculation
    @cuda.jit
    def tofDiff(tof):
        row = 2 * cuda.blockIdx.x
        col = cuda.blockIdx.y*cuda.blockDim.x + cuda.threadIdx.x
        tof[ row, col ] -= tof[ row + 1, col ]
    @cuda.jit
    def tofDiffGMD(tof, gmd):
        row = 2 * cuda.blockIdx.x
        col = cuda.blockIdx.y*cuda.blockDim.x + cuda.threadIdx.x
        tof[ row    , col ]  /= gmd[row]
        tof[ row + 1, col ]  /= gmd[row + 1]
        tof[ row, col ] -= tof[ row + 1, col ]

    #Get difference data
    tof = cp.array(tofTrace.to_numpy())

    if lowPass is not None:
        len = tof.shape[1]
        tof = cp.fft.rfft(tof, axis=1)
        tof[:,-lowPass:] *= (1-cp.arange(0,1,lowPass))
        tof = cp.fft.irfft(tof, axis=1, n=len)

    if gmd is not None:
        #move gmd data to gpu, but only the subset corresponing to the data in tofTrace
        cuGmd = cp.array(gmd.reindex(tofTrace.index).to_numpy())
        tofDiffGMD[ (tof.shape[0] // 2 , tof.shape[1] // 64) , 64 ](tof, cuGmd)
    else:
        tofDiff[ (tof.shape[0] // 2 , tof.shape[1] // 64) , 64 ](tof)

    if tof.shape[1] == 3009:
        #we have an extra point left there for future interpolation, drop it
        tof[::2,-1] = 0

    return pd.DataFrame( tof[::2].get(),
                         index = tofTrace.index[::2], columns=tofTrace.columns)

def traceAverage(tofTrace, gmd=None, accumulate=False):
    import cupy as cp
    '''
    Averages all the given traces together. If gmd is given, traces are
    gmd normalized before averaging.
    If accumulated is set to true traces are just summed together, not averaged
    '''
    tof = cp.array(tofTrace.to_numpy())

    if gmd is not None:
        cuGmd = cp.array(gmd.reindex(tofTrace.index).to_numpy())
        tof /= cuGmd

    if accumulate:
        out = tof.sum(axis=0)
    else:
        out = tof.mean(axis=0)

    return out.get()

def gmdCorrect(tofTrace, gmd):
    import cupy as cp
    '''
    Normalizes traces on gmd
    '''
    tof = cp.array(tofTrace.to_numpy(), dtype=cp.float32)

    cuGmd = cp.array(gmd.reindex(tofTrace.index).to_numpy())
    tof /= cuGmd[:, None]

    return pd.DataFrame( tof.get(),
                         index = tofTrace.index, columns=tofTrace.columns)


def getInteg(tofTrace,  integSlice, gmd = None, getDiff = False):
    '''
    Integrates the traces over the given slice and returns the results.
    If GMD is not None traces are normalized on gmd before difference calculation
    If getDiff is True, the even - odd differences are returned as well
    '''
    from numba import cuda
    import cupy as cp

    tof = cp.array(tofTrace.to_numpy())

    if gmd is not None:
        @cuda.jit
        def GMDCorrect(tof, gmd):
            row = cuda.blockIdx.x
            col = cuda.blockIdx.y*cuda.blockDim.x + cuda.threadIdx.x
            tof[ row, col ]  /= gmd[row]

        cuGmd = cp.array(gmd.reindex(tofTrace.index).to_numpy())
        GMDCorrect[ (tof.shape[0], tof.shape[1] // 64) , 64 ](tof, cuGmd)

    if tof.shape[1] == 3009:
        #we have an extra point left there for future interpolation, drop it
        tof[:,-1] = 0

    integ = tof[:, integSlice].sum(axis=1)
    integDF = pd.DataFrame( integ.get(), index = tofTrace.index[:])

    if getDiff:
        diffs = integ[::2] - integ[1::2]
        diffsDF = pd.DataFrame( diffs.get(), index = tofTrace.index[::2])
        return integDF, diffsDF

    return integDF


def getROI(shotsData, limits=None):
    ''' show user histogram of delays and get ROI boundaries dragging over the plot'''
    #Show histogram and get center point for binning
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,7))

    if limits is not None:
        shotsData.query('delay > @limits[0] and delay < @limits[1]').delay.hist(bins=60)
    else:
        shotsData.delay.hist(bins=60)

    def getBinStart(event):
        global binStart
        binStart = event.xdata
    def getBinEnd(event):
        global binEnd
        binEnd = event.xdata
        plt.close(event.canvas.figure)

    plt.gcf().suptitle("Drag over ROI for binning")
    plt.gcf().canvas.mpl_connect('button_press_event', getBinStart)
    plt.gcf().canvas.mpl_connect('button_release_event', getBinEnd)
    plt.show()

    if binStart < binEnd:
        return (binStart, binEnd)
    else:
        return (binEnd, binStart)

def plotParams(shotsData):
    ''' Shows histograms of GMD and uvPower as a sanity check to the user.
        Wait for user to press ESC or close the windows before continuing.
    '''
    import matplotlib.pyplot as plt
    f = plt.figure(figsize=(10, 10))
    f.suptitle("Press ESC to continue")
    a1 = f.add_subplot(311)
    a1.title.set_text("Uv Power histogram")
    shotsData.uvPow.hist(bins=20)
    a2 = f.add_subplot(312)
    a2.title.set_text(f"GMD histogram\nAverage:{shotsData.GMD.mean():.2f}")
    shotsData.GMD.hist(bins=70)
    a3 = f.add_subplot(313)
    a3.title.set_text("BAM histogram")
    shotsData.BAM.hist(bins=15)

    f.subplots_adjust(left=None, bottom=0.05, right=None, top=0.92, wspace=0.5, hspace=0.5)

    def closeFigs(event):
        if event.key == 'escape':
            plt.close(f)
    f.canvas.mpl_connect('key_press_event', closeFigs)
    plt.show()

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

        evRange = np.arange(evMin, evMax, 0.01)
        tofVals = self.ev2tof( evRange )
        self.interpolator = interpolate.interp1d(tofVals, evRange, kind='linear')

    def __call__(self, tof):
        if isinstance(tof, pd.Index):
            tof = tof.to_numpy(dtype=np.float32)
        return self.interpolator(tof)

    def ev2tof(self, e):
        #Parameters for evConversion
        l1 = 0.09#0.09      #meters
        l2 = 1.695#1.694     #meters
        l3 = 0.002
        m_over_2e = 5.69 / 2
        evOffset = 0.55 #eV

        new_e = e - evOffset
        return np.sqrt(m_over_2e) * ( l1 / np.sqrt(new_e) +
                                      l2 / np.sqrt(new_e + self.r) +
                                      l3 / np.sqrt(new_e + 300) )

class Slicer:
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
        from contextlib import suppress
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

class beamTimeEvConv:
    ''' EV CONVERSION USED AT THE BEAMTIME. IT IS WRONG
    '''
    def __init__(self, retarder):
        self.r = retarder

    def __call__(self, tof):
        if isinstance(tof, pd.Index):
            tof = tof.to_numpy(dtype=np.float32)
        s = 1.7
        m_over_e = 5.69

        # UNITS AND ORDERS OF MAGNITUDE DO CHECK OUT
        return 0.5 * m_over_e * ( s / tof )**2 - self.r


CUSTOM_BINS_RIGHT = np.array(
             [1180,1179.8,1179.6,1179.4,1179.2,1179,1178.975,1178.95,1178.925,1178.9,
             1178.875,1178.85,1178.825,1178.8,1178.775,1178.75,1178.725,1178.7,1178.675,1178.65,
             1178.625,1178.6,1178.575,1178.55,1178.525,1178.5,1178.475,1178.45,1178.425,1178.4,
             1178.375,1178.35,1178.325,1178.3,1178.275,1178.25,1178.225,1178.2,1178.175,1178.15,
             1178.125,1178.1,1178.075,1178.05,1178.025,1178,1177.975,1177.95,1177.925,1177.9,
             1177.65,1177.4,1177.15,1176.9,1176.65,1176.4,1176.15,1175.9,1175.65,1175.4,
             1175.15,1174.9,1173.9,1172.9,1171.9,1170.9,1169.9,1168.9,1160,1130,1080])

CUSTOM_BINS_LEFT = np.array(
            [1179.8,1179.6,1179.4,1179.2,1179,1178.975,1178.95,1178.925,1178.9,1178.875,
            1178.85,1178.825,1178.8,1178.775,1178.75,1178.725,1178.7,1178.675,1178.65,1178.625,
            1178.6,1178.575,1178.55,1178.525,1178.5,1178.475,1178.45,1178.425,1178.4,1178.375,
            1178.35,1178.325,1178.3,1178.275,1178.25,1178.225,1178.2,1178.175,1178.15,1178.125,
            1178.1,1178.075,1178.05,1178.025,1178,1177.975,1177.95,1177.925,1177.9,1177.65,
            1177.4,1177.15,1176.9,1176.65,1176.4,1176.15,1175.9,1175.65,1175.4,1175.15,1174.9,
            1173.9,1172.9,1171.9,1170.9,1169.9,1168.9,1167.9,1155,1125,1075])


if __name__ == "__main__":
    conv = opisEvConv()
    tof = np.arange(0.048, 0.244, 0.01)
    print(conv[1](tof))
