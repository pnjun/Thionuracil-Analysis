''' Moudle with helper classes for data manipulation / analysis '''
import scipy.integrate as integ
from scipy import interpolate
import numpy as np

class evConverter:
    ''' Converts between tof and ev for a given retardation voltage 
        Units are volts, electronvolts and microseconds
        Retarder must be a vectorized callable with the lenght and maxV attributes.
        Use the classmethods to create a converted already calibrated for the main tof 
        or the OPIS tofs
    '''
    def __init__(self, retarder, lenght, evMin, evMax, evStep):
        self.r = retarder
        self.l = lenght
        
        evRange = np.arange(evMin, evMax, evStep)
        tofVals = [ self.ev2tof(ev) for ev in evRange ]
        self.interpolator = interpolate.interp1d(tofVals, evRange, kind='linear')
        
    @classmethod
    def mainTof(cls, retarder):
        ''' Defines the potential model for the TOF spectrometer
            offset is a global 'retardation' we apply for absolutely no reason
            except that it makes the calibration better.
        '''
        offset = 0.7
        def potential(x):
            if   x < 0.087:               #free flight before retarder lens
                return 0 - offset   
            elif x < 1.784:               #flight inside tube
                return retarder - offset
            else:                         #acceleration before MCP
                return 300 - offset
        #retarder is negative! => we want electron energies above -retarder [eV] 
        return cls(potential, 1.787, -retarder+offset+2, -retarder+offset+300, 2)
        

    def __call__(self, tof):
        return self.interpolator(tof)
    
    def ev2tof(self, e):
        return integ.quad( lambda x : self._integrand(x,e), 0, self.l)[0]

    def _integrand(self, x, en):
        m_over_e = 5.69
        return np.sqrt( m_over_e / 2 / ( en + self.r(x) ) ) # energy + retarder because retarder is negative!
        



@np.vectorize



class Slicer:
    ''' Splits a ADC trace into slices given a set of slicing parameters '''
    def __init__(self, sliceParams, tof2ev = None, removeBg = False):
        ''' 
        Prepares a list of indexes for tof trace slicing
        self.slices is a list of np.ranges, each one corresponding to a slice
        '''
        shotsCuts = (sliceParams.offset + sliceParams.skipNum + ( sliceParams.period * np.arange(sliceParams.shotsNum) )).astype(int)
 
        self.slices = shotsCuts[:,None] + np.arange(sliceParams.window)
        
        self.removeBg = removeBg
        self.skipNum  = sliceParams.skipNum
        self.tof2ev = tof2ev # Time in us between each sample point (used for ev conversion, if None no ev conversion is done)
        
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
            
            #Name columns as tof to eV conversion
            if self.tof2ev: shots.columns = self.Tof2eV( ( shots.columns + self.skipNum ) ) 
            
            return shots
        except ValueError:
            return None
                        
    def Tof2eV(self, tof):
        ''' converts time of flight into ectronvolts '''
        # Constants for conversion:
        tof *= self.tof2ev.dt
        s = self.tof2ev.len
        m_over_e = 5.69
        # UNITS AND ORDERS OF MAGNITUDE DO CHECK OUT    
        return 0.5 * m_over_e * ( s / ( tof ) )**2
