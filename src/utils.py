''' Moudle with helper classes for data manipulation / analysis '''
import scipy as sp
import numpy as np

class evConverter:
    def __init__(self, potential, lenght):
        self.v = potential
        self.l = lenght
                
    def __call__(self, tof):
        pass       

    def _integrand(self, x, e):
        return np.sqrt( 1 / ( e - self.v(x) ) )

    def e2tof(self, e):
        return sp.integrate.quad( lambda x : self._integrand(x,e), 0, self.l)


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
