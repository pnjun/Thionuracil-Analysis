#!/usr/bin/python3
from collections import namedtuple
import warnings

import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import opisUtils as ou

FITNEW=False
ch  = 0
len = 300

if FITNEW == True:
    def getConv(n, ret):

        def ev2tof(e, l1,l2,l3,l4,l5):
            #Parameters for evConversion
            VMPec = [-234.2, -189.4, -323.1, -324.2 ]
            r = [0.0   ,  ret*0.64  ,  ret*0.84  , ret ,  VMPec[n] ]
            m_over_2e = 5.69 / 2

            evOffset = -(n % 2)/2 # <--------------- CAREFUL HERE
            new_e = e - evOffset

            l = [l1,l2,l3,l4,l5]
            a = [  lt / np.sqrt(new_e - rt) for lt,rt in zip(l,r) ]
            return np.sqrt(m_over_2e) * np.array(a).sum(axis=0)

        return ev2tof

    l_p =  (0.0350,  0.0150,  0.0636,  0.1884,  0.0072)

    l_b = ([0.0100,  0.0100,  0.0200,  0.1500,  0.0000],
           [0.0600,  0.0750,  0.0850,  0.3000,  0.0150])

    en0 = np.linspace(55,200,len)
    calConv0  = ou.calibratedEvConv(ou.CONVCAL_50)
    geoConv0  = ou.geometricEvConv(50, ou.GEOMOD_ORIG)
    testConv0 = getConv(ch, 50)
    calData0  = calConv0.ev2tof(ch,en0)
    geoData0  = geoConv0.ev2tof(ch,en0)

    en1 = np.linspace(175,250,len)
    calConv1  = ou.calibratedEvConv()
    geoConv1  = ou.geometricEvConv(170, ou.GEOMOD_ORIG)
    testConv1 = getConv(ch, 170)
    calData1  = calConv1.ev2tof(ch,en1)
    geoData1  = geoConv1.ev2tof(ch,en1)

    def combinedFIt(en, l1,l2,l3,l4,l5):
        return np.hstack([testConv0(en[:len], l1,l2,l3,l4,l5),
                          testConv1(en[len:], l1,l2,l3,l4,l5)])

    popt, pconv = curve_fit(combinedFIt, np.hstack([en0,en1]),
                                         np.hstack([calData0,calData1]),
                                         bounds = l_b)
    p_0 = np.array(l_p)
    print(popt , popt.sum())
    print(p_0 ,  p_0.sum() )
    print()

    plt.figure()

    fitData0 = testConv0(en0, *popt)
    plt.plot(en0, calData0, label='calibrated')
    plt.plot(en0, geoData0, label='geometric')
    plt.plot(en0, fitData0,  label='fitted')
    plt.legend()
    print("ret 50V, cal-fit", (calData0 - fitData0).sum() )
    print("ret 50V, cal-geo", (calData0 - geoData0).sum() )

    plt.figure()


    fitData1 = testConv1(en1, *popt)
    plt.plot(en1, calData1, label='calibrated')
    plt.plot(en1, geoData1, label='geometric')
    plt.plot(en1, fitData1,  label='fitted')
    plt.legend()

    print("ret 170V, cal-fit", (calData1 - fitData1).sum() )
    print("ret 170V, cal-geo", (calData1 - geoData1).sum() )

    plt.figure()
    calData = np.hstack([calData0,calData1])
    fitData = np.hstack([fitData0,fitData1])
    geoData = np.hstack([geoData0,geoData1])

    plt.plot(np.hstack([en0,en1]), calData - fitData, label='cal-fit')
    plt.plot(np.hstack([en0,en1]), calData - geoData, label='cal-geo')
    plt.legend()


else:
    #Sanitiy check
    plt.figure()
    ent = np.linspace(105,170,len)
    geoConvO = ou.geometricEvConv(100, ou.GEOMOD_ORIG)
    geoConvF = ou.geometricEvConv(100, ou.GEOMOD_FIT)
    geoOData = geoConvO.ev2tof(ch,ent)
    geoFData = geoConvF.ev2tof(ch,ent)

    plt.plot(ent, geoOData, label='orig')
    plt.plot(ent, geoFData, label='fit')
    plt.legend()

    plt.figure()
    tofs = np.arange(250,1950)*0.00014
    evO = geoConvO[ch](tofs)
    evF = geoConvF[ch](tofs)
    plt.plot(tofs, evO, label='orig')
    plt.plot(tofs, evF, label='fit')
    plt.legend()

    plt.figure()
    tofs = np.arange(250,1950)*0.00014
    evBase = geoConvO[ch](tofs)
    evOffs = geoConvF[ch](tofs, 32*0.00014/2)
    plt.plot(tofs, evBase, label='base')
    plt.plot(tofs, evOffs, label='offset')
    plt.legend()

plt.show()
