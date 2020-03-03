#!/usr/bin/python3
"""
Script for analysing sulfur 2p photoline

script uses csv files created with delayScan.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from attrdict import AttrDict

cfg = { "data" : { "folder" : "/home/dmayer/Schreibtisch/Thiouracil XPS/data/",
                   "file"   : "wp_10.2-random.csv", # only processed data from delayScan.py
                   },
        "out"  : { "img"        : "/home/dmayer/Schreibtisch/Thiouracil XPS/Plots/",
                   "fit"        : "/home/dmayer/Schreibtisch/Thiouracil XPS/data/Fits/",
                   "tag"        : "wp_10.2-tripleGauss",
                   "plotFormat" : "png",
                   "dpi"        : 300
                   },
        "pltOnly" : False, # requires saved fit parameters
        "plot" : {"show"    : False, # True of plots defined below shall be shown after script finished
                  "diffLine": True, # 2d plot for each delay with data and fit
                  "diffMap" : True, # colour plot with Ekin and delays
                  "fitAmp"  : True, # plot amplitudes from fit
                  "fitPeak" : True, # plot peaks from fit
                  "fitWidth": True, # plot width from fit
                  "zeroCr"  : True, # calculate & plot zero crossing
                  "area"    : True, # calculate & plot area, REQUIRES ZERO CROSSING
                  "PLCorr"  : False # correct for shift in depletion feature
                  },
        "ROI"  : { "ekin"  : (90, 110),  # kinetic energy at which the photoline is located
                   "delay" : (-0.3, 10.1)  # delays of interest
                   },
        "tZero"  : { "correct" : True,
                     "val" : 1180.
                     },  # time zero estimate
        "fit" : { "gauss" : 3,
                  "guess" : (2., 100.5, 1.2, 6., 103., 1.5, 2., 98.5, 1.2, 0.05),  # guess for gaussian fit, second and fifth value should be adjusted if switched from KINETIC to BINDING ( and v.v.)
                  "analyticZero" : False
                  }
        }
cfg = AttrDict(cfg)

color1 = "tab:red"
color2 = "tab:blue"
color3 = "tab:orange"

# just a bunch of functions
def gaussian(x, Amp, zero, width, const):
    return Amp * np.exp(-0.5 * np.square((x - zero) / width)) + const


def twoGaussianDiff(x, A1, x1, b1, A2, x2, b2, const):
    return gaussian(x, A1, x1, b1, const=0) - gaussian(x, A2, x2, b2, const=0) + const


def threeGaussianDiff(x, A1, x1, b1, A2, x2, b2, A3, x3, b3, const):
    return gaussian(x, A1, x1, b1, const=0) - gaussian(x, A2, x2, b2, const=0) + gaussian(x, A3, x3, b3, const=0) + const


def threeGaussianDiffDeriv(x, A1, x1, b1, A2, x2, b2, A3, x3, b3):
    return gaussian(x, A1, x1, b1, const=0)*(x-x1)/b1**2 - gaussian(x, A2, x2, b2, const=0)*(x-x2)/b2**2 + gaussian(x, A3, x3, b3, const=0)*(x-x3)/b3**2


def absMinZero(series, estMin=None, func="linear", n=3):
    """
    Takes pandas.series object and estimates its root by looking for minimum value in abs(series) and linear regression
    """
    if estMin is not None:

        max1 = np.argmax(abs(series.values[:estMin]))
        max2 = np.argmax(abs(series.values[estMin:])) + estMin
        minIdx = np.argmin(abs(series.values[max1: max2])) + max1
    else:
        half = int(series.size // 2)
        max1 = np.argmax(abs(series.values[:half]))
        max2 = np.argmax(abs(series.values[half:])) + half
        minIdx = np.argmin(abs(series.values[max1: max2])) + max1

    if n % 2 == 0 or n < 3:
        print("ValueError in absMin: n must be odd integer >= 3")
        exit()
    else:
        idx = [minIdx + q for q in range(-(n-1)//2, (n-1)//2+1)]
        xval = series.index.values[idx].astype(float)
        yval = series.values[idx]

    if func == "linear":
        p, c = np.polyfit(xval, yval, 1, cov=True)
        c = np.sqrt(np.diag(c))
        x0 = - p[1] / p[0]
        ux0 = (abs(c[1] / p[1]) + abs(c[0] / p[0])) * abs(x0)

    elif func == "cuberoot":
        f = lambda x, a, b: a*(x - b)**(1/3)
        guess = (1., 100.)
        p, c = curve_fit(f, xval, yval, p0=guess)
        c = np.sqrt(np.diag(c))
        x0 = p[1]
        ux0 = c[1]

    else:
        print("Error in method for deriving zero-crossing. Unknown parameter for func.")
        exit()

    return [x0, ux0]


def analyticZero(params, roi, error=None):
    """
    Takes params from double gaussian fit. Params and error must be lists
    """
    a1, x1, b1, a2, x2, b2, const = abs(params)
    a = 1/np.square(b1) - 1/np.square(b2)
    b = -2*( x1/np.square(b1) - x2/np.square(b2) )
    c = np.square(x1)/np.square(b1) - np.square(x2)/np.square(b2) - np.log(a1 / a2)
    x01 = (-b + np.sqrt(np.square(b) - 4 * a * c)) / (2*a)
    x02 = (-b - np.sqrt(np.square(b) - 4 * a * c)) / (2*a)
    if error is not None:

        # calculate errors for a, b and c
        ea1, ex1, eb1, ea2, ex2, eb2, eco = abs(error)
        ea = 2 * (eb1 / b1 ** 3 + eb2 / b2 ** 3)
        eb = 2 * (x1 / b1 ** 2 * (ex1 / x1 + 2 * eb1 / b1) + x2 / b2 ** 2 * (ex2 / x2 + 2 * eb2 / b2))
        ec = 2 * ((x1 / b1) ** 2 * (ex1 / x1 + eb1 / b1) + (x2 / b2) ** 2 * (ex2 / x2 + eb2 / b2)) + ea1 / a1 + ea2 / a2

        # redefine x0
        alpha = 0.5 * b / a
        beta = b ** 2 - 4 * a * c
        gamma = 2 * a

        ealpha = (abs(ea / a) + abs(eb / b)) * alpha
        ebeta = abs(2 * b * eb) + 4 * (abs(a * ec) + abs(c * ea))
        egamma = gamma * abs(ea / a)

        # calculate error for x0
        ex0 = abs(ealpha + 0.5 * ebeta / (np.sqrt(beta) * gamma) + egamma * np.sqrt(beta) / gamma ** 2)

    else:
        ex0 = None

    if x01 == x02:
        x0 = x01
    elif roi[0] < x01 < roi[1]:
        x0 = x01
    elif roi[0] < x02 < roi[1]:
        x0 = x02
    else:
        x0 = np.NaN
        ex0 = None

    return [x0, ex0]


def newtonGauss(x, params, gauss=2, guess=None):

    if gauss == 2:
        f = lambda y: twoGaussianDiff(y, *params)
        fprime = lambda y: twoGaussianDiffDeriv(y, *params[:-1])
    elif gauss == 3:
        f = lambda y: threeGaussianDiff(y, *params)
        fprime = lambda y: threeGaussianDiffDeriv(y, *params[:-1])
    else:
        print("Unknown difference modell: Number of gaussian has to be 2 or 3.")
        exit()

    if guess == None:
        xmin = x[np.argmin(f(x))]
        xmax = x[np.argmax(f(x))]
        guess = (xmin + xmax) / 2


    prevGuess = guess+0.1

    it = 0
    while abs(prevGuess - guess) > 1e-5:
        it += 1
        prevGuess = guess
        guess = guess - (f(guess) / fprime(guess))
        if it > 10000:
            print(f"Iteration reached 10000 steps. Current difference is {abs(prevGuess - guess)}. Stopping!")
            return [np.NaN, None]

    return [guess, None]


def integral(series):
    a = 0
    for i in range(len(series)-1):
        a += series.values[i]*(series.index.values.astype(float)[i+1] - series.index.values.astype(float)[i])
    return a


def calc_area(series, zero, sigma1=None, sigma2=None):

    if sigma1 is not None:
        s1 = series[series.index.values.astype(float) < zero & series.index.values.astype(float) >= zero-3*sigma1]
    else:
        s1 = series[series.index.values.astype(float) < zero]
    if sigma2 is not None:
        s2 = series[series.index.values > zero & series.index.values.astype(float) <= zero+3*sigma2]
    else:
        s2 = series[series.index.values.astype(float) > zero]

    a1 = integral(s1)
    a2 = integral(s2)

    return [a1, a2]


print("Loading data ...")
data = pd.read_csv(cfg.data.folder + cfg.data.file, sep=",", index_col=0)
if cfg.tZero.correct:
    data.index = cfg.tZero.val - data.index  # shift delays
delay = data.index

dMask = (data.index.values > cfg.ROI.delay[0]) & (data.index.values < cfg.ROI.delay[1])
data = data[dMask]
eMask = (data == data) & (data.columns.astype(float) >= cfg.ROI.ekin[0]) & (data.columns.astype(float) <= cfg.ROI.ekin[1])
data = data[eMask].T.dropna().T

energy = np.arange(cfg.ROI.ekin[0], cfg.ROI.ekin[1], 0.1)

if cfg.pltOnly == False:
### Fit difference spectra to double gaussian function ###
    mOpt = []
    mCov = []
    fit  = []

    print("Start with fitting gaussian modell...")
    for i in range(len(data.values)):

        try:
            if cfg.fit.gauss == 2:
                pOpt, pCov = curve_fit(twoGaussianDiff, data.columns.values, data.values[i], p0=cfg.fit.guess)
                fit.append(twoGaussianDiff(energy, *pOpt))
            else:
                pOpt, pCov = curve_fit(threeGaussianDiff, data.columns.values, data.values[i], p0=cfg.fit.guess)
                fit.append(threeGaussianDiff(energy, *pOpt))

            pCov = np.sqrt(np.diag(pCov))
            mOpt.append(pOpt)
            mCov.append(pCov)

        except RuntimeError:
            print(f"RuntimeError at delay {data.index.values[i]} ps")
            if cfg.fit.gauss == 2:
                mOpt.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
                mCov.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
                fit.append([np.NaN for i in range(len(energy))])
            else:
                mOpt.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
                mCov.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
                fit.append([np.NaN for i in range(len(energy))])

    if cfg.fit.gauss == 2:
        mOpt = pd.DataFrame(data=np.array(mOpt).T, index=["A1", "x1", "w1", "A2", "x2", "w2", "const"], columns=data.index.values).T.dropna().T
        mCov = pd.DataFrame(data=np.array(mCov).T, index=["A1", "x1", "w1", "A2", "x2", "w2", "const"], columns=data.index.values).T.dropna().T
    else:
        mOpt = pd.DataFrame(data=np.array(mOpt).T, index=["A1", "x1", "w1", "A2", "x2", "w2", "A3", "x3", "w3", "const"], columns=data.index.values).T.dropna().T
        mCov = pd.DataFrame(data=np.array(mCov).T, index=["A1", "x1", "w1", "A2", "x2", "w2", "A3", "x3", "w3", "const"], columns=data.index.values).T.dropna().T

    fit = pd.DataFrame(data=np.array(fit), index=data.index.values, columns=energy).dropna()

    mOpt.to_csv(cfg.out.fit+cfg.out.tag+"_diffMapOptParam.csv", sep=",")
    mCov.to_csv(cfg.out.fit+cfg.out.tag+"_diffMapOptError.csv", sep=",")

else:
    print("Load fit parameters...")
    mOpt = pd.read_csv(cfg.out.fit+cfg.out.tag+"_diffMapOptParam.csv", sep=",", index_col=0)
    mCov = pd.read_csv(cfg.out.fit+cfg.out.tag+"_diffMapOptError.csv", sep=",", index_col=0)
    mOpt.columns = mOpt.columns.values.astype(float)
    mCov.columns = mCov.columns.values.astype(float)
    if mOpt.index.size % 2 == 1:
        func = lambda y, p: twoGaussianDiff(y, *p)
    elif mOpt.index.size % 3 == 1:
        func = lambda y, p: threeGaussianDiff(y, *p)
    else:
        print(f"Unknown modell. Expected 7 or 11 parameters. Got {mOpt.index.size}.")

    fit = [ func(energy, mOpt.T.values[i]) for i in range(mOpt.columns.size)]
    fit = pd.DataFrame(data=fit, columns=energy, index=mOpt.columns)


print(f"Offsets derived from fit: {mOpt.T.const.mean():.3f} +/- {mOpt.T.const.std():.3f}")

#################################
### Difference map comparison ###
#################################
if cfg.plot.diffMap:
    print("Plot difference map...")

    if cfg.plot.diffLine:
        for i in range(fit.shape[0]):
            fx = plt.figure()
            id = fit.index[i]
            plt.plot(data.columns.values.astype(float), data.loc[id])
            plt.plot(energy, fit.iloc[i])
            plt.xlabel("Kinetic energy (eV)")
            plt.ylabel("Difference signal")
            plt.title(f"delay {id:.4f} ps")
            plt.savefig(cfg.out.img+cfg.out.tag+f"_delay_{id:.3f}."+cfg.out.plotFormat, dpi=cfg.out.dpi)
            plt.close(fx)

    f1, ax1 = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    ax1[0].set_title("Data")
    ax1[0].set_xlabel("Kinetic Energy (eV)")
    ax1[1].set_xlabel("Kinetic Energy (eV)")

    vMax = np.max(abs(data.values)) * 0.75
    cm = ax1[0].pcolormesh(data.columns.values.astype(float), data.index.values, data.values, cmap="bwr", vmax=vMax, vmin=-vMax)

    ax1[0].set_ylim(cfg.ROI.delay[0], cfg.ROI.delay[1])
    ax1[0].set_ylabel("Delay (ps)")

    ax1[1].pcolormesh(fit.columns.values, fit.index.values, fit.values, cmap="bwr", vmax=vMax, vmin=-vMax)
    if cfg.fit.gauss == 2:
        ax1[1].set_title("Double gaussian fit")
    elif cfg.fit.gauss == 3:
        ax1[1].set_title("Triple gaussian fit")

    cb = plt.colorbar(cm)
    cb.set_label("Difference counts")
    plt.tight_layout()
    plt.savefig(cfg.out.img+cfg.out.tag+"_diffMap."+cfg.out.plotFormat, dpi=cfg.out.dpi)

#########################################
### Amplitude of the fitted gaussians ###
#########################################

if cfg.plot.fitAmp:
    print("Plot amplitude...")

    f21, ax21 = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

    ax21[0].errorbar(mOpt.columns.values, mOpt.loc["A1"], yerr=mCov.loc["A1"], fmt=".", markersize=2, capsize=2, elinewidth=1, color=color1, label="Gauß 1")
    ax21[0].errorbar(mOpt.columns.values, mOpt.loc["A2"], yerr=mCov.loc["A2"], fmt=".", markersize=2, capsize=2, elinewidth=1, color=color2, label="Gauß 2")
    if cfg.fit.gauss == 3:
        ax21[0].errorbar(mOpt.columns.values, mOpt.loc["A3"], yerr=mCov.loc["A3"], fmt="^", markersize=2, capsize=2, elinewidth=1, color=color3, label="Gauß 3")

    ax21[0].set_xlim(cfg.ROI.delay[0], cfg.ROI.delay[1])
    ax21[0].set_ylim(-1, 10)
    ax21[0].set_ylabel("Amplitude (counts)")
    ax21[0].legend()

    rAmp = mOpt.loc["A1"] / mOpt.loc["A2"]
    urAmp = rAmp * (abs(mCov.loc["A1"] / mOpt.loc["A1"]) + abs(mCov.loc["A2"] / mOpt.loc["A2"]))

    ax21[1].errorbar(mOpt.columns.values, rAmp, yerr=urAmp, fmt=".", markersize=2, capsize=2, elinewidth=1)
    ax21[1].set_ylim(-1, 2)
    ax21[1].set_xlabel("Delay (ps)")
    ax21[1].set_ylabel("Amplitude ratio")

    plt.tight_layout()
    plt.savefig(cfg.out.img+cfg.out.tag+"_fitAmp."+cfg.out.plotFormat, dpi=cfg.out.dpi)

################################
### Peak position comparison ###
################################
if cfg.plot.fitPeak:
    print("Plot peak position...")

    f22, ax22 = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

    ax22[0].errorbar(mOpt.columns.values, mOpt.loc["x1"], yerr=mCov.loc["x1"], fmt=".", markersize=2, capsize=2, elinewidth=1, color=color1, label="Gauß 1")
    ax22[0].errorbar(mOpt.columns.values, mOpt.loc["x2"], yerr=mCov.loc["x2"], fmt=".", markersize=2, capsize=2, elinewidth=1, color=color2, label="Gauß 2")
    if cfg.fit.gauss == 3:
        ax22[0].errorbar(mOpt.columns.values, mOpt.loc["x3"], yerr=mCov.loc["x3"], fmt="^", markersize=2, capsize=2, elinewidth=1, color=color3, label="Gauß 3")

    ax22[0].set_ylabel("Peak position (eV)")
    ax22[0].set_ylim(cfg.ROI.ekin[0]*1.05, cfg.ROI.ekin[1]*0.96)
    ax22[0].legend()

    posDiff = mOpt.loc["x1"] - mOpt.loc["x2"]
    upDiff = abs(mCov.loc["x1"]) + abs(mCov.loc["x2"])

    ax22[1].errorbar(mOpt.columns.values, posDiff, yerr=upDiff, fmt=".", markersize=2, capsize=2, elinewidth=1, label="G1 - G2")
    if cfg.fit.gauss == 3:
        posDiff2 = mOpt.loc["x3"] - mOpt.loc["x2"]
        upDiff2 = abs(mCov.loc["x3"]) + abs(mCov.loc["x2"])
        ax22[1].errorbar(mOpt.columns.values, posDiff2, yerr=upDiff2, fmt=".", markersize=2, capsize=2, elinewidth=1, label="G3 - G2")
        ax22[1].legend()

    ax22[1].set_xlabel("Delay (ps)")
    ax22[1].set_ylabel("Difference position (eV)")
    ax22[1].set_ylim(-4, 1)
    ax22[0].set_xlim(cfg.ROI.delay[0], cfg.ROI.delay[1])

    plt.tight_layout()
    plt.savefig(cfg.out.img+cfg.out.tag+"_fitPeak."+cfg.out.plotFormat, dpi=cfg.out.dpi)

    if cfg.plot.PLCorr:
        print("Correct peak position by change in negative feature...")
        # correct for photoline wobbling
        # assumes that the depletion feature should stay at a constant energy
        negMean = mOpt.loc["x2"].mean()
        negStd  = mOpt.loc["x2"].std()
        diff = mOpt.loc["x2"] - negMean
        diffErr = mCov.loc["x2"] + negStd

        print(f"Mean position of depletion: {negMean:.2f} +- {negStd:.2f} eV")
        print(f"Average absolute displacement: {abs(diff).mean():.2f} +- {abs(diff).std():.2f} eV")

        shiftedPos = mOpt.loc["x1"] - diff
        shiftedPosErr = mCov.loc["x1"] + diffErr

        f2x, ax2x = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

        ax2x[0].errorbar(diff.index.values, diff.values, yerr=diffErr.values,
                        fmt=".", markersize=2, capsize=2, elinewidth=1)
        ax2x[1].plot(shiftedPos.index.values, shiftedPos.values, ".")#, yerr=shiftedPosErr.values,
        #                fmt=".", markersize=2, capsize=2, elinewidth=1)
        ax2x[1].hlines(negMean, xmin=diff.index.values[0], xmax=diff.index.values[-1], ls="--")
        ax2x[0].set_ylabel("Depletion wobbling (eV)")
        ax2x[1].set_ylabel("Corrected peak position (eV)")
        ax2x[1].set_xlabel("Delay (ps)")
        ax2x[1].set_ylim(95, 105)
        plt.tight_layout()
        plt.savefig(cfg.out.img+cfg.out.tag+"_peakCorr."+cfg.out.plotFormat, dpi=cfg.out.dpi)

#############################
### Peak width comparison ###
#############################
if cfg.plot.fitWidth:
    print("Plot peak width...")

    f23, ax23 = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

    ax23[0].errorbar(mOpt.columns.values, abs(mOpt.loc["w1"]), yerr=mCov.loc["w1"], fmt=".", markersize=2, capsize=2, elinewidth=1, color=color1, label="Gauß 1")
    ax23[0].errorbar(mOpt.columns.values, abs(mOpt.loc["w2"]), yerr=mCov.loc["w2"], fmt=".", markersize=2, capsize=2, elinewidth=1, color=color2, label="Gauß 2")
    if cfg.fit.gauss == 3:
        ax23[0].errorbar(mOpt.columns.values, abs(mOpt.loc["w3"]), yerr=mCov.loc["w3"], fmt="^", markersize=2, capsize=2, elinewidth=1, color=color3, label="Gauß 3")
        ax23[0].set_ylim(0, 5)
    else:
        ax23[0].set_ylim(0, 4)
    ax23[0].set_ylabel("Peak width (eV)")

    wDiff = mOpt.loc["w1"] - mOpt.loc["w2"]
    uwDiff = abs(mCov.loc["w1"]) + abs(mCov.loc["w2"])

    ax23[1].errorbar(mOpt.columns.values, wDiff, yerr=uwDiff, fmt=".", markersize=2, capsize=2, elinewidth=1, label="G1 - G2")
    if cfg.fit.gauss == 3:
        wDiff2 = mOpt.loc["w3"] - mOpt.loc["w2"]
        uwDiff2 = abs(mCov.loc["w3"]) + abs(mCov.loc["w2"])
        ax22[1].errorbar(mOpt.columns.values, posDiff2, yerr=upDiff2, fmt=".", markersize=2, capsize=2, elinewidth=1, label="G3 - G2")
        ax22[1].legend()

    ax23[1].set_ylim(-1, 2.5)
    ax23[1].set_xlabel("Delay (ps)")
    ax23[1].set_ylabel("Difference width (eV)")
    ax23[0].set_xlim(cfg.ROI.delay[0], cfg.ROI.delay[1])
    plt.tight_layout()
    plt.savefig(cfg.out.img+cfg.out.tag+"_fitWidth."+cfg.out.plotFormat, dpi=cfg.out.dpi)

##########################
### Find zero-crossing ###
##########################

print("Calculate zero-crossing...")
if cfg.plot.zeroCr:
    dZero = []

    # estimating from minimum abs(data)
    for i in range(len(data.index.values)):
        ld = absMinZero(data.loc[data.index.values[i]], estMin=None, func="linear", n=13)
        dZero.append(ld)

    dZero = pd.DataFrame(data=np.array(dZero).T, columns=data.index.values, index=["x0", "ux0"])
    dZero.to_csv(cfg.out.fit+cfg.out.tag+"_zcData.csv", sep=",")

    f3, ax3 = plt.subplots()
    ax3.plot(dZero.columns.values, dZero.loc["x0"], ".", markersize=4, color="tab:blue", label="data")

    if cfg.fit.gauss == 2 and cfg.fit.analyticZero:
        # estimating analyitcally from double gaussian fit
        fZero = []
        for i in range(len(fit.index.values)):
            lf = analyticZero(mOpt.values.T[i], (cfg.fit.guess[1], cfg.fit.guess[4]), error=mCov.values.T[i])
            fZero.append(lf)

        fZero = pd.DataFrame(data=np.array(fZero).T, columns=fit.index.values, index=["x0", "ux0"])
        fZero.to_csv(cfg.out.fit+cfg.out.tag+"_zcFit.csv", sep=",")
        ax3.plot(fZero.columns.values, fZero.loc["x0"], ".", markersize=4, color="tab:orange", label="fit")
        ax3.legend()

    ax3.set_xlim(cfg.ROI.delay[0], cfg.ROI.delay[1])
    zeroMean = dZero.loc["x0"].mean()
    ax3.set_ylim(100, 102.5)
    ax3.set_xlabel("Delay (ps)")
    ax3.set_ylabel("Zero position (eV)")
    plt.tight_layout()
    plt.savefig(cfg.out.img + cfg.out.tag + "_zeroPos."+cfg.out.plotFormat, dpi=cfg.out.dpi)

    if cfg.plot.PLCorr:
        print("Correct zero position with change in negative feature...")

        f3x = plt.figure()
        shiftedZeroD = dZero.loc["x0"] - diff
        plt.plot(shiftedZeroD.index.values, shiftedZeroD.values,".", label="Data")

        if cfg.fit.gauss == 2 and cfg.fit.analyticZero:
            shiftedZeroF = fZero.loc["x0"] - diff
            plt.plot(shiftedZeroF.index.values, shiftedZeroF.values,".", label="Data")

        plt.ylabel("Corrected zero position (eV)")
        plt.ylim(zeroMean*0.98, zeroMean*1.02)
        plt.xlabel("Delay (ps)")
        plt.tight_layout()
        plt.savefig(cfg.out.img+cfg.out.tag+"_zeroCorr."+cfg.out.plotFormat, dpi=cfg.out.dpi)

    ######################
    ### Calculate area ###
    ######################
    if cfg.plot.area:
        print("Calculate integrated signal...")

        data_area = np.array([calc_area(data.loc[i], dZero.loc["x0"].loc[i]) for i in dZero.columns]).T
        np.savetxt(cfg.out.fit+cfg.out.tag+"_areaData.csv",data_area, delimiter=",")

        if cfg.fit.gauss == 2 and cfg.fit.analyticZero:
            fit_area = np.array([calc_area(fit.loc[i], fZero.loc["x0"].loc[i]) for i in fZero.columns]).T
            np.savetxt(cfg.out.fit+cfg.out.tag+"_areaFit.csv",fit_area, delimiter=",")

        f4, ax4 = plt.subplots(1, 2, figsize=(8, 4))

        ax4[0].plot(dZero.columns.values, abs(data_area[0]), color=color1, label="Data +")
        ax4[0].plot(dZero.columns.values, abs(data_area[1]), color=color2, label="Data -")
        ax4[1].plot(dZero.columns.values, abs(data_area[0]) - abs(data_area[1]), label="Data (pos - neg)")

        if cfg.fit.gauss == 2 and cfg.fit.analyticZero:
            ax4[0].plot(fZero.columns.values, abs(fit_area[0]), "--", color=color1, label="Fit +")
            ax4[0].plot(fZero.columns.values, abs(fit_area[1]), "--", color=color2, label="Fit -")
            ax4[1].plot(fZero.columns.values, abs(fit_area[0]) - abs(fit_area[1]), label="Fit (pos - neg)")

        ax4[0].set_xlim(cfg.ROI.delay[0], cfg.ROI.delay[1])
        ax4[0].set_xlabel("Delay (ps)")
        ax4[0].set_ylabel(r"Area (counts $\cdot$ eV)")
        ax4[0].legend()

        ax4[1].set_xlim(cfg.ROI.delay[0], cfg.ROI.delay[1])
        ax4[1].set_xlabel("Delay (ps)")
        ax4[1].set_ylabel(r"Area difference (counts $\cdot$ eV)")
        ax4[1].legend()
        plt.tight_layout()
        plt.savefig(cfg.out.img+cfg.out.tag+"_area." + cfg.out.plotFormat, dpi=cfg.out.dpi)

print("Done.")

if cfg.plot.show:
    plt.show()
else:
    plt.close("all")
