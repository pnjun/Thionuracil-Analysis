#!/usr/bin/python3
"""
Script for analysing sulfur 2p photoline

script uses csv files created with delayScan.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from attrdict import AttrDict

cfg = { "data" : { "folder" : "./data/",
                   "file"   : "wp-10.2.csv", # only processed data from delayScan.py
                   },
        "out"  : { "folder"     : "./Plots/",
                   "tag"        : "wp_10.2_2.0ps",
                   "showPlots"  : False,
                   "plotFormat" : "png",
                   "dpi"        : 300
                   },
        "ROI"  : { "ekin"  : (90, 110),  # kinetic energy at which the photoline is located
                   "delay" : (-0.3, 2.1)  # delays of interest
                   },
        "t0"   : 1180.,  # time zero estimate
        "energy" : "kinetic",  # choose between kinetic or binding
        "Eph"  : 272.,  # photon energy of soft x-rays
        "fitGuess" : (3., 101, 5., 7., 103., 3., 0.01),  # guess for gaussian fit
        "PLCorr" : True  # Correcting photoline shift by shift in negative feature
        }
cfg = AttrDict(cfg)

data = pd.read_csv(cfg.data.folder + cfg.data.file, sep=",", index_col=0)
#data = data[(data.values != 0).all(axis=1)]  # filter rows with zero-only entries (no data rows)
data.index = (data.index - cfg.t0) * -1  # shift delays
delay = data.index

dMask = (data.index.values > cfg.ROI.delay[0]) & (data.index.values < cfg.ROI.delay[1])
data = data[dMask]
eMask = (data == data) & (data.columns.astype(float) >= cfg.ROI.ekin[0]) & (data.columns.astype(float) <= cfg.ROI.ekin[1])
data = data[eMask].T.dropna().T

if cfg.energy == "binding":
    data.columns = cfg.Eph - data.columns.astype(float)
elif cfg.energy != "kinetic":
    print("Parameter 'energy' in cfg should be either 'kinetic' or 'binding'!")
    exit()


def gaussian(x, Amp, zero, width, const):
    return Amp * np.exp(-0.5 * np.square((x - zero) / width)) + const


def modellDiff(x, A1, x1, b1, A2, x2, b2, const):
    return gaussian(x, A1, x1, b1, const=0) - gaussian(x, A2, x2, b2, const=0) + const


def decay(x, Amp, zero, tc, const):
    return Amp * np.exp(-(x - zero) / tc) + const


def modellTime(x, a0, t0, tau0, a1, tau1, a2, tau2, const=0):
    sig = np.zeros(len(x)-1)

    for a in range(len(x[:-1])):
        j = 0
        for b in range(len(x[:a+1])):
            f1 = gaussian(x[b], a0, t0, tau0, const=0)
            f2 = decay(x[a], a1, x[b], tau1, const=0)
            f3 = decay(x[a], a2, x[b], tau2, const=0)
            j += f1 * (f2 + f3) * abs(x[b] - x[b+1])
        sig[a] += j
    return sig + const


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


### Fit difference spectra to double gaussian function ###
mOpt = []
mCov = []
fit = []
if cfg.energy == "binding":
    eRange = cfg.Eph - np.array([cfg.ROI.ekin[1], cfg.ROI.ekin[0]])
    energy = np.arange(eRange[0], eRange[1], 0.1)
else:
    energy = np.arange(cfg.ROI.ekin[0], cfg.ROI.ekin[1], 0.1)

print("Start with fitting double gaussian modell...")
for i in range(len(data.values)):

    pGuess = cfg.fitGuess
    try:
        pOpt, pCov = curve_fit(modellDiff, data.columns.values, data.values[i], p0=pGuess)
        pCov = np.sqrt(np.diag(pCov))
        mOpt.append(pOpt)
        mCov.append(pCov)
        fit.append(modellDiff(energy, *pOpt))
    except RuntimeError:
        print(f"RuntimeError at delay {data.index.values[i]} ps")
        mOpt.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
        mCov.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
        fit.append([np.NaN for i in range(len(energy))])


mOpt = pd.DataFrame(data=np.array(mOpt).T, index=["A1", "x1", "w1", "A2", "x2", "w2", "const"], columns=data.index.values).T.dropna().T
mCov = pd.DataFrame(data=np.array(mCov).T, index=["A1", "x1", "w1", "A2", "x2", "w2", "const"], columns=data.index.values).T.dropna().T
fit = pd.DataFrame(data=np.array(fit), index=data.index.values, columns=energy).dropna()

mOpt.to_csv(cfg.data.folder+"Fits/"+cfg.out.tag+"_diffMapOptParam.csv", sep=",")
mCov.to_csv(cfg.data.folder+"Fits/"+cfg.out.tag+"_diffMapOptError.csv", sep=",")

print("Offsets derived from fit:")
print(mOpt.loc["const"])

#mCov = mCov.T.query("A1 < 5 & A2 < 2").T
#mOpt = mOpt.T.drop(mOpt.T.index.difference(mCov.T.index)).T
#fit = fit.drop(fit.index.difference(mCov.T.index))

#################################
### Difference map comparison ###
#################################
print("Plot difference map...")

f1, ax1 = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

vMax = np.max(abs(data.values)) * 0.75
cm = ax1[0].pcolormesh(data.columns.values.astype(float), data.index.values, data.values, cmap="bwr", vmax=vMax, vmin=-vMax)
ax1[0].set_ylim(cfg.ROI.delay[0], cfg.ROI.delay[1])

ax1[0].set_ylabel("Delay (ps)")
ax1[0].set_title("Data")

ax1[1].pcolormesh(fit.columns.values, fit.index.values, fit.values, cmap="bwr", vmax=vMax, vmin=-vMax)

if cfg.energy == "kinetic":
    ax1[0].set_xlabel("Kinetic Energy (eV)")
    ax1[1].set_xlabel("Kinetic Energy (eV)")
else:
    ax1[0].set_xlabel("Binding Energy (eV)")
    ax1[1].set_xlabel("Binding Energy (eV)")


ax1[1].set_title("Fit")

cb = plt.colorbar(cm)
cb.set_label("Difference counts")
plt.tight_layout()
plt.savefig(cfg.out.folder+cfg.out.tag+"_diffMap."+cfg.out.plotFormat, dpi=cfg.out.dpi)

#########################################
### Amplitude of the fitted gaussians ###
#########################################
print("Plot amplitude...")

color1 = "tab:red"
color2 = "tab:blue"

f21, ax21 = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

ax21[0].errorbar(mOpt.columns.values, mOpt.loc["A1"], yerr=mCov.loc["A1"], fmt=".", markersize=2, capsize=2, elinewidth=1, color=color1, label="Gauß 1")
ax21[0].errorbar(mOpt.columns.values, mOpt.loc["A2"], yerr=mCov.loc["A2"], fmt=".", markersize=2, capsize=2, elinewidth=1, color=color2, label="Gauß 2")
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
plt.savefig(cfg.out.folder+cfg.out.tag+"_fitAmp."+cfg.out.plotFormat, dpi=cfg.out.dpi)

################################
### Peak position comparison ###
################################
print("Plot peak position...")

f22, ax22 = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

ax22[0].errorbar(mOpt.columns.values, mOpt.loc["x1"], yerr=mCov.loc["x1"], fmt=".", markersize=2, capsize=2, elinewidth=1, color=color1, label="Gauß 1")
ax22[0].errorbar(mOpt.columns.values, mOpt.loc["x2"], yerr=mCov.loc["x2"], fmt=".", markersize=2, capsize=2, elinewidth=1, color=color2, label="Gauß 2")
ax22[0].set_ylabel("Peak position (eV)")
if cfg.energy == "kinetic":
    ax22[0].set_ylim(cfg.ROI.ekin[0]*1.05, cfg.ROI.ekin[1]*0.96)
else:
    ax22[0].set_ylim(eRange[0]*1.025, eRange[1]*0.97)
ax22[0].legend()

posDiff = mOpt.loc["x1"] - mOpt.loc["x2"]
upDiff = abs(mCov.loc["x1"]) + abs(mCov.loc["x2"])

ax22[1].errorbar(mOpt.columns.values, posDiff, yerr=upDiff, fmt=".", markersize=2, capsize=2, elinewidth=1)
ax22[1].set_xlabel("Delay (ps)")
ax22[1].set_ylabel("Difference position (eV)")
ax22[1].set_ylim(-5, 5)
ax22[0].set_xlim(cfg.ROI.delay[0], cfg.ROI.delay[1])

plt.tight_layout()
plt.savefig(cfg.out.folder+cfg.out.tag+"_fitPeak."+cfg.out.plotFormat, dpi=cfg.out.dpi)

if cfg.PLCorr:
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
    ax2x[1].errorbar(shiftedPos.index.values, shiftedPos.values, yerr=shiftedPosErr.values,
                    fmt=".", markersize=2, capsize=2, elinewidth=1)
    ax2x[1].hlines(negMean, xmin=diff.index.values[0], xmax=diff.index.values[-1])
    ax2x[0].set_ylabel("Depletion wobbling (eV)")
    ax2x[1].set_ylabel("Corrected peak position (eV)")
    ax2x[1].set_xlabel("Delay (ps)")
    plt.tight_layout()
    plt.savefig(cfg.out.folder+cfg.out.tag+"_peakCorr."+cfg.out.plotFormat, dpi=cfg.out.dpi)

#############################
### Peak width comparison ###
#############################
print("Plot peak width...")

f23, ax23 = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

ax23[0].errorbar(mOpt.columns.values, abs(mOpt.loc["w1"]), yerr=mCov.loc["w1"], fmt=".", markersize=2, capsize=2, elinewidth=1, color=color1, label="Gauß 1")
ax23[0].errorbar(mOpt.columns.values, abs(mOpt.loc["w2"]), yerr=mCov.loc["w2"], fmt=".", markersize=2, capsize=2, elinewidth=1, color=color2, label="Gauß 2")
ax23[0].set_ylim(0, 4)
ax23[0].set_ylabel("Peak width (eV)")

wDiff = mOpt.loc["w1"] - mOpt.loc["w2"]
uwDiff = abs(mCov.loc["w1"]) + abs(mCov.loc["w2"])

ax23[1].errorbar(mOpt.columns.values, wDiff, yerr=uwDiff, fmt=".", markersize=2, capsize=2, elinewidth=1, label="Difference")
ax23[1].set_ylim(-1, 2.5)
ax23[1].set_xlabel("Delay (ps)")
ax23[1].set_ylabel("Difference width (eV)")
ax23[0].set_xlim(cfg.ROI.delay[0], cfg.ROI.delay[1])
plt.tight_layout()
plt.savefig(cfg.out.folder+cfg.out.tag+"_fitWidth."+cfg.out.plotFormat, dpi=cfg.out.dpi)

##########################
### Find zero-crossing ###
##########################
print("Calculate zero-crossing...")

dZero = []
fZero = []

# estimating from minimum abs(data)
for i in range(len(data.index.values)):
    ld = absMinZero(data.loc[data.index.values[i]], estMin=270, func="linear", n=13)
    dZero.append(ld)

# estimating analyitcally from double gaussian fit
for i in range(len(fit.index.values)):
    lf = analyticZero(mOpt.values.T[i], (cfg.fitGuess[1], cfg.fitGuess[4]), error=mCov.values.T[i])
    fZero.append(lf)


dZero = pd.DataFrame(data=np.array(dZero).T, columns=data.index.values, index=["x0", "ux0"])
fZero = pd.DataFrame(data=np.array(fZero).T, columns=fit.index.values, index=["x0", "ux0"])

dZero.to_csv(cfg.data.folder+"Fits/"+cfg.out.tag+"_zcData.csv", sep=",")
fZero.to_csv(cfg.data.folder+"Fits/"+cfg.out.tag+"_zcFit.csv", sep=",")

f3, ax3 = plt.subplots()
ax3.plot(dZero.columns.values, dZero.loc["x0"], ".", markersize=4, color="tab:blue", label="data")
ax3.plot(fZero.columns.values, fZero.loc["x0"], ".", markersize=4, color="tab:orange", label="fit")
ax3.set_xlim(cfg.ROI.delay[0], cfg.ROI.delay[1])
zeroMean = dZero.loc["x0"].mean()
ax3.set_ylim(zeroMean*0.98, zeroMean*1.02)
ax3.set_xlabel("Delay (ps)")
ax3.set_ylabel("Zero position (eV)")
ax3.legend()
plt.tight_layout()
plt.savefig(cfg.out.folder + cfg.out.tag + "_zeroPos."+cfg.out.plotFormat, dpi=cfg.out.dpi)

######################
### Calculate area ###
######################
print("Calculate integrated signal...")

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


fit_area = np.array([calc_area(fit.loc[i], fZero.loc["x0"].loc[i]) for i in fZero.columns]).T
data_area = np.array([calc_area(data.loc[i], dZero.loc["x0"].loc[i]) for i in dZero.columns]).T

np.savetxt(cfg.data.folder+"Fits/"+cfg.out.tag+"_areaData.csv",data_area, delimiter=",")
np.savetxt(cfg.data.folder+"Fits/"+cfg.out.tag+"_areaFit.csv",fit_area, delimiter=",")

f4, ax4 = plt.subplots(1, 2, figsize=(8, 4))

ax4[0].plot(dZero.columns.values, abs(data_area[0]), color="tab:red", label="Data +")
ax4[0].plot(dZero.columns.values, abs(data_area[1]), color="tab:blue", label="Data -")
ax4[0].plot(fZero.columns.values, abs(fit_area[0]), "--", color="tab:red", label="Fit +")
ax4[0].plot(fZero.columns.values, abs(fit_area[1]), "--", color="tab:blue", label="Fit -")
ax4[0].set_xlim(cfg.ROI.delay[0], cfg.ROI.delay[1])
ax4[0].set_xlabel("Delay (ps)")
ax4[0].set_ylabel(r"Area (counts $\cdot$ eV)")
ax4[0].legend()

ax4[1].plot(dZero.columns.values, abs(data_area[0]) - abs(data_area[1]), label="Data (pos - neg)")
ax4[1].plot(fZero.columns.values, abs(fit_area[0]) - abs(fit_area[1]), label="Fit (pos - neg)")
ax4[1].set_xlim(cfg.ROI.delay[0], cfg.ROI.delay[1])
ax4[1].set_xlabel("Delay (ps)")
ax4[1].set_ylabel(r"Area difference (counts $\cdot$ eV)")
ax4[1].legend()
plt.tight_layout()
plt.savefig(cfg.out.folder+cfg.out.tag+"_area." + cfg.out.plotFormat, dpi=cfg.out.dpi)

print("Done.")

if cfg.out.showPlots:
    plt.show()
else:
    plt.close("all")
