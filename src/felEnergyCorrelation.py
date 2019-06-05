#!/usr/bin/python3.6
'''
Calculates correlation coefficients between FEL photon energy and pulse energy.
To make the script faster, gmd data needed for the calculation should be saved in a separat file beforehand.
'''
from attrdict import AttrDict
import pandas as pd
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

cfg = { 'data'   : { 'path1'  : '/media/Data/Beamtime/processed/',
                     'path2'  : './',
                     'index'  : 'index.h5',
                     'gmd'    : 'GMDdata.h5'},
        'hdf'    : { 'pulses' : '/pulses',
                     'time'   : 'time',
                     'undu'   : 'undulatorEV',
                     'opis'   : 'opisEV',
                     'gmd'    : 'GMD'},
        'time'   : { 'start'  : '25.03.19 13:29:00',
                     'stop'   : '25.03.19 15:35:00'},
        'output' : { 'folder' : './Graphs/'}}

cfg = AttrDict(cfg)

def getIndices(data, lowerLim, upperLim):

    indices = []
    for i in range(len(data)):
        value = data.iloc[i]
        index = data.index.values[i]
        if value >= lowerLim and value <= upperLim and index not in indices:
            indices.append(index)
        if value > upperLim: break

    return indices

index = pd.read_hdf(cfg.data.path1 + cfg.data.index, cfg.hdf.pulses).sort_index()
gmd = pd.read_hdf(cfg.data.path2 + cfg.data.gmd, cfg.hdf.gmd).sort_index()

start = int(time.mktime(datetime.datetime.strptime(cfg.time.start, '%d.%m.%y %H:%M:%S').timetuple()))
stop  = int(time.mktime(datetime.datetime.strptime(cfg.time.stop, '%d.%m.%y %H:%M:%S').timetuple()))

times   = index[cfg.hdf.time]
indices = np.array(getIndices(times, start, stop))
opis    = index[cfg.hdf.opis].loc[indices]
undu    = index[cfg.hdf.undu].loc[indices]
gmd     = gmd.loc[indices]

# correlation with macrobunch mean
mbEn = np.array([[gmd.loc[i].mean(), gmd.loc[i].std(), gmd.loc[i].sum()] for i in indices]).transpose()
df = pd.DataFrame({'undulatorEV': undu,
                   'opis'       : opis,
                   'mbMean'     : mbEn[0],
                   'mbStd'      : mbEn[1],
                   'mbSum'      : mbEn[2]
                  })

fig = plt.figure()
plt.plot(df['undulatorEV'].values, '.',markersize=2, label='undulator setpoints')
plt.plot(df['opis'].values, '.',markersize=2, label='OPIS values')
plt.title('FEL photon energy\n{0} -- {1}'.format(cfg.time.start, cfg.time.stop))
plt.ylabel('Photon energy (eV)')
plt.xlabel('Shot number')
plt.legend()
plt.tight_layout()
plt.savefig(cfg.output.folder+'FEL_pulse_vs_photon_energy/photon_energy.png')
plt.close(fig)

fig = plt.figure()
plt.plot(df['mbMean'].values, '.',markersize=2)
plt.ylabel(u'Pulse energy (µJ)')
plt.xlabel('shot number')
plt.title('FEL pulse energy\n{0} -- {1}'.format(cfg.time.start, cfg.time.stop))
plt.tight_layout()
plt.savefig(cfg.output.folder+'FEL_pulse_vs_photon_energy/pulse_energy.png')
plt.close(fig)

fig = plt.figure()
plt.plot(df['undulatorEV'].values, df['mbMean'].values,'.',markersize=2, label='undulator setpoints')
plt.plot(df['opis'].values, df['mbMean'].values,'.',markersize=2, label='OPIS values')
plt.xlabel('FEL photon energy (eV)')
plt.ylabel('Pulse energy (µJ)')
plt.title('FEL photon vs. pulse energy\n{0} -- {1}'.format(cfg.time.start, cfg.time.stop))
plt.legend()
plt.tight_layout()
plt.savefig(cfg.output.folder+'FEL_pulse_vs_photon_energy/pulse_vs_photon_energy.png')
plt.close(fig)

'''
mask = df['mbMean'] < 15
df = df.where(mask)
df_bm = df.groupby(pd.cut(df['undulatorEV'], 20))
df_bmStd = df_bm.std()
df_bm = df_bm.mean()
corr1 = df_bm['mbMean'].corr(df_bm['undulatorEV'], method='pearson')
corr2 = df_bm['mbMean'].corr(df_bm['undulatorEV'], method='spearman')
fig = plt.figure()
plt.plot(df_bm['undulatorEV'],df_bm['mbMean'], '.')
plt.xlabel(u'FEL photon energy (undulator set) (eV)')
plt.ylabel(u'FEL pulse energy (µJ)')
plt.title('FEL photon vs pulse energy (MB mean)\n{0} -- {1}\nPearson: {2}, Spearman: {3}'.format(cfg.time.start, cfg.time.stop, round(corr1, 3), round(corr2, 3)))
plt.tight_layout()
plt.savefig(cfg.output.folder+'FEL_pulse_vs_photon_energy/mbMean.png')
plt.close(fig)

# pulse wise correlation
corr = []
for i in range(50):

    #fig, ax = plt.subplots(1,1)
    #gmd.hist(column=i, bins=30, ax=ax)
    #ax.set_xlabel('pulse energy (µJ)')
    #ax.set_ylabel('counts')
    #plt.tight_layout()
    #plt.savefig('./Graphs/hist_pulseEn_pulse{}.png'.format(i+1))
    #plt.close(fig)

    h = 15
    mask = gmd[i] < h
    data = gmd[i].where(mask)
    data = pd.DataFrame({'undulatorEV' : undu.values,
                        'pulseEn'      : data.values})
    dataB = data.groupby(pd.cut(data['undulatorEV'], 20))
    dataBStd = dataB.std()
    dataB = dataB.mean()
    corr1 = dataB['pulseEn'].corr(dataB['undulatorEV'], method='pearson')
    corr2 = dataB['pulseEn'].corr(dataB['undulatorEV'], method='spearman')
    corr.append([corr1, corr2])
    fig, ax = plt.subplots(1,1)
    ax.plot(dataB['undulatorEV'], dataB['pulseEn'], '.')
    ax.set_xlabel(u'FEL photon energy (undulator set) (eV)')
    ax.set_ylabel(u'FEL pulse energy (µJ)')
    ax.set_title('FEL photon vs pulse energy (Pulse {0})\n{1} -- {2}\nPearson: {3}, Spearman: {4}'.format(i, cfg.time.start, cfg.time.stop, round(corr1, 3), round(corr2, 3)))
    plt.tight_layout()
    plt.savefig(cfg.output.folder+'FEL_pulse_vs_photon_energy/pulse_{0}.png'.format(i))
    plt.close(fig)

corr = np.array(corr).transpose()

# plot correlation coefficients
fig = plt.figure()
plt.plot(corr[0], '.',color='tab:blue', label='Pearson')
plt.plot(corr[1], '^',color='tab:orange', label='Spearman')
plt.hlines(corr1, 0, 50, color='tab:blue', ls='dashed', label='MB mean, Pearson')
plt.hlines(corr2, 0, 50, color='tab:orange', ls='dashed', label='MB mean, Spearman')
plt.legend()
plt.title('FEL photon vs pulse energy (correlation coefficients)\n{0} -- {1}'.format(cfg.time.start, cfg.time.stop))
plt.xlabel('Pulse number')
plt.ylabel('Correlation coefficient')
plt.tight_layout()
plt.savefig(cfg.output.folder+'FEL_pulse_vs_photon_energy/corrCoeffs.png')
plt.close(fig)
'''
