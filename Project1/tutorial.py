#----------------------------------------------------------------
# Load libraries
#----------------------------------------------------------------

import numpy as np
import math
from gwpy.timeseries import TimeSeries

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py

import os

def savefig(plt, name):
    plt.savefig(name)
    print('Figure ' + os.path.abspath(name) + ' is created.')

def printstep(step):
    print('\033[32;1m\n------- '+ step + ' >>>\033[0m')

#----------------------------------------------------------------
# Set parameters
#----------------------------------------------------------------

printstep('Set parameters')
fn = 'data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5' # data file
tevent = 1126259462.422
evtname = 'GW150914' # event name

detector = '' # detecotr: L1 or H1
if 'L1' in fn : detector = 'L1'
elif 'H1' in fn : detector = 'H1'
else : exit()
frequency = '4KHZ'

dirpath = './plots/'+evtname+'_'+frequency+'_'+detector;
os.makedirs(dirpath, exist_ok=True)
print(dirpath)

#----------------------------------------------------------------
# Load LIGO data
#----------------------------------------------------------------

printstep('Load data')
strain = TimeSeries.read(fn, format='hdf5.losc') # gwpy.timeseries.TimeSeries https://gwpy.github.io/docs/latest/api/gwpy.timeseries.TimeSeries.html
center = int(tevent)
strain = strain.crop(center-16, center+16)

#----------------------------------------------------------------
# Show LIGO strain vs. time
#----------------------------------------------------------------

printstep('Draw strain')
plt.figure()
strain.plot()
plt.ylabel('strain')
savefig(plt, dirpath + '/strain_raw.png')

#----------------------------------------------------------------
# Obtain the power spectrum density PSD / ASD
#----------------------------------------------------------------

printstep('Draw ASD')
asd = strain.asd(fftlength=8)

plt.figure()
asd.plot()
plt.xlim(10, 2000)
plt.ylim(1e-24, 1e-19)
plt.ylabel('ASD (strain/Hz$^{1/2})$')
plt.xlabel('Frequency (Hz)')
savefig(plt, dirpath + '/ASDs.png')

#----------------------------------------------------------------
# Whitening data
#----------------------------------------------------------------

printstep('Whitening')
white_data = strain.whiten()

plt.figure()
white_data.plot()
plt.ylabel('strain (whitened)')
savefig(plt, dirpath + '/strain_whiten.png')

#----------------------------------------------------------------
# Bandpass filtering
#----------------------------------------------------------------

printstep('Band-pass filter')
bandpass_low = 30
bandpass_high = 400
white_data_bp = white_data.bandpass(bandpass_low, bandpass_high)

plt.figure()
white_data_bp.plot()
plt.ylabel('strain (whitened + band-pass)')
savefig(plt, dirpath + '/strain_whiten_bandpass.png')

#----------------------------------------------------------------
# q-transform
#----------------------------------------------------------------

printstep('q-transform')
dt = 1  #-- Set width of q-transform plot, in seconds
hq = strain.q_transform(outseg=(tevent-dt, tevent+dt))

plt.figure()
fig = hq.plot()
ax = fig.gca()
fig.colorbar(label="Normalised energy")
ax.grid(False)
ax.set_yscale('log')
plt.ylabel('Frequency (Hz)')
savefig(plt, dirpath + '/qtrans.png')

#----------------------------------------------------------------
# Frequency analytic
#----------------------------------------------------------------

printstep('Build analytic model for frequency')
def gwfreq(iM,iT,iT0):
    const = (948.5)*np.power((1./iM),5./8.)
    output = const*np.power(np.maximum((iT0-iT),3e-2),-3./8.) # we can max it out above 500 Hz-ish
    return output

times = np.linspace(0., 4., 50)
freq = gwfreq(20, times, 4)

plt.figure()
plt.plot(times, freq)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
savefig(plt, dirpath + '/qtrans_analytic.png')

#----------------------------------------------------------------
# Wave form analytic
#----------------------------------------------------------------

printstep('Build analytic model for wave')
def osc(t,Mc,t0,C,phi):
    freq = gwfreq(Mc,t,t0)
    val = C*(np.cos(freq*(t0-t)+phi))*1e-12
    val = val*np.power(Mc*freq,10./3.)*(1*(t<=t0)+np.exp((freq/(2*np.pi))*(t0-t))*(t>t0))
    return val

def osc_dif(params, x, data, eps):
    iM=params["Mc"]
    iT0=params["t0"]
    norm=params["C"]
    phi=params["phi"]
    val=osc(x, iM, iT0, norm, phi)
    return (val-data)/eps

times = np.linspace(-0.1, 0.3, 1000)
freq = osc(times, 30, 0.18, 1, 0.0)
plt.figure(figsize=(12, 4))
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
plt.plot(times, freq)
plt.xlabel('Time (s) since '+str(tevent))
plt.ylabel('strain')
savefig(plt, dirpath + '/strain_analytic.png')

#----------------------------------------------------------------
# Fit
#----------------------------------------------------------------

printstep('Fit')
sample_times = white_data_bp.times.value
sample_data = white_data_bp.value
indxt = np.where((sample_times >= (tevent-0.17)) & (sample_times < (tevent+0.13)))
x = sample_times[indxt]
x = x-x[0]
white_data_bp_zoom = sample_data[indxt]

plt.figure(figsize=(12, 4))
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
plt.plot(x, white_data_bp_zoom)
plt.xlabel('Time (s)')
plt.ylabel('strain (whitened + band-pass)')
savefig(plt, dirpath + '/strain_whiten_bandpass_nofit.png')

import lmfit
from lmfit import Model, minimize, fit_report, Parameters

model = lmfit.Model(osc)
p = model.make_params()
p['Mc'].set(20)     # Mass guess
p['t0'].set(0.18)  # By construction we put the merger in the center
p['C'].set(1)      # normalization guess
p['phi'].set(0)    # Phase guess
unc = np.full(len(white_data_bp_zoom),20)
out = minimize(osc_dif, params=p, args=(x, white_data_bp_zoom, unc))
print(fit_report(out))
plt.plot(x, model.eval(params=out.params,t=x),'r',label='best fit')
savefig(plt, dirpath + '/strain_whiten_bandpass_fit.png')

#----------------------------------------------------------------
# Significance vs. time
#----------------------------------------------------------------

printstep('Search long time range')
def fitrange(data,xx,tcenter,trange):
    findxt = np.where((xx >= tcenter-trange*0.5) & (xx < tcenter+trange*0.5))
    fwhite_data = data[findxt]
    x = xx[findxt]
    x = x-x[0]
    model = lmfit.Model(osc)
    p = model.make_params()
    p['Mc'].set(30)
    p['t0'].set(trange*0.5)
    p['C'].set(1)
    p['phi'].set(0)
    unc=np.full(len(fwhite_data),20)
    out = minimize(osc_dif, params=p, args=(x, fwhite_data, unc))
    return abs(out.params["C"].value/out.params["C"].stderr),out.redchi

times = np.arange(-14, 14, 0.05)
times += tevent
sigs=[]
chi2=[]
for time in times:
        pSig,pChi2 = fitrange(white_data_bp.value, sample_times, time, 0.4)
        sigs.append(pSig)
        chi2.append(pChi2)

plt.figure(figsize=(12, 4))
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
plt.plot(times, sigs)
plt.xlabel('Time (s)')
plt.ylabel('N/$\sigma_{N}$')
savefig(plt, dirpath + '/strain_search_significance.png')

plt.figure(figsize=(12, 4))
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
plt.plot(times, chi2)
plt.xlabel('Time (s)')
plt.ylabel('$\chi^{2}$')
savefig(plt, dirpath + '/strain_search_chi2.png')

