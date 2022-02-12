import numpy as np
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plotly.colors as pcol

colors=getattr(pcol.qualitative, "Plotly")
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

import corner as corner
import os
import sys
from astropy import units as u
from astropy.modeling.models import BlackBody
from astropy.constants import c
from astropy.constants import k_B
from astropy.constants import h
from astropy.visualization import quantity_support
from scipy.stats import norm
from scipy.optimize import curve_fit
from decimal import Decimal
from itertools import compress

mpl.rcParams['text.usetex'] = True

def planck(x,temp):
    # Inputs need to be in GHz.
    z = h.value / (k_B.value*temp)
    return (2*h.value*x**3/c.value**2) * (1.0/(np.exp(x*z)-1))


def read_params(filename):
    labels  = []
    freq    = []
    datfile = []
    rmsfile = []
    fitgain = []
    fitoffs = []
    with open(filename,'r') as infile:
        for line in infile:
            if line.startswith('NUMBAND'):
                numbands = int(line.split('=')[1])
            if line.startswith('NUMGIBBS'):
                numgibbs = int(line.split('=')[1][:5])
            if line.startswith('NUMINCLUDE'):
                numinc   = int(line.split('=')[1])
    if numbands != numinc:
        incs = []
        with open(filename,'r') as infile:
            for line in infile:
                if line.startswith('INCLUDE_BAND'):
                    dummy = line.split('=')[1].strip()
                    if  '.true.' in dummy:
                        incs.append(line.split('=')[0].strip()[-3:])
                

    blabs = []
    bfreq = []
    dfile = []
    nfile = []
    gainz = []
    offz  = []
    if numbands == numinc:
        for band in range(numbands):
            blabs.append('BAND_LABEL'+str(band+1).zfill(3))
            bfreq.append('BAND_FREQ'+str(band+1).zfill(3))
            dfile.append('BAND_FILE'+str(band+1).zfill(3))
            nfile.append('BAND_RMS'+str(band+1).zfill(3))
        for band in range(numbands):
            with open(filename,'r') as infile:
                for line in infile:
                    if line.startswith(blabs[band]):
                        name = str.strip(line.split('=')[1])
                        labels.append(name)
                    if line.startswith(bfreq[band]):
                        fre  = str.strip(line.split('=')[1])
                        freq.append(float(fre))
    else:
        for inc in range(numinc):
        # for band in range(numbands):
            blabs.append('BAND_LABEL'+incs[inc])
            bfreq.append('BAND_FREQ'+incs[inc])
            dfile.append('BAND_FILE'+incs[inc])
            nfile.append('BAND_RMS'+incs[inc])
            gainz.append('BAND_FIT_GAIN'+incs[inc])
            offz.append('BAND_FIT_OFFSET'+incs[inc])            
        for inc in range(numinc):
            with open(filename,'r') as infile:
                for line in infile:
                    if line.startswith(blabs[inc]):
                        name = str.strip(line.split('=')[1])
                        labels.append(name)
                    if line.startswith(bfreq[inc]):
                        fre  = str.strip(line.split('=')[1])
                        freq.append(float(fre))
                    if line.startswith(dfile[inc]):
                        dafil = str.strip(line.split('=')[1])
                        name  = str.strip(dafil.split('/')[-1])
                        datfile.append(str(name))
                    if line.startswith(nfile[inc]):
                        rmfil = str.strip(line.split('=')[1])
                        name  = str.strip(rmfil.split('/')[-1])
                        rmsfile.append(str(name))
                    if line.startswith(gainz[inc]):
                        gfit = str.strip(line.split('=')[1])
                        name = str.strip(gfit.split('/')[-1])
                        if str(name) == '.false.':
                            fitgain.append(False)
                            # print(False)
                        elif str(name) == '.true.':
                            fitgain.append(True)
                            # print(True)
                        else:
                            print("didn't work")
                    if line.startswith(offz[inc]):
                        offfit = str.strip(line.split('=')[1])
                        name = str.strip(offfit.split('/')[-1])
                        if str(name) == '.false.':
                            fitoffs.append(False)
                            # print(False)
                        elif str(name) == '.true.':
                            fitoffs.append(True)
                            # print(True)
                        else:
                            print("didn't work")

    return labels, freq, numgibbs, datfile, rmsfile, numinc, fitgain, fitoffs

def load_data():
    global burnin, dir, files
    global freq, num_samp, labels, numbands
    global hfi, firas, dirbe
    global datfile, rmsfile
    global fitgain, fitoffs
    global amps, chisq, tdmean, yerr
    global gains, offs, band_chisqs

    try:
        dir     = sys.argv[1] #str(input('Which directory to mean? '))
        files   = os.listdir(dir)
        thing   = [file for file in files if 'param' in file]
        names, freq, num_samp, datfiles, rmsfiles, numbands, fitgain, fitoffs = read_params(dir+'/'+thing[0])
        labels  = [name.replace("'","") for name in names]
        labels  = [w.replace("_","\_") for w in labels]
        hfi     = ['npipe' in lab for lab in labels]
        firas   = ['firas' in lab for lab in labels]
        dirbe   = ['dirbe' in lab for lab in labels]
        datfile = [dat.replace("'","") for dat in datfiles]
        rmsfile = [rms.replace("'","") for rms in rmsfiles]
    except:
        print("Input which directory you wish to point to.")
        exit()

    freq        = np.asarray(freq)
    amps        = np.loadtxt(dir+'/HI_amplitudes.dat')
    amps        = amps.T
    chisq       = np.loadtxt(dir+'/HI_chisq.dat')
    tdmean      = np.loadtxt(dir+'/HI_Td_mean.dat')
    tdmean      = tdmean.T
    gains       = np.loadtxt(dir+'/band_gains.dat')
    offs        = np.loadtxt(dir+'/band_offsets.dat')
    gains       = gains.T
    offs        = offs.T
    try:
        band_chisqs = np.loadtxt(dir+'/band_chisq.dat')
        band_chisqs = band_chisqs.T
    except:
        print('old version, no band_chisq.dat file found')
        
#-----------------------------------------

def amp_dist():
    print("Plotting amplitude distributions")
    
    for i in range(numbands):
        #plt.title(labels[i]+'  Amplitude Distribution')
        plt.hist(amps[i],bins=int(np.sqrt(num_samp)),color='orange')
        xmin, xmax = plt.xlim()
        plt.xlabel(labels[i]+' Dust Amplitude',size=20)
        plt.ylabel('Count',size=20)
        plt.savefig(dir+'/amplitude_distribution_'+labels[i]+'.png',dpi=300,bbox_inches='tight')
        plt.close()
        #plt.show()

def check_corrs():

    niter1 = np.linspace(1,len(chisq),len(chisq))
    niter2 = np.linspace(1,len(tdmean),len(tdmean))
    pars = 1
    numgains = sum(fitgain)
    numoffs  = sum(fitoffs)
    
    if numgains != 0:
        pars += 1
        gainslabels = list(compress(labels,fitgain))
    if numoffs != 0:
        pars += 1
        offslabels = list(compress(labels,fitoffs))

    if pars == 1:

        tudes = np.empty(numbands)
        yerr  = np.empty(numbands)
        
        for i in range(numbands):    
            yerr[i]  = np.std(amps[i][burnin:])
            tudes[i] = np.mean(amps[i][burnin:])

        modelfreq = np.linspace(np.min(freq),np.max(freq),1000)
        beta      = np.linspace(1,len(chisq),len(chisq))

        for i in range(len(chisq)):
            par, perr = fit_beta(freq,amps[:,i],yerr)
            beta[i]   = par[1]
        
        diff = np.max(chisq[burnin:])-np.min(chisq[burnin:])

        mean = np.mean(chisq[burnin:])
        
        fig, ax = plt.subplots(3,1,figsize=(12,12),sharex=True)
        ax[0].plot(niter1[burnin:],chisq[burnin:],color='orange')
        ax[0].set_ylabel(r'$\chi^2$',size=12)
        ax[0].set_ylim([mean-diff,mean+diff])
        # ax[0].set_ylim([0.95*np.min(chisq[burnin:]),1.05*np.max(chisq[burnin:])])
        ax[1].plot(niter2[burnin:],tdmean[burnin:],color='red')
        ax[1].set_ylabel(r'$\langle T_d \rangle$',size=12)        
        ax[2].plot(niter1[burnin:],beta[burnin:],color='blue')
        
    elif pars == 2:
        fig, ax = plt.subplots(3,1,figsize=(12,9),sharex=True)
        ax[0].plot(niter1[burnin:],chisq[burnin:],color='orange')
        ax[0].set_ylabel(r'$\chi^2$',size=12)
        # ax[0].set_xlim([5,5000])
        ax[0].set_ylim([0.8*np.min(chisq[burnin:]),1.2*np.max(chisq[burnin:])])
        ax[1].plot(niter2[burnin:],tdmean[burnin:],color='red')
        ax[1].set_ylabel(r'$\langle T_d \rangle$',size=12)        
        if numgains != 0:
            for i in range(numgains):
                ax[2].plot(niter[burnin:],gains[fitgain][i][burnin:],label=gainslabels[i])
                ax[2].set_ylabel(r'Map gain',size=12)
                ax[2].legend()
        if numoffs != 0:
            for i in range(numoffs):
                ax[2].plot(niter[burnin:],offs[fitoffs][i][burnin:],label=offslabels[i])
                ax[2].set_ylabel(r'Map offset [MJy/sr]',size=12)
                ax[2].legend()

    elif pars == 3:

        print(np.min(chisq[burnin:]),np.max(chisq[burnin:]))
        fig, ax = plt.subplots(4,1,figsize=(8,6),sharex=True)
        ax[0].plot(niter[burnin:],chisq[burnin:],color='orange')
        ax[0].set_ylabel(r'$\chi^2$',size=12)
        # ax[0].set_xlim([5,5000])
        ax[0].set_ylim([0.95*np.min(chisq[burnin:]),1.05*np.max(chisq[burnin:])])
        ax[1].plot(niter[burnin:],tdmean[burnin:],color='red')
        ax[1].set_ylabel(r'$\langle T_d \rangle$',size=12)
        for i in range(numgains):
            ax[2].plot(niter[burnin:],gains[fitgain][i][burnin:],label=gainslabels[i])
            ax[2].legend()
            ax[2].set_ylabel(r'Map gain',size=12)
        for i in range(numoffs):
            ax[3].plot(niter[burnin:],offs[fitoffs][i][burnin:],label=offslabels[i])
            # ax[3].legend()
            ax[3].set_ylabel(r'Map offset',size=12)
                
    fig.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(dir+'/all_trace.png',dpi=300,bbox_inches='tight')
    plt.close()

    if pars == 1:
        data = np.vstack((chisq[burnin:],tdmean[burnin:],beta[burnin:]))
        cornerlabs = [r'$\chi^2$',r'$\langle T_d \rangle$',r'$\beta$']

        corner.corner(data.T,labels=cornerlabs)
    plt.savefig(dir+'/all_corner',dpi=300,bbox_inches='tight')
    # plt.show()
    plt.close()
        
def plot_amplitudes(pdf=False):

    print("Plotting mean amplitudes and fitting beta")
    print("")
    # print(freq)
    
    tudes = np.empty(numbands)
    yerr  = np.empty(numbands)
    
    for i in range(numbands):    
        yerr[i]  = np.std(amps[i][burnin:])
        tudes[i] = np.mean(amps[i][burnin:])

    modelfreq = np.linspace(np.min(freq),np.max(freq),1000)
    pars,perr = fit_beta(freq,tudes,yerr)
    slope = r'$\beta$ = '+ str(round(pars[1],3))+r' $\pm$ '+str(round(perr[1],3))
    
    plt.figure(figsize=(16,6))
    plt.text(1250,5e-5,slope,size=25)
    plt.plot(modelfreq,pars[0]*(modelfreq/353.)**pars[1], color='red',zorder=1000)
    # Plot FIRAS
    plt.scatter(freq[firas],tudes[firas],color='black')
    plt.errorbar(freq[firas],tudes[firas], yerr=yerr[firas],fmt='o',markersize=2,color='black',label='FIRAS')
    # Plot HFI
    plt.scatter(freq[hfi],tudes[hfi],color='grey',zorder=999)
    plt.errorbar(freq[hfi],tudes[hfi], yerr=yerr[hfi],fmt='o',markersize=2,color='grey',zorder=999,label='HFI')
    plt.legend()
    plt.xlim(0,3000)
    plt.ylim(-5e-5,1e-4)
    plt.title(r'Dust per $N_{\rm H_I}$ Amplitudes',size=20)
    plt.ylabel(r'Amplitude ${\rm [cm^2]}$',size=20)
    plt.xlabel('Frequency [GHz]',size=20)
    plt.tick_params(labelsize=15)
    if pdf == True:
        plt.savefig(dir+'/amplitudes.pdf',dpi=300,bbox_inches='tight')
    else:
        plt.savefig(dir+'/amplitudes.png',dpi=300,bbox_inches='tight')
    plt.close()

def plot_band_amp_chisq(pdf=False):

    print("Plotting mean band amplitudes/chisqs and fitting beta")
    print("")
    # print(freq)

    tudes       = np.empty(numbands)
    yerr        = np.empty(numbands)
    xsqmeans    = np.empty(numbands)
    chisquerror = np.empty(numbands)
    
    for i in range(numbands):
        tudes[i]       = np.mean(amps[i,burnin:])
        yerr[i]        = np.std(amps[i][burnin:])
        xsqmeans[i]    = np.mean(band_chisqs[i][burnin:])
        chisquerror[i] = np.std(band_chisqs[i][burnin:])
        
    modelfreq = np.linspace(np.min(freq),np.max(freq),1000)
    pars,perr = fit_beta(freq,tudes,yerr)
    slope = r'$\beta$ = '+ str(round(pars[1],3))+r' $\pm$ '+str(round(perr[1],3))
    
    fig, ax = plt.subplots(2,1,figsize=(16,8),gridspec_kw={'height_ratios': [3,1]},sharex=True)
    
    ax[0].plot(modelfreq,pars[0]*(modelfreq/353.)**pars[1], color='red',zorder=1000)
    ax[0].scatter(freq[firas],tudes[firas],color='black')
    ax[0].errorbar(freq[firas],tudes[firas], yerr=yerr[firas],fmt='o',markersize=2,color='black',label='FIRAS')
    ax[0].text(1250,5e-5,slope,size=25)
    # Plot HFI
    ax[0].scatter(freq[hfi],tudes[hfi],color='grey',zorder=999)
    ax[0].errorbar(freq[hfi],tudes[hfi], yerr=yerr[hfi],fmt='o',markersize=2,color='grey',zorder=999,label='HFI')
    ax[0].set_xlim(0,3000)
    ax[0].set_ylim(-5e-5,1e-4)
    # plt.title(r'Dust per $N_{\rm H_I}$ Amplitudes',size=20)
    ax[0].set_ylabel(r'Amplitude ${\rm [cm^2]}$',size=20)
    # Lower Panel
    ax[1].set_ylabel(r'Band $\chi^2$',size=20)
    ax[1].set_xlabel('Frequency [GHz]',size=20)
    ax[1].scatter(freq[firas],xsqmeans[firas],color='black')
    ax[1].errorbar(freq[firas],xsqmeans[firas],yerr=chisquerror[firas],fmt='o',markersize=2,color='k')
    ax[1].scatter(freq[hfi],xsqmeans[hfi],color='grey')
    ax[1].errorbar(freq[hfi],xsqmeans[hfi],yerr=chisquerror[hfi],fmt='o',markersize=2,color='grey')
    ax[1].set_yscale('log')
    ax[1].set_ylim([2e2,1e8])
    fig.subplots_adjust(wspace=0,hspace=0)
    plt.tick_params(labelsize=15)
    if pdf == True:
        plt.savefig(dir+'/amplitudes.pdf',dpi=300,bbox_inches='tight')
    else:
        plt.savefig(dir+'/amplitudes.png',dpi=300,bbox_inches='tight')
    plt.close()
    
def trace_amplitude():

    print("Plotting amplitude trace")
    
    niter = np.linspace(1,len(amps[0]),len(amps[0]))

    for i in range(numbands):
        plt.plot(niter,amps[i],label=labels[i])
    plt.yscale('log')
    plt.title('Trace of Amplitude')
    plt.xlabel('Iteration Number')
    plt.ylabel(r'Amplitude ${\rm [cm^2]}$')
    plt.legend(loc='best')
    plt.savefig(dir+'/amplitude_trace',dpi=300,bbox_inches='tight')
    #plt.show()
    plt.close()

def trace_chisq():

    print("Plotting chisq trace")

    niter = np.linspace(1,len(chisq),len(chisq))

    plt.plot(niter,chisq,color='orange')
    # plt.yscale('log')
    plt.title(r'Trace of $\chi^2$')
    plt.xlabel('Iteration Number')
    plt.ylabel(r'$\chi^2$')
    plt.savefig(dir+'/chisq_trace',dpi=300,bbox_inches='tight')
    plt.show()
    plt.close()

def trace_gains():

    print("Plotting gain trace")

    num = sum(fitgains)
    
    niter = np.linspace(1,len(gains[0]),len(gains[0]))
    
    fig,axes = plt.subplots(3,2,figsize=(6,6),sharex=True)
    fig.tight_layout(pad=2.0)

    axes[0][0].plot(niter,gains[0])
    axes[0][0].set_title(labels[0])
    axes[1][0].plot(niter,gains[1])
    axes[1][0].set_title(labels[2])

    axes[2][0].plot(niter,gains[2])
    axes[2][0].set_title(labels[2])
    axes[2][0].set_xlabel('Gibbs Iteration',size=10)

    axes[0][1].plot(niter,gains[3])
    axes[0][1].set_title(labels[3])
    
    axes[1][1].plot(niter,gains[4])
    axes[1][1].set_title(labels[4])
    
    axes[2][1].plot(niter,gains[5])
    axes[2][1].set_title(labels[5])
    axes[2][1].set_xlabel('Gibbs Iteration',size=10)
    plt.savefig(dir+'/trace_gains',dpi=300,bbox_inches='tight')
    plt.close()
    # plt.show()
    
def trace_all():                
    print("Plotting all trace")

    niter = np.linspace(1,len(gains[0]),len(gains[0]))
    
    fig,axes = plt.subplots(3,6,figsize=(12,6),sharex=True)
    fig.tight_layout(pad=2.0)

    for i in range(numbands):

        axes[0][i].plot(niter[1:],gains[i][1:])
        axes[0][i].set_title(labels[i])
        axes[1][i].plot(niter[1:],offs[i][1:])
        axes[2][i].plot(niter[1:],amps[i][1:])
        axes[2][i].set_xlabel('Gibbs Iteration')

    axes[0][0].set_ylabel('Gain')
    axes[1][0].set_ylabel('Offset')
    axes[2][0].set_ylabel('HI Amplitude')
    plt.savefig(dir+'/trace_all',dpi=300,bbox_inches='tight')
    plt.close()
    # plt.show()

def inspect_pix_sed(pix,sample):
    mask  = hp.read_map('data/FIRAS_mask.fits')
    npix  = np.shape(mask)[0]
    nside = hp.npix2nside(npix)
    data  = np.empty((numbands,npix))
    model = np.empty((numbands,npix))
    sample = str(int(sample)).zfill(5)
    td_map = hp.read_map(dir+'/T_d_'+sample+'.fits')

    bb = BlackBody(temperature=td_map[pix]*u.K)
    
    for i in range(numbands):
        flux = bb(freq[i]*u.GHz).to(u.MJy/u.sr)
        fre = str(int(freq[i])).zfill(4)
        if freq[i] < 652.0 :
            mode = 'lowf'
            # print('lowf')
        else:
            mode = 'highf'
            # print('highf')
        data[i][:] = hp.read_map('/home/daniel/data/firas/fits/FIRAS_map_'+fre+'GHz_'+mode+'_nocmb.fits')
        model[i][:] = hp.read_map(dir+'/firas_'+fre+'_hi_amplitude_'+sample+'.fits')

        plt.scatter(freq[i],data[i][pix],label='data')
        plt.scatter(freq[i],model[i][pix]*flux,label='model')
    plt.savefig(dir+'/sed_sample_'+sample+'_pix_'+str(pix),dpi=300,bbox_inches='tight')
    plt.close()

def posterior(band,sample):
    samplestr = str(sample).zfill(5)
    print(samplestr)
    
    amp_mean  = np.mean(amps[band])
    amp_std   = yerr[band]
    amp_range = np.linspace(-1e-5,1e-5,50)
    chisq     = np.empty(50)
    chisq[:]  = 0.0
    
    hi = hp.read_map('data/hi/HI_vel_filter_7deg_0016.fits')
    td = hp.read_map(dir+'/T_d_k'+samplestr+'.fits')

    model_map = hi*planck(freq[band]*1e9,td)

    if 'firas' in labels[band]:
        band_dir = '/home/daniel/data/firas/fits/'
    if 'npipe' in labels[band]:
        band_dir = 'data/npipe/'

    data_map = hp.read_map(band_dir+datfile[band])
    rms_map  = hp.read_map(band_dir+rmsfile[band])
    chi_map  = np.empty(len(data_map))

    band_name = labels[band].replace("\_","_")

    # os.mkdir(dir+'/'+band_name)
    for i in range(len(amp_range)):
        count = 0
        suff = str(round(Decimal(amp_range[i]),7))
        for j in range(len(model_map)):
            if td[j] == hp.UNSEEN:
                chi_map[j] = hp.UNSEEN
                continue
            chisq[i]  += ((amp_range[i]*model_map[j]-data_map[j])/rms_map[j])**2
            chi_map[j] = ((amp_range[i]*model_map[j]-data_map[j])/rms_map[j])**2
            count = count + 1
        # hp.mollview(chi_map,min=0,max=1)
        # plt.savefig(dir+'/'+band_name+'/'+band_name+'_chisq_map_'+suff+'.png')
        # plt.close()

    chisq = chisq/((count*numbands)-count-1)
    post = np.exp(-0.50*chisq)

    plt.title(labels[band],size=15)
    # plt.plot(amp_range,chisq)
    plt.plot(amp_range,post)
    plt.axvline(amp_mean,color='k',linestyle='--')
    plt.xlabel('Amplitude',size=15)
    plt.ylabel(r'$P(a|T_d)$',size=15)
    plt.savefig(dir+'/'+labels[band]+'_post_given_amplitude')
    plt.close()

def fit_beta(x,y,sigmay):
    def func(x,a,beta):
        return a*(x/353.)**beta

    res = curve_fit(func,x,y,sigma=sigmay)#,p0=p0)

    perr = np.sqrt(np.diag(res[1]))
    
    return res[0], perr
    
    
USAGE = f"Usage: python {sys.argv[0]} [directory] [option]\n Option list: \n -- help \n -corr \n -amps \n -amp_chisqs"
    
def plot() -> None:
    command = sys.argv[2:]
    if not command:
        raise SystemExit(USAGE)
    load_data()
    trace_chisq()
    global burnin
    burnin = int(input("Burnin #? "))
    for i in command:
        if (i == '--help'):
            raise SystemExit(USAGE)
        if (i == '-corr'):
            check_corrs()
        if (i == '-amps'):
            plot_amplitudes(pdf=False)
        if (i == '-amp_chisqs'):
            plot_band_amp_chisq()

plot()
