import numpy as np
from numpy.linalg import inv
import os.path 
import pandas as pd

import astropy.io.votable as vot
from astropy.io import ascii
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.coordinates import SkyCoord

from astroquery.gaia import Gaia
import matplotlib.pyplot as plt

from . import DATADIR
from . import gloess_plotting_options as gf_plot
from . import utils as vs

from pathlib import Path


import itertools

np.seterr(divide='ignore')
np.seterr(over='ignore')

## Main Gloess class

class Gloess(object):
    """ Gloess class
    initialise as gloess = Gloess(jds, mags, errs, period, smooth='auto', band='')
    jds = mjds
    mags = mags
    errs = photometric errors
    period = period of star (days)
    smooth = smoothing parameter for that band. if 'auto' then gloess finds the best smoothing paramters
    band = bandpass. defaults to '', not usually needed but can be useful for bookkeeping. 
        """
    @staticmethod
    def phase_jds(jds, period):
        """ phases the JDs according to the period """
        
        phase = (jds / period) - np.floor(jds/ period)
        return(phase)
        
    def __init__(self, jds, mags, errs, period, smooth='auto', band=''):
        """ initialising the Gloess object
        sets up the phase, y, yerr arrays to repeat over 5 cycles
        finds the smoothing parameter if not fixed to user value
        gets the phase matrix
        """

        y_temp = mags[~mags.isna()]
        if len(y_temp) == 0:
            raise Exception('No data in this band.')
        self.y = np.concatenate((y_temp, y_temp, y_temp, y_temp, y_temp))

        jd_temp = jds[~mags.isna()]
        self.jds = np.concatenate((jd_temp, jd_temp, jd_temp, jd_temp, jd_temp))
        
        phase_temp = self.phase_jds(jds[~mags.isna()], period)     
        self.phase = np.concatenate((phase_temp, phase_temp+1., phase_temp+2., phase_temp+3., phase_temp+4.))

        err_temp = errs[~mags.isna()]
        
        ## If no uncertainty given, set uncertainty equal to 9.99 mag
        ## This is so it will still attempt the fit even if there are no uncertainties but will downweight the ones with missing uncertainties if some exist.  
        
        err_temp = np.nan_to_num(err_temp, copy=False, nan=9.99)

        self.yerr = np.concatenate((err_temp, err_temp,err_temp,err_temp,err_temp))

        self.band = band

        if smooth=='auto':
            self.sm = self.find_smoothing_params()
        elif (type(smooth) == float and smooth < 1 and smooth > 0)  :
            self.sm = smooth
        else:
            print('bad smoothing parameter - using auto smoothing')
            self.sm = self.find_smoothing_params()
        mat = self.phases_to_fit()
        self.phase_matrix = mat
 
    def find_smoothing_params(self):
        """ find the smoothing parameters by finding the largest distance between two points"""
        sorted_phases = np.sort(self.phase)
        sorted_phases = np.concatenate((sorted_phases, sorted_phases+1.))
        sm = np.max(np.diff(sorted_phases))
        """ putting a floor of 0.03 on the smoothing parameter so that very well sampled data isn't overfit """
        if sm < 0.03:
            sm = 0.03
        return(sm)

    def phases_to_fit(self):
        """ gets the phase matrix
        for every fake_phase point (500 points over 5 cycles), 
        find the distance beween that point and every data phase point. 
        returns a matrix with n_points x 500 elements"""
        fake_phases = -0.99 + 0.01*(np.arange(0,500))
        fake_phases = np.reshape(fake_phases, (500, 1))
        data_phases = np.reshape(self.phase, (1,len(self.phase)))

        fit_phases = np.subtract(data_phases, fake_phases)
        return(fit_phases)

    def get_weights(self, phase_array):
        """ finds the weight of each point in the fit
        uses the distances from the phase matrix, the smoothing parameter and the photometric error
        each point has a gaussian weighting centred on the current datapoint. """
        dist  = np.abs(phase_array)
        weight = np.exp(-0.5*(dist**2) /self.sm**2) / self.yerr 
        return(weight)

    def fit_one_band(self):
        """ does the gloess fit on a single band, looping over the 500 fake phase points
        for each fake phase point, determine the weights of each data point to the fit at the fake phase point
        solve the y = a*x**2 + b*x + c fit at that point
        fitted mag at that point is c

        If the fit is a singular matrix, sets the fitted mag to np.nan
        """
        dat = np.zeros(500)
        for x in range(0,500):
            phase_array = np.asarray(self.phase_matrix[x].flat[:])
            weights = self.get_weights(phase_array)
            try:
                dat[x] = self.fit2(phase_array, weights)
    
            except np.linalg.LinAlgError as e:
                if 'Singular matrix' in str(e):
                    dat[x] = np.nan            
        return(dat)

    def fit2(self, x1, wt):
        """ gloess fitting function using numpy linalg.
        does the weighted LSQ fit of y = a*x**2 + b*x + c at the given data point using the phase array and weights
        fitted mag at that point is c from the LSQ fit
        """
        y1 = self.y
        sigma = 1.0 / wt
        sigma2 = sigma**2
        x2 = x1**2
        x3 = x1**3
        x4 = x1**4
	

        C =  np.nansum(y1 / sigma2)
        E =  np.nansum((y1*x1) / sigma2)
        G = np.nansum((y1*x2) / sigma2)
	
        a11 = np.nansum(1.0 / sigma2)
        a12 = np.nansum(x1 / sigma2)
        a13 = np.nansum(x2 / sigma2)
        a23 = np.nansum(x3 / sigma2)
        a33 = np.nansum(x4 / sigma2)
        a21 = a12
        a22 = a13
        a31 = a13
        a32 = a23
        
        A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
        A_I = np.linalg.inv(A)
        M = np.array([C, E, G])
        R = np.dot(M, A_I)
        yest = R[0]
        return(yest)
    
    def mean_mag(self):
        """ 
        find the flux averaged mean magnitude of the fit in one cycle.
        Uses the middle(ish) of the fit to avoid the weirdness going on at boundaries.  
        """
        mags = self.fit_one_band()[200:300]
        fluxes = 10**(-(mags - 25)/2.5)
        av_flux = np.mean(fluxes)
        av_mag = 25 - 2.5 * np.log10(av_flux)
        return(av_mag)
    

""" Below here are convenience functions for reading data, plotting, saving output etc. """

""" Functions that do calculations """

def rising_branch_mean(fit_results, band_num):
    """ Find the phase of the mean mag on the rising branch 
        Use this to offset to common zero-phase
        fit_results = fit results from gloess (all bands)
        band_num = int corresponding to the number of the band that you want to set zero point with
        TO DO: Update this so you can give it a band name instead?
    """
    mags = fit_results[band_num, 200:300]
    fake_phases = -0.99 + 0.01*(np.arange(0,500))
    phases = fake_phases[200:300]
    fluxes = 10**(-mags / 2.5)
    mean_flux = np.mean(fluxes)
    mean_mag = -2.5*np.log10(mean_flux)
    ## find rising branch
    
    brightest_idx = np.argmin(mags)
    faintest_idx = np.argmax(mags)
    
    ## check direction
    
    while faintest_idx > brightest_idx:
        ## shift back by 0.5 in phase
        phases = fake_phases[150:250]
        mags = fit_results[band_num, 150:250]
        brightest_idx = np.argmin(mags)
        faintest_idx = np.argmax(mags)
    
    rising_mags = mags[faintest_idx:brightest_idx]
    
    idx_mean = (np.abs(rising_mags-mean_mag)).argmin()
    
    mean_rising_phase = phases[idx_mean + faintest_idx] - np.floor(phases[idx_mean + faintest_idx])

    return(mean_rising_phase)

def fitted_phases(fit_results, phase_ref_band=None, filters=None, phase_ref_type='rbm', period=None, JD=None):
    """ function to get the offset phases for plotting etc
    input:
        fit_results: fit results from gloess
        phase_ref_band: which band to offset to. Will find rising branch mean in that band for phase=0
        filters: filters in the fit results
        phase_ref_type: how to set phi=0 point. Defaults to 'rbm'.
            options for phase_ref_type:
                'rbm': (Default) rising branch mean
                'JD': use arb_phase_offset with 'JD' as off_type
                'max_light': use arb_phase_offset with 'max_light' as off_type
                'min_light': use arb_phase_offset with 'min_light' as off_type
        period = period in days
        JD: JD to use for reference if using phase_ref_type='JD'.

    output: 
    fake_phases: phases of the fit result points offset accoring to phase_ref_band rising branch mean
    """
    ref_phase = 0
    fake_phases = -0.99 + 0.01*(np.arange(0,500))

    offset_types = ['rbm', 'JD', 'max_light', 'min_light']
    if phase_ref_type not in offset_types:
        raise ValueError("Invalid phase_ref_type type. Expected one of: %s" % offset_types)

    if phase_ref_band != None:
        if phase_ref_band in filters:
            if phase_ref_type == 'rbm':
                band_num = np.argwhere(np.array(filters)==phase_ref_band)[0,0]
                ref_phase = rising_branch_mean(fit_results, band_num)
                fake_phases = (fake_phases - ref_phase)
            elif phase_ref_type == 'JD':
                if JD==None:
                    raise ValueError('Must provide reference JD for offset to JD')
                if period==None:
                    raise ValueError('Must provide period for offset to JD')
                ref_phase = phase_jds(JD, period)
                fake_phases = (fake_phases - ref_phase) - np.floor(fake_phases - ref_phase)
                fake_phases_offset = -0.99 + 0.01*(np.arange(0,500))
                fake_phases = fake_phases + fake_phases_offset  
            elif phase_ref_type=='max_light' or phase_ref_type=='min_light':
                if phase_ref_band==None:
                    raise ValueError('Must provide phase reference band for offset to max/min light')
                band_num = np.argwhere(np.array(filters)==phase_ref_band)[0,0]

                if len(fit_results[band_num]) <= 200:
                    mags = fit_results[band_num]
                else:   
                    mags = fit_results[band_num, 200:400]
                if phase_ref_type=='max_light':
                    ref_phase = fake_phases[np.argmin(mags)] ## backwards because mags
                if phase_ref_type=='min_light':
                    ref_phase = fake_phases[np.argmax(mags)] ## backwards because mags
                fake_phases = (fake_phases - ref_phase) - np.floor(fake_phases - ref_phase)
                fake_phases_offset = -0.99 + 0.01*(np.arange(0,500))
                fake_phases = fake_phases + fake_phases_offset  

            if np.min(fake_phases) < -1.5:
                fake_phases = fake_phases + 1.
                ref_phase = ref_phase + 1.
            
    return(fake_phases, ref_phase)

def arb_phase_offset(fit_results, off_type='JD', JD=None, phase_ref_band=None, filters=None, period=None):
    """ function to offset the phases so that phi=0 corresponds to a specific point in the light curve
        input:
            fit_results: fit results from gloess
            off_type: what to use as the reference points. Defaults to 'JD'
                available options:
                    'JD': set phi=0 to this JD
                    'max_light': maximum light in reference band
                    'min_light': minimum light in reference band
                'max_light' and 'min_light' both need 'phase_ref_band' to be set
            JD: JD value to use as reference if off_type='JD'
            phase_ref_band: band to use as reference if off_type='min_light' or 'max_light'. 
            filters: filters that you want to apply the offset to. 
            period: period in days
        output:
            fake_phases: phases corresponding to each point in the gloess fit
            ref_phase: phase of the reference point from the original phase_jds function  
    """
    offset_types = ['JD', 'max_light', 'min_light']
        
    fake_phases = -0.99 + 0.01*(np.arange(0,500))

    if off_type not in offset_types:
        raise ValueError("Invalid offset type. Expected one of: %s" % offset_types)
    
    if off_type=='JD':
        if JD==None:
            raise ValueError('Must provide reference JD for offset to JD')
        if period==None:
            raise ValueError('Must provide period for offset to JD')
        ref_phase = phase_jds(JD, period)
    
    if off_type=='max_light' or off_type=='min_light':
        if phase_ref_band==None:
            raise ValueError('Must provide phase reference band for offset to max/min light')
        if filters==None:
            raise ValueError('Must provide filters for offset to max/min light')
        band_num = np.argwhere(np.array(filters)==phase_ref_band)[0,0]

        if len(fit_results[band_num]) <= 200:
            mags = fit_results[band_num]
        else:
            mags = fit_results[band_num, 200:400]
        if off_type=='max_light':
            ref_phase = fake_phases[np.argmin(mags)] ## backwards because mags
        if off_type=='min_light':
            ref_phase = fake_phases[np.argmax(mags)] ## backwards because mags

  
    fake_phases = (fake_phases - ref_phase) - np.floor(fake_phases - ref_phase)

    fake_phases_offset = -0.99 + 0.01*(np.arange(0,500))

    fake_phases = fake_phases + fake_phases_offset
    
    return(fake_phases, ref_phase)



def get_lc_stats(fit_results):
    """ Determine statistics of fitted lightcurve from gloess
        av_mag = mean of mag values
        int_av_mag = flux averaged mag
        amp = amplitude
    """
    if len(fit_results) <= 200:
        mags = fit_results
    else:
        mags = fit_results[200:400]
    av_mag = np.mean(mags)
    fluxes = 10**(-mags/2.5)
    av_flux = np.mean(fluxes)
    int_av_mag = -2.5*np.log10(av_flux)
    amp = np.max(mags) - np.min(mags)
    return(av_mag, int_av_mag, amp)

def convert_to_jd(time, target=None, input_format='mjd', location=None):
    """ Convert input time to JD
        time = input time
        target = Name of target to get skycoords
        input_format = format of input time. 
        location = None, needed for getting correct heliocentric corrected time
    """
    if input_format!='hjd':
        in_time = Time(time, format=input_format, scale='utc')
        jd = in_time.jd
    else:
        in_time = Time(time, format='jd', scale='utc')
        coords = SkyCoord.from_name(target)
        ltt_helio = Time(time, format='jd', scale='utc').light_travel_time(coords, 'heliocentric', location=location)
        jd = Time(in_time.utc - ltt_helio, format='jd', scale='utc', location=location).jd
    return(jd)



def phase_jds(jds, period):
        """ 
        phases the JDs according to the period 
        jd can be either MJD or JD (but make sure you know what you're using!)
        period should be in days. 
        """
        phase = (jds / period) - np.floor(jds/ period)
        return(phase)


""" 
gloess_fit_plot_df is the main function most people will probably use. 
This uses the dataframe containing the photometry and dates to do the gloess fits (defaults to all the bands in the file)
Default is to save the fit output and to save a pdf of the lightcurve. 

"""

def gloess_fit_plot_df(df, period, target, filters = [], ax=None, plot=True, multi_panel=False,
                       save_pdf=True, save_fit=True, return_means=False, 
                       phase_ref_band=None, phase_ref_type='rbm', phase_epoch=None,source_id=None, save_means=False, ismags=True, yrange=None, JD=None):
    
    """
    df = photometry dataframe
    period = known period
    target = object name
    filters = which bands to fit. defaults to all bands in the file (i.e. any columns that start with mag_)
    ax = useful if you want to add things to a plot later. defaults to None
    plot = True (default). set to false if you don't want it to plot the lightcurve
    save_pdf = True (default). save the lightcurve to a pdf. Default name is target.replace(' ', '_') + '_gloess_lc.pdf'
    save_fit = True (default). save the gloess fits to a csv file. Default filename is target.replace(' ', '_') + '_gloess_fit.csv'. 
                                Should add option to change filename for this (and for the plot pdf)
    return_means = False (default). Return mean mags and amplitudes
    phase_ref_band = None (default). Select which band you want to use as phase reference. 
                    If used, shifts all phases so that phase=0 is defined in this band. Type of offset it set by phase_ref_type (defaults to rising branch mean)
    phase_ref_type = 'rbm' (default). Other options are 'JD', 'max_light' and 'min_light'
    phase_epoch = None (default). Shifts phases so that phase=0 is at specified epoch. 
    source_id =  None (default). Gaia DR3 source_id. Can either pass the source_id if you know it, use 'lookup' to look up the source_id from the target name
                or leave as None if you don't want it. Only used here for plots. 
    save_means = False (default). Save means to output text file
    JD = None (default). JD to use for phase offset if phase_ref_type='JD'

    returns:
        (if return_means == False):
            fit_results: gloess fit results, 
            gloess: the gloess object
            filters: which filters used
            phase_offset: phase offset value (will be 0 if no phase_ref_band given)
            fig: matplotlib figure object so you can edit the plot if you want. 
        (if return_means == True):
            fit_results: gloess fit results, 
            gloess: the gloess object
            filters: which filters used 
            fig: matplotlib figure object so you can edit the plot if you want. 
            av_mags: av mags from get_lc_stats
            int_av_mags: flux averaged mags from get_lc_stats
            amps: amplitudes from get_lc_stats
            phase_offset: phase_offset found in rising_branch_mean
        
    """
    if len(filters)==0:
        mag_cols = [i for i in df.columns if 'mag_' in i]
        keep_cols = []
        for i in range(len(mag_cols)):
            n_data = len(df.dropna(subset=mag_cols[i]))
            if n_data >0:
                keep_cols.append(mag_cols[i])
        mag_cols = keep_cols
        filters = [sub.replace('mag_', '') for sub in keep_cols]
    else:
        mag_cols = []
        for i in filters:
            mag_cols.append('mag_' + i)
        keep_cols = []
        for i in range(len(mag_cols)):
            n_data = len(df.dropna(subset=mag_cols[i]))
            if n_data >0:
                keep_cols.append(mag_cols[i])
        mag_cols = keep_cols

    fit_results = np.zeros((len(filters), 500))
    gloess = []
    if return_means==True:

        av_mags = np.ones(len(filters)) * np.nan
        int_av_mags = np.ones(len(filters)) * np.nan
        amps = np.ones(len(filters)) * np.nan

    fake_phases = -0.99 + 0.01*(np.arange(0,500))
    for i in range(len(filters)):
        band = filters[i]
        #print(f'{band}')
        mag_col = 'mag_' + band
        err_col = 'err_' + band
        xx = df['JD']
        yy = df[mag_col]
        yerr = df[err_col]
        gloess.append(Gloess(xx, yy, yerr, period, smooth='auto', band=band))
        fit_results[i] = gloess[i].fit_one_band()
          
    gloess = np.array(gloess)
    

    ref_phase = 0.0
    phase_offset = 0.0

    if phase_ref_band != None:
        if phase_ref_band in filters:
            band_num = np.argwhere(np.array(filters)==phase_ref_band)[0,0]

            fake_phases, ref_phase = fitted_phases(fit_results, phase_ref_band, filters, phase_ref_type=phase_ref_type, period=period, JD=JD)

            #ref_phase = rising_branch_mean(fit_results, band_num)
            #print(f'ref_phase = {ref_phase}')
            #fake_phases = (fake_phases - ref_phase)
            phase_offset = ref_phase
        
            if np.min(fake_phases) < -1.5:
                fake_phases = fake_phases + 1.
                phase_offset = ref_phase + 1.
              
        else:
            phase_ref_band = None
    
    elif phase_epoch != None:
        ref_phase = phase_jds(phase_epoch, period)
        fake_phases = (fake_phases - ref_phase)
        phase_offset = ref_phase
        
        if np.min(fake_phases) < -1.5:
            fake_phases = fake_phases + 1.
            phase_offset = ref_phase + 1.

        

    if plot==True:

        if source_id == 'lookup':
            source_id = vs.get_gaia_source_id(target)
            if np.isnan(float(source_id)):
                source_id = None
        if multi_panel==False:
            fig = lightcurve_plotter(target, df, gloess, fit_results, filters, period, phase_offset, ax=ax, source_id = source_id, save_pdf=save_pdf, ismags=ismags,yrange=yrange)

        if multi_panel==True:
           fig = lightcurve_plotter_multi_panel(target, df, gloess, fit_results, filters, period, phase_offset, ax=ax, source_id = source_id, save_pdf=save_pdf,yrange=yrange)
           #return(fit_results, gloess, filters, phase_offset, fig, fit_res, yy)
    else:
        fig = np.nan
        
    if save_fit==True:
        """ Updated 21/5/25 to save offset phases in file rather than original phases. Wrapped to go between 0 and 2"""
        #fake_phases = -0.99 + 0.01*(np.arange(0,500))
        df_fit = pd.DataFrame(columns=['phase', 'mag_G', 'mag_BP', 'mag_RP'])
        df_fit['phase'] = (fake_phases[200:400]-1.0) % 2.0
        for i in range(len(filters)):
            band = filters[i]
            mag_col = 'mag_' + band
            df_fit[mag_col] = fit_results[i, 200:400]
    

        fit_csv =  target.replace(' ', '_') + '_gloess_fit.csv'
        #print(fit_csv)
        df_fit.to_csv(fit_csv, index=False, float_format='%.4f')
        print(f'Saved gloess fit to {fit_csv}')

    if return_means==True:
        for i in range(len(filters)):
            av_mags[i], int_av_mags[i], amps[i] = get_lc_stats(fit_results[i, 200:400])

    if save_means==True:
        outfile = target + '_means_data.txt'
        means_array = list(zip(filters, av_mags, int_av_mags, amps))
        np.savetxt(outfile, means_array)

    if return_means==True:
        if phase_ref_band == None:
            phase_offset = np.nan
        return(fit_results, gloess, filters, fig, av_mags, int_av_mags, amps, phase_offset)

    else:
        return(fit_results, gloess, filters, phase_offset, fig)

def lightcurve_plotter(target, phot_df, gloess, fit_results, filters, period, phase_offset=0, ax=None, source_id = None, save_pdf=True, ismags=True,yrange=None):
    """
    Plotting gloess lightcurves
    input: 
    target = target name
    df = dataframe with the photometric data
    gloess = gloess object containing the light curve fits
    fit_results = gloess fit results
    filters = list of filters you want to fit
    fake phases = x values of the light curve. can be offset elsewhere in the code to phase up data to a specific reference point
    period = period in days
    ax = axis to plot on. Defaults to none but you can pass it an existing axis to add to the plot
    source_id = Gaia source ID. Defaults to None
    save_pdf = save pdf of the figure. Defaults to True and *WILL OVERWRITE PREVIOUS PLOTS OF THE SAME LIGHTCURVE*

    returns fig object with the figure
    """
    # glo_offs = gf_plot.glo_offs
    # glo_cols = gf_plot.glo_cols
    # plot_symbols = gf_plot.plot_symbols

    glo_cols, glo_offs, plot_symbols, glo_standard_filters, glo_plot_order,glo_labels,glo_mags_axis= gf_plot.extra_bands(filters)

    fake_phases = -0.99 + 0.01*(np.arange(0,500))
    fake_phases = fake_phases - phase_offset
    if ax==None:
        fig = plt.figure(figsize=(14,8))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
    else:
        ax1=ax
    ax_min = 99.
    ax_max = -99.
    
    #for i in range(len(filters)):
    for key in glo_standard_filters:
        f = key
        #f = filters[i]
        mag_col = 'mag_' + f
        if (mag_col in phot_df) and (f in filters):

            i = filters.index(f)
            if len(phot_df[mag_col].dropna()) > 0:
                err_col = 'err_' + f    
                xx = phot_df['JD']
                yy = phot_df[mag_col]
                yerr = phot_df[err_col]
                xp = gloess[i].phase_jds(xx, period) - phase_offset
                fp = fake_phases[200:300]-1.
                fr = fit_results[i, 200:300]
                
                #ax1.errorbar(np.concatenate((xp, xp+1, xp+2,xp+3)), np.concatenate((yy,yy,yy,yy)), yerr=np.concatenate((yerr,yerr,yerr,yerr)),marker=plot_symbols[f], ls='None', ms=5, color=glo_cols[f])
                #axs.plot(np.concatenate((fp, fp+1, fp+2,fp+3)), np.concatenate((fr, fr, fr, fr)), ls='-', color='Gray',zorder=10)
                if glo_standard_filters[f] == True:
                    ax1.plot(np.concatenate((xp, xp+1, xp+2,xp+3)), np.concatenate((yy,yy,yy,yy))+glo_offs[f],marker=plot_symbols[f], ls='None', ms=5, color=glo_cols[f])
                    #ax1.plot(fake_phases[200:500]-1, fit_results[i, 200:500]+glo_offs[f], ls='-', color=glo_cols[f])
                    ax1.plot(np.concatenate((fp, fp+1, fp+2,fp+3)), np.concatenate((fr, fr, fr, fr))+glo_offs[f], ls='-', color='Gray',zorder=10)
                else:
                    ax2.plot(np.concatenate((xp, xp+1, xp+2,xp+3)), np.concatenate((yy,yy,yy,yy))+glo_offs[f],marker=plot_symbols[f], ls='None', ms=5, color=glo_cols[f])
                    #ax2.plot(fake_phases[200:500]-1, fit_results[i, 200:500]+glo_offs[f], ls='-', color=glo_cols[f])
                    ax2.plot(np.concatenate((fp, fp+1, fp+2,fp+3)), np.concatenate((fr, fr, fr, fr))+glo_offs[f], ls='-', color='Gray',zorder=10)


                band_min = fit_results[i, 200:300].min()+glo_offs[f]
                band_max = fit_results[i, 200:300].max()+glo_offs[f]

                if band_min < ax_min:
                    ax_min = band_min
                if band_max > ax_max:
                    ax_max = band_max
    if source_id != None:
        plt.suptitle(f'{target}, Gaia DR3 {source_id}, P = {period:.5f} d')
    else:
        plt.suptitle(f'{target}, P = {period:.5f} d')
    handles, labels = ax1.get_legend_handles_labels() 
    ax1.legend(handles[::-1],labels[::-1],bbox_to_anchor=(-0.35, 1), loc='upper left', borderaxespad=0, numpoints=1)
    #ax1.legend(flip_legend(handles[::-1], 3), flip_legend(labels[::-1],3), mode="expand", borderaxespad=0, ncol=3, bbox_to_anchor=(0, 1.1, 1, 0.2), loc="lower left")
    if ismags==True:
        ax1.axis([0,2,ax_max+0.2, ax_min-0.2])  
    else:
        ax1.axis([0,2,ax_min-0.1,ax_max+0.1])  

    if yrange==None:
        ax1.axis([0,2,ax_max+0.2, ax_min-0.2])  
    else:
        ax1.axis([0,2,yrange[0], yrange[1]]) 


    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Magnitude')

    handles, labels = ax2.get_legend_handles_labels() 
    #ax1.legend(handles[::-1],labels[::-1],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., numpoints=1)
    ax2.legend(handles[::-1],labels[::-1],bbox_to_anchor=(1.3, 1), loc='upper right', borderaxespad=0, numpoints=1)
    #ax2.legend(flip_legend(handles[::-1],3), flip_legend(labels[::-1],3), mode="expand", borderaxespad=0, ncol=3, bbox_to_anchor=(0, 1.1, 1, 0.2), loc="lower left")
    if yrange==None:
        ax2.axis([0,2,ax_max+0.2, ax_min-0.2])  
    else:
        ax2.axis([0,2,yrange[0], yrange[1]]) 


    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Magnitude')


    if save_pdf==True:
        plot_file = target.replace(' ', '_') + '_gloess_lc.pdf'
        plt.savefig(plot_file, bbox_inches="tight")
    return(fig)

def lightcurve_plotter_multi_panel(target, phot_df, gloess, fit_results, filters, period, phase_offset=0, ax=None, source_id = None, save_pdf=True,yrange=None):
    """
    Plotting gloess lightcurves
    input: 
    target = target name
    df = dataframe with the photometric data
    gloess = gloess object containing the light curve fits
    fit_results = gloess fit results
    filters = list of filters you want to fit
    fake phases = x values of the light curve. can be offset elsewhere in the code to phase up data to a specific reference point
    period = period in days
    ax = axis to plot on. Defaults to none but you can pass it an existing axis to add to the plot
    source_id = Gaia source ID. Defaults to None
    save_pdf = save pdf of the figure. Defaults to True and *WILL OVERWRITE PREVIOUS PLOTS OF THE SAME LIGHTCURVE*

    returns fig object with the figure
    """
    # glo_offs = gf_plot.glo_offs
    # glo_cols = gf_plot.glo_cols
    # plot_symbols = gf_plot.plot_symbols

    glo_cols, glo_offs, plot_symbols, glo_standard_filters, glo_plot_order, glo_labels, glo_mags_axis = gf_plot.extra_bands(filters)

    N_filters = len(filters)

    fake_phases = -0.99 + 0.01*(np.arange(0,500))
    fake_phases = fake_phases - phase_offset
    if ax==None:
        fig, axs = plt.subplots(N_filters, figsize=(4,2*N_filters))
        
    else:
        fig = ax
        #axs = fig.subplots(N_filters, figsize=(4,2*N_filters))
        axs = fig.subplots(N_filters)
    ax_min = 99.
    ax_max = -99.
    
    #for i in range(len(filters)):
    for key in glo_standard_filters:
        f = key
        #f = filters[i]
        mag_col = 'mag_' + f
        if (mag_col in phot_df) and (f in filters):

            i = filters.index(f)
            if len(phot_df[mag_col].dropna()) > 0:
                err_col = 'err_' + f    
                xx = phot_df['JD']
                yy = phot_df[mag_col]
                yerr = phot_df[err_col]
                xp = gloess[i].phase_jds(xx, period) - phase_offset
                fp = fake_phases[200:300]-1.
                fr = fit_results[i, 200:300]
                
                axs[i].errorbar(np.concatenate((xp, xp+1, xp+2,xp+3)), np.concatenate((yy,yy,yy,yy)), yerr=np.concatenate((yerr,yerr,yerr,yerr)),marker=plot_symbols[f], ls='None', ms=5, color=glo_cols[f])
                axs[i].plot(np.concatenate((fp, fp+1, fp+2,fp+3)), np.concatenate((fr, fr, fr, fr)), ls='-', color='Gray',zorder=10)
                #axs[i].legend(handles[::-1],labels[::-1],bbox_to_anchor=(-0.35, 1), loc='upper left', borderaxespad=0, numpoints=1)
                if np.isnan(np.max(yerr)) or np.isnan(np.min(yerr)):
                    yerr = 0
                    
                band_min = np.concatenate((fit_results[i, 200:300],yy.dropna().values)).min() - np.max(yerr)
                band_max = np.concatenate((fit_results[i, 200:300],yy.dropna().values)).max() + np.max(yerr)
                band_amp = band_max - band_min
                print(band_min, band_max, band_amp, band_max - band_min)
                if yrange==None:
                    if glo_mags_axis[f]==True:
                        axs[i].axis([0,2,band_max+(0.1*band_amp), band_min-(0.1*band_amp)]) 
                    else:
                        axs[i].axis([0,2, band_min-(0.1*band_amp),band_max+(0.1*band_amp)]) 
                else:
                    axs[i].axis([0,2, yrange[0],yrange[1]]) 
                axs[i].set_ylabel(glo_labels[f])
                axs[i].tick_params(direction='in')


                
    if source_id != None:
        plt.suptitle(f'{target}, Gaia DR3 {source_id}, P = {period:.5f} d')
    else:
        plt.suptitle(f'{target}, P = {period:.5f} d')
    #handles, labels = ax1.get_legend_handles_labels() 
    
    #ax1.legend(flip_legend(handles[::-1], 3), flip_legend(labels[::-1],3), mode="expand", borderaxespad=0, ncol=3, bbox_to_anchor=(0, 1.1, 1, 0.2), loc="lower left")

    axs[N_filters-1].set_xlabel('Phase')
    for a in axs[:-1]:
        a.set_xticklabels([])
    #handles, labels = ax2.get_legend_handles_labels() 
    #ax1.legend(handles[::-1],labels[::-1],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., numpoints=1)
    #ax2.legend(handles[::-1],labels[::-1],bbox_to_anchor=(1.3, 1), loc='upper right', borderaxespad=0, numpoints=1)
    #ax2.legend(flip_legend(handles[::-1],3), flip_legend(labels[::-1],3), mode="expand", borderaxespad=0, ncol=3, bbox_to_anchor=(0, 1.1, 1, 0.2), loc="lower left")
    #ax2.axis([0,2,ax_max+0.2, ax_min-0.2])  

    #ax2.set_xlabel('Phase')
    #ax2.set_ylabel('Magnitude')
    #plt.tight_layout()

    fig.subplots_adjust(hspace=0)


    if save_pdf==True:
        plot_file = target.replace(' ', '_') + '_gloess_lc.pdf'
        plt.savefig(plot_file, bbox_inches="tight")
    #return(fig, fit_results[i, 200:300], yy)
    return(fig)

""" Functions to do with data management. Move to separate file? """

def set_up_dataframe_cols(filters, ref_col=False):
    """ set up column names for dataframe in correct order. 
        makes list of names with JD, [mag_X, err_X] for each filter X. 
        Defaults to no reference column but can add one if needed,  
    """
    mags = np.array([])
    for i in range(0, len(filters)):
        mags = np.append(mags, 'mag_' + filters[i])  
    errs = np.array([])
    for i in range(0, len(filters)):
        errs = np.append(errs, 'err_' + filters[i])
    names = np.array(['JD'])
    for i in range(0, len(filters)):
        names = np.append(names, mags[i])
        names = np.append(names, errs[i])
    if ref_col == True:
        names = np.append(names, 'Reference')
    return(names)

def clean_old_gloess_file(df, bad_ref_files = []):
    bad_refs = []
    """
    if no filenames given, use Vicky's default lists
    """
    if bad_ref_files == []:
        bad_ref_files = ['bad_references_list.txt', 'not_irac_references.txt']
    for fn in bad_ref_files:
        file = DATADIR + '/' + fn 
        with open(file) as fn:
            for line in fn:
                line = line.strip()
                bad_refs.append(line)
    df = df[~df['Reference'].isin(bad_refs)]
    return(df)

def old_gloess_to_df(filename, filter_refs=False, input_dir='./'):
    """ 
    takes .gloess_in formatted file as input 
    Assumes the old format with UBVRIJHK (checks for IRAC bands 1 - 4)
    Shouldn't run this on files with non-standard band list and/or order
    filter_refs can filter out data that comes from references listed in bad_references_list.txt
    input_dir defaults to current directory. Can be useful to use environment variables here. 

    returns gloess_input, period, smoothing
    where gloess_input = dataframe with photometry. Includes period and smoothing parameters as attributes.
    period = float
    smoothing = np.array of smoothing params

    """

    """ 
    check if file exists. 
    if it doesn't, check if adding the .gloess_in extention works 
    """
    if input_dir != './':
        filename = input_dir + '/' + filename
    if os.path.isfile(filename)==False:
        filename = filename + '.gloess_in'
        if os.path.isfile(filename)==False:
            print("input file doesn't exist")
            return(1)
    
    """ 
    getting the period and the smoothing paramters from the file 
    also find number of columns to work out if irac data exists
    """
    with open(filename, 'r') as fp:
        period, smoothing, first_line = [x.strip() for ei,x in enumerate(fp) if ei in [1,3,4]]
    try:
        period = float(period)
    except ValueError:
        period = np.nan
        print('No period found in file. Setting Period to np.nan')

    if smoothing[0] == '[':
        smoothing = np.nan
    else:
        smoothing = np.fromstring(smoothing, sep=" ")


    n_cols_old = len(first_line.split())
    
    """ 
    Check if file contains reference column. 
    Will have even number of cols if it does, odd number if it doesn't.   
    """
    if n_cols_old % 2 == 0:
        ref_col = True
    else:
        ref_col = False

    old_gloess_bands = ['U', 'B', 'V', 'R', 'I', 'J','H', 'Ks', 'IRAC1', 'IRAC2', 'IRAC3', 'IRAC4']
    ## for files that haven't had irac data added. 
    if (n_cols_old == 18) or (n_cols_old==17):
        old_gloess_bands = old_gloess_bands[:-4]
    
    """ setting up the dataframe stuff """
    
    filters = np.array(old_gloess_bands)
    names = set_up_dataframe_cols(filters, ref_col)

    """ gloess_input is dataframe with the jds, mags, errs. 
    bad mags replaced with np.nan rather than 99.99
    missing uncertainties replaced with np.nan instead of 9.99
    """
    gloess_input = pd.read_csv(filename, skiprows=4, names=names, sep=r'\s+')
    
    """ fixing the 99 -> np.nans"""
    mag_cols = [c for c in names if 'mag' in c]
    err_cols = [d for d in names if 'err' in d]
    for c in mag_cols:
        gloess_input.replace({c: 99.99}, np.nan, inplace=True)
        gloess_input.replace({c: -99.99}, np.nan, inplace=True)
    for d in err_cols:
        gloess_input.replace({d: 9.99}, np.nan, inplace=True)
        gloess_input.replace({d: 9.99}, np.nan, inplace=True)

    gloess_input.replace({'JD': -99.99}, np.nan, inplace=True)
    gloess_input.dropna(subset='JD', inplace=True)
 
    """
    Can filter out data with ambiguous/unknown references, not on standard filters etc. 
    Just add the references to a text file and give it to clean_old_gloess_file as input. 
    Ones already in the three default files are things that aren't on standard systems, reference couldn't be found etc. 
    """

    if filter_refs==True:
        gloess_input = clean_old_gloess_file(gloess_input, bad_ref_files = [])

    """ 
    returns dataframe to be used later in gloess (gloess_input)
    period and smoothing paramters are also included as an attribute of the dataframe
    Use gloess_input.attrs['period'] and gloess_input.attrs['smoothing'] to access them. 
    UPDATE 14/4/25 - dataframe attributes not working correctly any more. 
    """
    return(gloess_input, period, smoothing)



def save_gloess_input_h5(gloess_input, target_name, period, smoothing, output_dir='./', usename=''):
    if usename !='':
        target_name = usename
    target = target_name
    metadata = gloess_h5_metadata(target, period, smoothing)
    outfile = target_name.replace(' ', '_') + '_gloess_in.h5'
    if output_dir[-1]!='/':
        output_dir = output_dir + '/'
    output = output_dir + outfile
    with pd.HDFStore(output) as hdf_store:
        hdf_store.put('gloess_phot', gloess_input, format='table')
        hdf_store.get_storer('gloess_phot').attrs.metadata = metadata
        hdf_store.close()
    print(f'Saved updated gloess df to {outfile}')
    return(0)

def gloess_h5_metadata(target, period, smoothing):
    last_update = pd.to_datetime('today').strftime('%Y%m%d')
    metadata = {'period': period, 'smoothing': smoothing, 'target': target, "last_update": last_update}
    return(metadata)

def read_gloess_h5(target, input_dir='./', clean=False):
    target = target.replace(' ', '_')
    if input_dir!='./':
        if input_dir[-1]!='/':
            input_dir = input_dir + '/'
    old_gloess_file = input_dir + target + '_gloess_in.h5'

    g_file = Path(old_gloess_file)
    if g_file.is_file():
    
        with pd.HDFStore(old_gloess_file) as hdf_store:
            metadata = hdf_store.get_storer('gloess_phot').attrs.metadata
            og_df = hdf_store.get('gloess_phot')
            hdf_store.close()
        if "period" in metadata:
            period = metadata['period']
        else: 
            period = np.nan
        if "smoothing" in metadata:
            smoothing = metadata['smoothing']
        else: 
            smoothing = np.nan
        if "last_update" in metadata:
            last_update = metadata['last_update']
        else:
            last_update = np.nan
        if clean==True:
            og_df = clean_old_gloess_file(og_df)

        return(og_df, period, smoothing, last_update)
    else:
        return(-1, -1, -1, -1)
    


def add_new_data_to_gloess(target, new_data_file = False, new_data_df=False, gloess_formatted=False, old_file_h5 = False, data_source_note='', time_col=0, 
                                time_format='jd', bands=[], mag_cols=[], err_cols=[], input_dir='./', output_dir='./', usename = '', location=None):

    if time_format=='hjd' and isinstance(location, EarthLocation)==False:
        raise TypeError('You must specify EarthLocation for heliocentric correction')

    # First get the old gloess file 
    if input_dir[-1]!='/':
            input_dir = input_dir + '/'
    if old_file_h5 == False:
        old_gloess_file = input_dir + target
        og_df, period, smoothing = old_gloess_to_df(old_gloess_file, filter_refs=True, input_dir = input_dir)
        
    if old_file_h5 == True:
        og_df, period, smoothing, last_update = read_gloess_h5(target, input_dir=input_dir)

 
    if new_data_file!=False:
        # For data that's already in the correct format 
        if gloess_formatted == True:
            if new_data_file[-3:]=='.h5':
                new_df, period, smoothing, last_update = read_gloess_h5(new_data_file)
            else:
                new_df = pd.read_csv(new_data_file)

        else:
            n_bands = len(bands)
            if n_bands == 0:
                raise Exception('Need more info about this file. Provide jd_col, bands, mag_cols and err_cols.')
            else:
                new_df = pd.read_csv(input_dir + new_data_file)
            if time_format!='jd':
                new_df['correct_JD'] = new_df.apply(lambda x: convert_to_jd(x[time_col], target, time_format, location), axis=1)
                time_col = 'correct_JD'
            new_names = list(set_up_dataframe_cols(bands))
            cols_to_keep =  list(sum(zip(mag_cols, err_cols),()))
            cols_to_keep.insert(0, time_col)
            new_df = new_df[cols_to_keep]
            new_df.rename(columns=dict(zip(cols_to_keep, new_names)), inplace=True)
    
    if type(new_data_df)!=bool:
        new_df = new_data_df
        if time_format!='jd':
                new_df['correct_JD'] = new_df.apply(lambda x: convert_to_jd(x[time_col], target, time_format, location), axis=1)
                time_col = 'correct_JD'
        new_names = list(set_up_dataframe_cols(bands))
        cols_to_keep =  list(sum(zip(mag_cols, err_cols),()))
        cols_to_keep.insert(0, time_col)
        new_df = new_df[cols_to_keep]
        new_df.rename(columns=dict(zip(cols_to_keep, new_names)), inplace=True)

    
    # add info to reference column so you know where the data has come from. 
    # If nothing given here, use the filename. 
    # add a warning in this case to remind user to put a useful comment here in future. 
    if data_source_note == '':
        print(f'Setting Reference column to {new_data_file}. But you should really use a proper reference...')
        data_source_note = new_data_file
    new_df['Reference'] = data_source_note
    
    big_df = pd.concat([og_df, new_df], ignore_index=True, sort=False)

    big_df.drop_duplicates(keep='first', inplace=True)
    
    if usename !='':
        target = usename
    
    metadata = gloess_h5_metadata(target, period, smoothing)
    outfile = target.replace(' ', '_') + '_gloess_in.h5'
    if output_dir[-1]!='/':
        output_dir = output_dir + '/'
    output = output_dir + outfile
    with pd.HDFStore(output) as hdf_store:
        hdf_store.put('gloess_phot', big_df, format='table')
        hdf_store.get_storer('gloess_phot').attrs.metadata = metadata
        hdf_store.close()                
    return(big_df)

def get_gaia_jds(vot_df, jd_col = 'time', filt='all'):
    """ convert times in Gaia votable into JD using offset provided in documentation. 
        Technically these are in TCB, but the difference is very small. 
        When I'm feeling smarter I'll put the correction in (possibly non-trivial - 
        undoing the barycentric correction is not symmetric (I think...))
    """
    if filt!='all':
        """ times are JD-2455197.5"""
        times = vot_df[vot_df.band==filt][jd_col]
        jds = times + 2455197.5
    else:
        times = vot_df[jd_col]
        jds = times + 2455197.5
    return(jds)

def get_gaia_errs(flux_over_error, filt):

    """ Need to update to get the correct zp_errs for DR3"""
    filters = ['G', 'BP', 'RP']
    zp_err = [0.0027553202, 0.0027901700, 0.0037793818]
    errs = np.sqrt((-2.5/(np.log(10)*flux_over_error))**2 + zp_err[filters==filt]**2)
    return(errs)


def read_gaia_epoch_photometry_from_query(source_id):
    """ read in the gaia epoch photometry files from datalink
    and convert to the right type of file for gloess
    """
    """ 
    Gaia datalink products for epoch photometry changed (some time before 10/4/25)
    Updating to use new format
    """
    retrieval_type = 'EPOCH_PHOTOMETRY'          # Options are: 'EPOCH_PHOTOMETRY', 'MCMC_GSPPHOT', 'MCMC_MSC', 'XP_SAMPLED', 'XP_CONTINUOUS', 'RVS', 'ALL'
    data_structure = 'INDIVIDUAL'   # Options are: 'INDIVIDUAL', 'COMBINED', 'RAW'
    data_release   = 'Gaia DR3'     # Options are: 'Gaia DR3' (default), 'Gaia DR2'


    datalink = Gaia.load_data(ids=source_id, data_release = data_release, retrieval_type=retrieval_type, data_structure = data_structure, verbose = False, output_file = None)

    dl_key = 'EPOCH_PHOTOMETRY-Gaia DR3 ' + str(source_id) + '.xml'
    vot_df = datalink[dl_key][0].to_table().to_pandas()

    """
    Obs times for BP and RP are slightly (~30s) offset from G transit times so using their own obs times for each band
    """
    
    g_df = vot_df.loc[:,['g_transit_time', 'g_transit_mag', 'g_transit_flux_over_error']]
    bp_df = vot_df.loc[:,['bp_obs_time', 'bp_mag', 'bp_flux_over_error']]
    rp_df = vot_df.loc[:,['rp_obs_time', 'rp_mag', 'rp_flux_over_error']]
    
    g_df.rename(columns={'g_transit_time': 'Gaia_JD', 'g_transit_mag': 'mag_G', 'g_transit_flux_over_error': 'flux_over_error_G'}, inplace=True)
    bp_df.rename(columns={'bp_obs_time': 'Gaia_JD', 'bp_mag': 'mag_BP', 'bp_flux_over_error': 'flux_over_error_BP'}, inplace=True)
    rp_df.rename(columns={'rp_obs_time': 'Gaia_JD', 'rp_mag': 'mag_RP', 'rp_flux_over_error': 'flux_over_error_RP'}, inplace=True)

    for df in [g_df, bp_df, rp_df]:
        df.dropna(subset='Gaia_JD', inplace=True)

    combined_df = pd.concat([g_df, bp_df, rp_df])

    filters = ['G', 'BP', 'RP']
    ## Converting Gaia times to JD
    combined_df['JD'] = get_gaia_jds(combined_df, jd_col='Gaia_JD')
    for filt in filters:
        err_col = 'err_' + filt
        flux_err_col = 'flux_over_error_' + filt
        combined_df[err_col] = combined_df.apply(lambda x: get_gaia_errs(x[flux_err_col], filt), axis=1)
    #combined_df.attrs['source_id'] = source_id
    combined_df.drop(columns=['Gaia_JD', 'flux_over_error_G', 'flux_over_error_BP', 'flux_over_error_RP'], inplace=True)    
    return(combined_df)

def gloess_csv_to_df(target, new_data_file = False, period=np.nan, gloess_formatted=False, data_source_note='', 
                     time_col=0, time_format='jd', bands=[], mag_cols=[], err_cols=[], input_dir='./', output_dir='./', usename = '', location=None):

    if time_format=='hjd' and isinstance(location, EarthLocation)==False:
        raise TypeError('You must specify EarthLocation for heliocentric correction')

    # First get the old gloess file 
    if input_dir[-1]!='/':
            input_dir = input_dir + '/'
 
    if new_data_file!=False:
        # For data that's already in the correct format 
        if gloess_formatted == True:
            if new_data_file[-3:]=='.h5':
                new_df, period, smoothing, last_update = read_gloess_h5(new_data_file)
            else:
                new_df = pd.read_csv(new_data_file)

        else:
            n_bands = len(bands)
            if n_bands == 0:
                raise Exception('Need more info about this file. Provide time_col, bands, mag_cols and err_cols.')
            else:
                new_df = pd.read_csv(input_dir + new_data_file)
            if time_format!='jd':
                new_df['correct_JD'] = new_df.apply(lambda x: convert_to_jd(x[time_col], target, time_format, location), axis=1)
                time_col = 'correct_JD'
            new_names = list(set_up_dataframe_cols(bands))
            cols_to_keep =  list(sum(zip(mag_cols, err_cols),()))
            cols_to_keep.insert(0, time_col)
            new_df = new_df[cols_to_keep]
            new_df.rename(columns=dict(zip(cols_to_keep, new_names)), inplace=True)

    
    # add info to reference column so you know where the data has come from. 
    # If nothing given here, use the filename. 
    # add a warning in this case to remind user to put a useful comment here in future. 
    if data_source_note == '':
        print(f'Setting Reference column to {new_data_file}. But you should really use a proper reference...')
        data_source_note = new_data_file
    new_df['Reference'] = data_source_note
        
    if usename !='':
        target = usename
    metadata = gloess_h5_metadata(target, period, smoothing)
    outfile = target.replace(' ', '_') + '_gloess_in.h5'
    if output_dir[-1]!='/':
        output_dir = output_dir + '/'
    output = output_dir + outfile
    with pd.HDFStore(output) as hdf_store:
        hdf_store.put('gloess_phot', new_df, format='table')
        hdf_store.get_storer('gloess_phot').attrs.metadata = metadata
        hdf_store.close()                
    return(new_df)



def combine_old_gloess_with_gaia(target, local_file = '', has_h5=False, gloess_dir = '',vartype='cep',period='query', period_col='pf', plot=True, 
                                 save_pdf=True, save_fit=True, alt_period = np.nan, save_df=True, update_gaia=False, show_means=True, input_dir = './', galaxy='MilkyWay'):
    """ combine existing gloess data with gaia
        queries gaia for period
        defaults to cepheids and fundamental mode
        local_file is for e.g wise light curves
    """
    if vartype=='cep':
        vartable = 'vari_cepheid'
    elif vartype=='rrl':
        vartable = 'vari_rrlyrae'
    else:
        print("Options for vartype are 'cep' and 'rrl'. Please provide a valid option")
        return(-1)
    if has_h5==True:
        target = target.replace(' ', '_')
        if input_dir!='./':
            if input_dir[-1]!='/':
                input_dir = input_dir + '/'
        old_gloess_file = input_dir + target + '_gloess_in.h5'
        if os.path.isfile(old_gloess_file):
            og_df, period, smoothing, last_update = read_gloess_h5(target, input_dir)
        else: 
            target_ns  = target.replace(' ', '')
            old_gloess_file = gloess_dir + target_ns + '.gloess_in'
            if os.path.isfile(old_gloess_file)==False:
                old_gloess_file = old_gloess_file + '.gloess_in'
                if os.path.isfile(old_gloess_file)==False:
                    old_gloess_file = '/Users/vs522/Dropbox/All_Cepheids_ever/' + galaxy + '/cepheids/' + str.upper(target)
                    if os.path.isfile(old_gloess_file)==False:
                        print("input file doesn't exist")
                        return(1)
                    else:
                        print('using all cepheids ever data')
    else: 
        target_ns  = target.replace(' ', '')
        old_gloess_file = gloess_dir + target_ns + '.gloess_in'
        if os.path.isfile(old_gloess_file)==False:
            old_gloess_file = old_gloess_file + '.gloess_in'
            if os.path.isfile(old_gloess_file)==False:
                old_gloess_file = '/Users/vs522/Dropbox/All_Cepheids_ever/' + galaxy + '/cepheids/' + str.upper(target)
                if os.path.isfile(old_gloess_file)==False:
                    print("input file doesn't exist")
                    return(1)
                else:
                    print('using all cepheids ever data')
        og_df, period, smoothing = clean_old_gloess_file(old_gloess_file)

    
    target_ns = target.replace(' ', '') ## get rid of spaces for file names
    

    """ get the source_id and the gaia epoch photometry """
    source_id = vs.get_gaia_source_id(target)
    if update_gaia == True:
        gaia_df = read_gaia_epoch_photometry_from_query(source_id)
        big_df = pd.concat([og_df, gaia_df], ignore_index=True, sort=False)
        """ grab the period from the correct gaia variability table """

        query_string = f"select {period_col} from gaiadr3.{vartable} where source_id in ({source_id})"
        job = Gaia.launch_job_async(query_string)
        if len(job.get_results()) == 0:
            if np.isnan(alt_period):
                print(f'{target}: No Gaia period. Please provide an alternative period with the alt_period option')
                return(-1)
            else:
                print(f'{target}: No Gaia period. Using alt_period = {alt_period}')
                period = alt_period
        else:
            period = job.get_results()[period_col][0]
        #big_df.attrs['period'] = period
        #big_df.attrs['last_update'] = pd.to_datetime('today').strftime('%Y%m%d')

    else:
        big_df = og_df
        #big_df.attrs['period'] = og_df.attrs['period']
    """ combine the old gloess file with the gaia data """
    #big_df = pd.concat([og_df, gaia_df, wise_df], ignore_index=True, sort=False)

    #big_df['MJD'] = big_df.apply(lambda x: check_jd_mjd(x.MJD), axis=1)
 
    #big_df.attrs['smoothing'] = smoothing

    """ do the gloess fitting on all the bands """
    filters  = []
    for i in range(len(big_df.columns)):
        if big_df.columns[i][0:4] == 'mag_':
            col_name = big_df.columns[i]
            if len(big_df[col_name].dropna()) > 0:
                filt = big_df.columns[i][4:]
                filters.append(filt)
    print(f'{filters}')
    
    fit_results = np.zeros((len(filters), 500))
    mag_cols = []
    av_mags = np.ones(len(filters)) * np.nan
    int_av_mags = np.ones(len(filters)) * np.nan
    amps = np.ones(len(filters)) * np.nan
    gloess = []
    for i in range(len(filters)):
        band = filters[i]
        mag_col = 'mag_' + band
        mag_cols.append(mag_col)
        #if len(big_df[mag_col].dropna()) > 0:
        print(f'band = {band}, n_data = {len(big_df[mag_col].dropna())}, period={period}')
        err_col = 'err_' + band
        xx = big_df['JD']
        yy = big_df[mag_col]
        yerr = big_df[err_col]
        gloess.append(Gloess(xx, yy, yerr, period, smooth='auto', band=band))
        fit_results[i] = gloess[i].fit_one_band()
    
        # else: 
        #     fit_results[i] = np.ones(500) * np.nan
    gloess = np.array(gloess)

    fake_phases = -0.99 + 0.01*(np.arange(0,500))

    if plot==True:
        
        fig = lightcurve_plotter(target, big_df, gloess, fit_results, filters, fake_phases, period, ax=None, source_id = source_id, save_pdf=save_pdf)


    if save_fit==True:
        #fake_phases = -0.99 + 0.01*(np.arange(0,500))
        df_cols = ['phase'] + mag_cols
        df_fit = pd.DataFrame(columns=df_cols)
        df_fit['phase'] = fake_phases[200:400]-1.0
        for i in range(len(filters)):
            band = filters[i]
            mag_col = 'mag_' + band
            df_fit[mag_col] = fit_results[i, 200:400]
        fit_csv = target.replace(' ', '_') + '_gloess_fit.csv'
        df_fit.to_csv(fit_csv, index=False, float_format='%.4f')
        print(f'Saved gloess fit to {fit_csv}') 
    if save_df==True:
        outfile = target.replace(' ', '_') + '_gloess_in.h5'
        output_dir = '/Users/vs522/Dropbox/All_Cepheids_ever/new_gloess_files/' + galaxy + '/cepheids/'
        output = output_dir + outfile
        metadata = gloess_h5_metadata(target, period, smoothing)

        with pd.HDFStore(output) as hdf_store:
            hdf_store.put('gloess_phot', big_df, format='table')
            hdf_store.get_storer('gloess_phot').attrs.metadata = metadata
            hdf_store.close()
        #new_gloess_fn = target + '_updated_gloess_data.csv'
        #big_df.to_csv(new_gloess_fn, index=False, header=True)
        print(f'Saved updated gloess df to {outfile}')

    return(fit_results, period, big_df, gloess, filters, fig)

def flip_legend(items, ncol):
    return list(itertools.chain(*[items[i::ncol] for i in range(ncol)]))

def read_asassn_lightcurve(asassn_id, lcs):
    """ read in the asassn epoch photometry files from lc table
    and convert to the right type of file for gloess
    """
    """ grab lightcurve for single object and filter on data quality """
    lc_df = lcs[asassn_id].data[lcs[asassn_id].data.quality=='G']
    #piv_df = lc_df[['phot_filter', 'jd', 'mag', 'mag_err']].pivot(index="jd", columns="phot_filter", values=["mag", 'mag_err'])
    
    g_df = lc_df.loc[lc_df.phot_filter=='g',['jd', 'mag', 'mag_err']]
    V_df = lc_df.loc[lc_df.phot_filter=='V',['jd', 'mag', 'mag_err']]
    
    g_df.rename(columns={'jd': 'JD', 'mag': 'mag_g', 'mag_err': 'err_g'}, inplace=True)
    V_df.rename(columns={'jd': 'JD', 'mag': 'mag_V', 'mag_err': 'err_V'}, inplace=True)
    for df in [g_df, V_df]:
        df.dropna(subset='JD', inplace=True)
    combined_df = pd.concat([g_df, V_df])
 
    return(combined_df)
    return(df)


def gaia_gloess_fit_and_plot(source_id, period='query', period_col='pf', plot=True, save_pdf=True, save_fit=True, alt_id = '', vartype='cep', show_means=False, obs_dates=False, start_dates=np.nan, end_dates=np.nan):
    print(f'Fitting Gaia DR3 {source_id}')
    fit_results, gaia_period, df, gloess, = fit_gloess_from_gaia_query(source_id, period, period_col, vartype=vartype)
    if plot==True:
        if obs_dates==True:
            fit_filters = ['BP', 'G', 'RP']
            fn=plot_observing_dates(fit_results, df, gloess, source_id, gaia_period, start_dates, end_dates, fit_filters, plot_bands='gaia', save_pdf=save_pdf, alt_id=alt_id, show_means=show_means)
        else:
            fn = plot_gaia_gloess_fits(fit_results, df, gloess, source_id, gaia_period, save_pdf, alt_id, show_means=show_means)
        print(f'saved lightcurve plot to {fn}')
    if save_fit==True:
        fake_phases = -0.99 + 0.01*(np.arange(0,500))
        df_fit = pd.DataFrame(columns=['phase', 'mag_G', 'mag_BP', 'mag_RP'])
        df_fit['phase'] = fake_phases[200:400]-1.0
        filters = ['BP', 'G', 'RP']
        for i in range(len(filters)):
            band = filters[i]
            mag_col = 'mag_' + band
            df_fit[mag_col] = fit_results[i, 200:400]
        fit_csv = 'GaiaDR3_' + str(source_id) + '_gloess_fit.csv'
        print(fit_csv)
        df_fit.to_csv(fit_csv, index=False, float_format='%.4f')
        print(f'Saved gloess fit to {fit_csv}')
    
    return(fit_results, gaia_period, df, gloess)


# def plot_observing_dates(fit_results, df, gloess, source_id, period, start_dates, end_dates, fit_filters, plot_bands='gaia', save_pdf=True, alt_id='', show_means=False):
#     fake_phases = -0.99 + 0.01*(np.arange(0,500))
#     ax_min = 99.
#     ax_max = -99.
#     fit_filters = np.array(fit_filters)
#     glo_cols = {'U' : 'Violet', 'B' : 'MediumSlateBlue', 'V' : 'DodgerBlue', 'R': 'Turquoise', 
#         'I': 'LawnGreen', 'J': 'Gold', 'H': 'DarkOrange', 'Ks' :'Red', 'IRAC1' : 'MediumVioletRed', 
#         'IRAC2' : 'DeepPink', 'IRAC3' :'HotPink', 'IRAC4' : 'PeachPuff', 'G': 'Green', 'BP': 'Blue',
#         'RP': 'Red', 'W1' : 'MediumVioletRed', 
#         'W2' : 'DeepPink', 'W3' :'HotPink', 'W4' : 'PeachPuff'}
#     glo_offs = {'U' : 3, 'B' : 1.5, 'V' : 1.2, 'R': 0.7, 'I': 0.2, 'J': 0, 'H': -0.4, 'Ks' :-0.8,
#          'IRAC1' : -1.4, 'IRAC2' : -1.8, 'IRAC3' :-2.2, 'IRAC4' : -2.6, 'G': 5, 'BP': 5, 'RP': 5, 
#          'W1' : -1.4, 'W2' : -1.8, 'W3' :-2.2, 'W4' : -2.6}
#     plot_symbols = {'U' : 'o', 'B' : 'o', 'V' : 'o', 'R': 'o', 'I': 'o', 'J': 'o', 'H': 'o', 'Ks' :'o',
#          'IRAC1' : 'o', 'IRAC2' : 'o', 'IRAC3' :'o', 'IRAC4' : 'o', 'G': 'o', 'BP': 'o', 'RP': 'o', 
#          'W1' : 'x', 'W2' : 'x', 'W3' :'x', 'W4' : 'x'}

#     if plot_bands=='gaia':
#         filters = ['BP', 'G', 'RP']
#         #glo_offs = {'BP': 0, 'G':  0, 'RP': 0}
#     elif plot_bands=='all':
#         filters = list(glo_cols.keys())
#     else: 
#         filters = list(plot_bands)
        
#     av_mags = np.ones(len(filters)) * np.nan
#     int_av_mags = np.ones(len(filters)) * np.nan
#     amps = np.ones(len(filters)) * np.nan

#     print(filters)
#     fig = plt.figure(figsize=(8,8))
#     ax = fig.add_subplot(1,1,1)
#     for i in range(len(filters)):
#         band = filters[i]
#         mag_col = 'mag_' + band
#         err_col = 'err_' + band
#         if len(df[mag_col].dropna()) > 0:
#             xx = df['JD']
#             yy = df[mag_col]
#             yerr = df[err_col]
#             fit_col = np.where(fit_filters==band)[0][0]

#             xp = gloess[i].phase_jds(xx, period)
#             ax.plot(np.concatenate((xp, xp+1)), np.concatenate((yy,yy))+glo_offs[band], marker=plot_symbols[band],ls='None', label=band+ ' + ' + str(glo_offs[band]), ms=5, color=glo_cols[band])
#             ax.plot(fake_phases[200:400]-1, fit_results[fit_col, 200:400]+glo_offs[band], 'k-')

#             if show_means==True:
#                 av_mags[i], int_av_mags[i], amps[i] = get_lc_stats(fit_results[fit_col, 200:400])
#                 ax.axhline(av_mags[i], xmin=fake_phases.min(), xmax=fake_phases.max(), color='k', ls='--', label='Mean mag')
#                 ax.axhline(int_av_mags[i], xmin=fake_phases.min(), xmax=fake_phases.max(), color='k', ls='-.', label='Flux averaged mag')
        
#             band_min = fit_results[fit_col, 200:300].min()+glo_offs[band]
#             band_max = fit_results[fit_col, 200:300].max()+glo_offs[band]
        
#             if band_min < ax_min:
#                 ax_min = band_min
#             if band_max > ax_max:
#                 ax_max = band_max
#     if len(alt_id) > 0:
#         ax.set_title(f'{alt_id}, Gaia DR3 {source_id}, P = {period:.5f} d')
#     else:
#         ax.set_title(f'Gaia DR3 {source_id}, P = {period:.5f} d')
#     ax.set_xlabel('Phase')
#     ax.set_ylabel('Magnitude')
    
#     for i in range(len(start_dates)):
        
#         start_time = start_dates[i].to_value('jd')
#         end_time = end_dates[i].to_value('jd')

#         start_phase = (start_time / period) - np.floor(start_time/ period)
#         end_phase = (end_time / period) - np.floor(end_time/ period)
#         ax.axvspan(start_phase, end_phase, alpha=.5, color='gold')
#         ax.axvspan(start_phase+1, end_phase+1, alpha=.5, color='gold')
#         ax.annotate(i+1, xy=(start_phase, ax_min - 0.1), xycoords='data', weight='bold')
#         ax.annotate(i+1, xy=(start_phase+1, ax_min - 0.1), xycoords='data', weight='bold')
#         print(f'start_time[{i}] = {start_dates[i].iso}, start_phase = {start_phase}.')

#     handles, labels = ax.get_legend_handles_labels() 
#     ax.legend(handles[::-1],labels[::-1],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., numpoints=1)
#         #ax1.legend(handles, labels, mode="expand", borderaxespad=0, ncol=6, bbox_to_anchor=(0, 1.1, 1, 0.2), loc="lower left")
#     ax.axis([0,2,ax_max+0.2, ax_min-0.2]) 
#     #handles, labels = ax.get_legend_handles_labels()
#     #ax.legend(handles[::-1], labels[::-1])
#     #ax.axis([0,2,ax_max+0.2, ax_min-0.2])
#     if save_pdf==True:
#         fn = alt_id.replace(' ', '_') + '_gloess_lc.pdf'
#         #fn = 'GaiaDR3_' + str(source_id) + '_gloess_lc.pdf'
#         plt.savefig(fn)
#         plt.close()
#     else:
#         fn=''
#     return(fn)

# def check_jd_mjd(jd, is_gaia=False):
#     if is_gaia==False:
#         if jd > 2400000.5:
#             mjd = jd - 2400000.5
#         else:
#             mjd = jd
#     return(mjd)

# def combine_old_gloess_with_gaia(target, local_file = '', has_h5=False, gloess_dir = '',vartype='cep',period='query', period_col='pf', plot=True, 
#                                  save_pdf=True, save_fit=True, alt_period = np.nan, save_df=True, update_gaia=False, show_means=True, galaxy='MilkyWay'):
#     """ combine existing gloess data with gaia
#         queries gaia for period
#         defaults to cepheids and fundamental mode
#         local_file is for e.g wise light curves
#     """
#     if vartype=='cep':
#         vartable = 'vari_cepheid'
#     elif vartype=='rrl':
#         vartable = 'vari_rrlyrae'
#     else:
#         print("Options for vartype are 'cep' and 'rrl'. Please provide a valid option")
#         return(-1)
#     if has_h5==True:
#         og_df = read_gloess_h5(target, galaxy=galaxy)
#         period = og_df.attrs['period']
#     else: 
#         target_ns  = target.replace(' ', '')
#         old_gloess_file = gloess_dir + target_ns + '.gloess_in'
#         if os.path.isfile(old_gloess_file)==False:
#             old_gloess_file = old_gloess_file + '.gloess_in'
#             if os.path.isfile(old_gloess_file)==False:
#                 old_gloess_file = '/Users/vs522/Dropbox/All_Cepheids_ever/' + galaxy + '/cepheids/' + str.upper(target)
#                 if os.path.isfile(old_gloess_file)==False:
#                     print("input file doesn't exist")
#                     return(1)
#                 else:
#                     print('using all cepheids ever data')
#         og_df, period, smoothing = clean_old_gloess_file(old_gloess_file)

#     if local_file!='':
#         wise_df = read_allwise_lc_from_file(local_file)
#     target_ns = target.replace(' ', '') ## get rid of spaces for file names
    

#     """ get the source_id and the gaia epoch photometry """
#     source_id = vs.get_gaia_source_id(target)
#     if update_gaia == True:
#         gaia_df = read_gaia_epoch_photometry_from_query(source_id)
#         big_df = pd.concat([og_df, gaia_df], ignore_index=True, sort=False)
#         """ grab the period from the correct gaia variability table """

#         query_string = f"select {period_col} from gaiadr3.{vartable} where source_id in ({source_id})"
#         job = Gaia.launch_job_async(query_string)
#         if len(job.get_results()) == 0:
#             if np.isnan(alt_period):
#                 print(f'{target}: No Gaia period. Please provide an alternative period with the alt_period option')
#                 return(-1)
#             else:
#                 print(f'{target}: No Gaia period. Using alt_period = {alt_period}')
#                 period = alt_period
#         else:
#             period = job.get_results()[period_col][0]
#         big_df.attrs['period'] = period
#         big_df.attrs['last_update'] = pd.to_datetime('today').strftime('%Y%m%d')

#     else:
#         big_df = og_df
#         big_df.attrs['period'] = og_df.attrs['period']
#     """ combine the old gloess file with the gaia data """
#     #big_df = pd.concat([og_df, gaia_df, wise_df], ignore_index=True, sort=False)

#     #big_df['MJD'] = big_df.apply(lambda x: check_jd_mjd(x.MJD), axis=1)
 
#     #big_df.attrs['smoothing'] = smoothing

#     """ do the gloess fitting on all the bands """
#     filters  = []
#     for i in range(len(big_df.columns)):
#         if big_df.columns[i][0:4] == 'mag_':
#             filt = big_df.columns[i][4:]
#             filters.append(filt)
#     print(f'{filters}')
    
#     fit_results = np.zeros((len(filters), 500))
#     mag_cols = []
#     av_mags = np.ones(len(filters)) * np.nan
#     int_av_mags = np.ones(len(filters)) * np.nan
#     amps = np.ones(len(filters)) * np.nan
#     for i in range(len(filters)):
#         band = filters[i]
#         mag_col = 'mag_' + band
#         mag_cols.append(mag_col)
#         if len(big_df[mag_col].dropna()) > 0:
#             print(f'band = {band}, n_data = {len(big_df[mag_col].dropna())}, period={period}')
#             err_col = 'err_' + band
#             xx = big_df['JD']
#             yy = big_df[mag_col]
#             yerr = big_df[err_col]
#             gloess = Gloess(xx, yy, yerr, period, smooth='auto')
#             fit_results[i] = gloess.fit_one_band()
#         else: 
#             fit_results[i] = np.ones(500) * np.nan

#     if plot==True:
        
#         fig = plt.figure(figsize=(8,8))
#         ax1 = fig.add_subplot(1,1,1)
#         fake_phases = -0.99 + 0.01*(np.arange(0,500))
#         ax_min = 99.
#         ax_max = -99.
#         glo_cols = {'U' : 'Violet', 'B' : 'MediumSlateBlue', 'V' : 'DodgerBlue', 'R': 'Turquoise', 
#         'I': 'LawnGreen', 'J': 'Gold', 'H': 'DarkOrange', 'Ks' :'Red', 'K': 'Red', 'IRAC1' : 'MediumVioletRed', 
#         'IRAC2' : 'DeepPink', 'IRAC3' :'HotPink', 'IRAC4' : 'PeachPuff', 'G': 'Green', 'BP': 'Blue',
#         'RP': 'Red', 'W1' : 'MediumVioletRed', 
#         'W2' : 'DeepPink', 'W3' :'HotPink', 'W4' : 'PeachPuff'}
#         glo_offs = {'U' : 3, 'B' : 1.5, 'V' : 1.2, 'R': 0.7, 'I': 0.2, 'J': 0, 'H': -0.4, 'Ks' :-0.8, 'K': -0.8,
#          'IRAC1' : -1.4, 'IRAC2' : -1.8, 'IRAC3' :-2.2, 'IRAC4' : -2.6, 'G': 5, 'BP': 5, 'RP':5, 
#          'W1' : -1.4, 'W2' : -1.8, 'W3' :-2.2, 'W4' : -2.6}
#         plot_symbols = {'U' : 'o', 'B' : 'o', 'V' : 'o', 'R': 'o', 'I': 'o', 'J': 'o', 'H': 'o', 'Ks' :'o', 'K': 'x',
#          'IRAC1' : 'o', 'IRAC2' : 'o', 'IRAC3' :'o', 'IRAC4' : 'o', 'G': 'o', 'BP': 'o', 'RP': 'o', 
#          'W1' : 'x', 'W2' : 'x', 'W3' :'x', 'W4' : 'x'}
#         for i in range(len(filters)):
#             f = filters[i]
#             mag_col = 'mag_' + f
#             if len(big_df[mag_col].dropna()) > 0:
#                 err_col = 'err_' + f    
#                 xx = big_df['JD']
#                 yy = big_df[mag_col]
#                 yerr = big_df[err_col]
#                 xp = gloess.phase_jds(xx, period)
#                 ax1.plot(np.concatenate((xp, xp+1)), np.concatenate((yy,yy))+glo_offs[f], marker=plot_symbols[f], ls='None', label=f + ' + ' + str(glo_offs[f]), ms=5, color=glo_cols[f])
#                 ax1.plot(fake_phases[200:400]-1, fit_results[i, 200:400]+glo_offs[f], 'k-')

#                 if show_means==True:
#                     av_mags[i], int_av_mags[i], amps[i] = get_lc_stats(fit_results[i, 200:400])
#                     #ax1.axhline(av_mags[i], xmin=fake_phases.min(), xmax=fake_phases.max(), color='k', ls='--', label='Mean mag')
#                     #ax1.axhline(int_av_mags[i], xmin=fake_phases.min(), xmax=fake_phases.max(), color='k', ls='-.', label='Flux averaged mag')
#                     print(f'{f}: mean = {int_av_mags[i]}')

#                 band_min = fit_results[i, 200:300].min()+glo_offs[f]
#                 band_max = fit_results[i, 200:300].max()+glo_offs[f]

#                 if band_min < ax_min:
#                     ax_min = band_min
#                 if band_max > ax_max:
#                     ax_max = band_max
#         ax1.set_title(f'{target}, Gaia DR3 {source_id}, P = {period:.5f} d')
#         handles, labels = ax1.get_legend_handles_labels() 
#         ax1.legend(handles[::-1],labels[::-1],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., numpoints=1)
#         #ax1.legend(handles, labels, mode="expand", borderaxespad=0, ncol=6, bbox_to_anchor=(0, 1.1, 1, 0.2), loc="lower left")
#         ax1.axis([0,2,ax_max+0.2, ax_min-0.2])  

#         ax1.set_xlabel('Phase')
#         ax1.set_ylabel('Magnitude')

#         if save_pdf==True:
#             plot_file = target.replace(' ', '_') + '_gloess_lc.pdf'
#             plt.savefig(plot_file, bbox_inches="tight")
#     if save_fit==True:
#         fake_phases = -0.99 + 0.01*(np.arange(0,500))
#         df_cols = ['phase'] + mag_cols
#         df_fit = pd.DataFrame(columns=df_cols)
#         df_fit['phase'] = fake_phases[200:400]-1.0
#         for i in range(len(filters)):
#             band = filters[i]
#             mag_col = 'mag_' + band
#             df_fit[mag_col] = fit_results[i, 200:400]
#         fit_csv = target.replace(' ', '_') + '_gloess_fit.csv'
#         df_fit.to_csv(fit_csv, index=False, float_format='%.4f')
#         print(f'Saved gloess fit to {fit_csv}') 
#     if save_df==True:
#         outfile = target.replace(' ', '_') + '_gloess_in.h5'
#         output_dir = '/Users/vs522/Dropbox/All_Cepheids_ever/new_gloess_files/' + galaxy + '/cepheids/'
#         output = output_dir + outfile
#         with pd.HDFStore(output) as hdf_store:
#             hdf_store.put('gloess_phot', big_df, format='table')
#             hdf_store.get_storer('gloess_phot').attrs.metadata = big_df.attrs
#             hdf_store.close()
#         #new_gloess_fn = target + '_updated_gloess_data.csv'
#         #big_df.to_csv(new_gloess_fn, index=False, header=True)
#         print(f'Saved updated gloess df to {outfile}')

#     return(fit_results, period, big_df, gloess, filters, fig)
    
# def read_allwise_lc_from_file(filename):
#     """ read in the gaia epoch photometry files from datalink
#     and convert to the right type of file for gloess
#     """
    
#     table = ascii.read(filename, format='ipac')
#     vot_df = table.to_pandas()

#     """ check that it only contains one star """
#     n_stars = len(vot_df.source_id_mf.unique())
#     if n_stars > 1:
#         raise Exception("data file contains more than one object. ")
#     else:
#         wise_id = vot_df.source_id_mf.unique()[0]

#     cut_df = vot_df[['mjd',
#        'w1mpro_ep', 'w1sigmpro_ep', 'w2mpro_ep', 'w2sigmpro_ep', 'w3mpro_ep', 'w3sigmpro_ep', 'w4mpro_ep',
#        'w4sigmpro_ep']]

#     phot_cols = ['w1mpro_ep', 'w2mpro_ep', 'w3mpro_ep', 'w4mpro_ep'] 
#     err_cols = ['w1sigmpro_ep', 'w2sigmpro_ep', 'w3sigmpro_ep', 'w4sigmpro_ep'] 

#     nice_names = ['W1', 'W2', 'W3', 'W4']
#     """ times are MJD (?? check this)"""
#     names = vs.set_up_dataframe_cols(nice_names)
#     df = pd.DataFrame(columns=names)
#     df['MJD'] = cut_df['mjd']
#     for filt in range(len(nice_names)):
#         mag_col = 'mag_' + nice_names[filt]
#         err_col = 'err_' + nice_names[filt]
#         df[mag_col] = cut_df[phot_cols[filt]]
#         df[err_col] = cut_df[err_cols[filt]]
#     df.reset_index(inplace=True, drop=True)

#     return(df)



# def add_new_data_to_gloess_file(target, new_data_file, gloess_formatted=False, old_file_h5 = False, data_source_note='', time_col=0, 
#                                 time_format='jd', bands=[], mag_cols=[], err_cols=[], galaxy='MilkyWay', usename = '', location=None):

#     if time_format=='hjd' and isinstance(location, EarthLocation)==False:
#         raise TypeError('You must specify EarthLocation for heliocentric correction')

#     # First get the old gloess file 
#     if old_file_h5 == False:
#         allceps_dir = '/Users/vs522/Dropbox/All_Cepheids_ever/' + galaxy + '/cepheids/'
#         old_gloess_file = allceps_dir + target
#         og_df, period, smoothing = clean_old_gloess_file(old_gloess_file)
        
#     if old_file_h5 == True:
#         og_df = read_gloess_h5(target, galaxy='MilkyWay')
#         period = og_df.attrs['period']
#         #smoothing = og_df.attrs['smoothing']
 
#     # For data that's already in the correct format 
#     if gloess_formatted == True:
#         new_df = pd.read_csv(new_data_file)

#     else:
#         n_bands = len(bands)
#         if n_bands == 0:
#             raise Exception('Need more info about this file. Provide jd_col, bands, mag_cols and err_cols.')
#         else:
#             new_df = pd.read_csv(new_data_file)
#         if time_format!='jd':
#             new_df['correct_JD'] = new_df.apply(lambda x: convert_to_jd(x[time_col], target, time_format, location), axis=1)
#             time_col = 'correct_JD'
#         new_names = list(set_up_dataframe_cols(bands))
#         cols_to_keep =  list(sum(zip(mag_cols, err_cols),()))
#         cols_to_keep.insert(0, time_col)
#         new_df = new_df[cols_to_keep]
#         new_df.rename(columns=dict(zip(cols_to_keep, new_names)), inplace=True)
    
#     # add info to reference column so you know where the data has come from. 
#     # If nothing given here, use the filename. 
#     # add a warning in this case to remind user to put a useful comment here in future. 
#     if data_source_note == '':
#         print(f'Setting Reference column to {new_data_file}. But you should really use a proper reference...')
#         data_source_note = new_data_file
#     new_df['Reference'] = data_source_note

#     #return(og_df, new_df)
    
#     big_df = pd.concat([og_df, new_df], ignore_index=True, sort=False)
    
#     big_df.attrs['period'] = period
#     #big_df.attrs['smoothing'] = smoothing
#     big_df.attrs['last_update'] = pd.to_datetime('today').strftime('%Y%m%d')
#     if usename !='':
#         target = usename
#     outfile = target.replace(' ', '_') + '_gloess_in.h5'
#     output_dir = '/Users/vs522/Dropbox/All_Cepheids_ever/new_gloess_files/' + galaxy + '/cepheids/'
#     output = output_dir + outfile
#     with pd.HDFStore(output) as hdf_store:
#         hdf_store.put('gloess_phot', big_df, format='table')
#         hdf_store.get_storer('gloess_phot').attrs.metadata = big_df.attrs
#         hdf_store.close()                
#     return(big_df)

# def convert_to_mjd(time, target=None, input_format='jd', location=None):
#     if input_format!='hjd':
#         if time < 1e6:
#             ## horrible hack for dealing with columns that might have mixed values. fix this!
#             mjd = time
#         else:
#             in_time = Time(time, format=input_format, scale='utc')
#             mjd = in_time.mjd
#     else:
#         in_time = Time(time, format='jd', scale='utc')
#         coords = SkyCoord.from_name(target)
#         ltt_helio = Time(time, format='jd', scale='utc').light_travel_time(coords, 'heliocentric', location=location)
#         mjd = Time(in_time.utc - ltt_helio, format='mjd', scale='utc', location=location).mjd
#     return(mjd)
    
# 
       

# def add_new_data_to_gloess_from_df(target, new_df, old_file_h5 = False, data_source_note='', time_col=0, 
#                                 time_format='jd', galaxy='MilkyWay', usename = '', location=None):

#     if time_format=='hjd' and isinstance(location, EarthLocation)==False:
#         raise TypeError('You must specify EarthLocation for heliocentric correction')

#     # First get the old gloess file 
#     if old_file_h5 == False:
#         allceps_dir = '/Users/vs522/Dropbox/All_Cepheids_ever/' + galaxy + '/cepheids/'
#         old_gloess_file = allceps_dir + target
#         og_df, period, smoothing = clean_old_gloess_file(old_gloess_file)
        
#     if old_file_h5 == True:
#         og_df = read_gloess_h5(target, galaxy='MilkyWay')
#         period = og_df.attrs['period']
#         # smoothing = og_df.attrs['smoothing']
 
#     # For data that's already in the correct format 
#     if time_format!='jd':
#         new_df['correct_JD'] = new_df.apply(lambda x: convert_to_jd(x[time_col], target, time_format, location), axis=1)
#         #new_df.drop(columns=['JD'], inplace=True)
#         new_df.rename(columns={'correct_JD': 'JD'}, inplace=True)
    
#     # add info to reference column so you know where the data has come from. 
#     # If nothing given here, use the filename. 
#     # add a warning in this case to remind user to put a useful comment here in future. 
#     if data_source_note == '':
#         print(f'Setting Reference column to "Added from {new_df}". But you should really use a proper reference...')
#         data_source_note = f"Added from {new_df}"
#     new_df['Reference'] = data_source_note

#     #return(og_df, new_df)
    
#     big_df = pd.concat([og_df, new_df], ignore_index=True, sort=False)
    
#     big_df.attrs['period'] = period
#     #big_df.attrs['smoothing'] = smoothing
#     big_df.attrs['last_update'] = pd.to_datetime('today').strftime('%Y%m%d')
#     if usename !='':
#         target = usename
#     outfile = target.replace(' ', '_') + '_gloess_in.h5'
#     output_dir = '/Users/vs522/Dropbox/All_Cepheids_ever/new_gloess_files/' + galaxy + '/cepheids/'
#     output = output_dir + outfile
#     with pd.HDFStore(output) as hdf_store:
#         hdf_store.put('gloess_phot', big_df, format='table')
#         hdf_store.get_storer('gloess_phot').attrs.metadata = big_df.attrs
#         hdf_store.close()
                
#     return(big_df)

# def rising_branch_mean(fit_results, band_num):
#     mags = fit_results[band_num, 200:300]
#     fake_phases = -0.99 + 0.01*(np.arange(0,500))
#     phases = fake_phases[200:300]
#     fluxes = 10**(-mags / 2.5)
#     mean_flux = np.mean(fluxes)
#     mean_mag = -2.5*np.log10(mean_flux)
#     ## find rising branch
    
#     brightest_idx = np.argmin(mags)
#     faintest_idx = np.argmax(mags)
    
#     ## check direction
    
#     while faintest_idx > brightest_idx:
#         ## shift back by 0.5 in phase
#         phases = fake_phases[150:250]
#         mags = fit_results[band_num, 150:250]
#         brightest_idx = np.argmin(mags)
#         faintest_idx = np.argmax(mags)
    
#     rising_mags = mags[faintest_idx:brightest_idx]
    
#     idx_mean = (np.abs(rising_mags-mean_mag)).argmin()
    
#     mean_rising_phase = phases[idx_mean + faintest_idx] - np.floor(phases[idx_mean + faintest_idx])
    
#     return(mean_rising_phase)
        


# def gloess_fit_plot_df(df, period, target, filters = [], ax=None, time_col='JD', have_phase=False, phase_col='', plot=True, 
#                        save_pdf=True, save_fit=True, alt_id = '', vartype='cep', show_means=False, 
#                        obs_dates=False, start_dates=np.nan, end_dates=np.nan, phase_ref_band=None, phase_ref_TESS=False):
#     if len(filters)==0:
#         mag_cols = [i for i in df.columns if 'mag_' in i]
#         #filters = [sub.replace('mag_', '') for sub in mag_cols]
#         keep_cols = []
#         for i in range(len(mag_cols)):
#             n_data = len(df.dropna(subset=mag_cols[i]))
#             #print(f'{mag_cols[i]} -- {n_data} data points')
#             if n_data >0:
#                 keep_cols.append(mag_cols[i])
#         mag_cols = keep_cols
#         filters = [sub.replace('mag_', '') for sub in keep_cols]
#     else:
#         mag_cols = []
#         for i in filters:
#             mag_cols.append('mag_' + i)
#         keep_cols = []
#         for i in range(len(mag_cols)):
#             n_data = len(df.dropna(subset=mag_cols[i]))
#             #print(f'{mag_cols[i]} -- {n_data} data points')
#             if n_data >0:
#                 keep_cols.append(mag_cols[i])
#         mag_cols = keep_cols

#     fit_results = np.zeros((len(filters), 500))
#     ## previously would only keep last filter's gloess object. Not sure that's what I wanted....
#     ## initialising as a NaN filled array of len(filters)
#     gloess = []
#     fake_phases = -0.99 + 0.01*(np.arange(0,500))
#     for i in range(len(filters)):
#         band = filters[i]
#         print(f'{band}')
#         mag_col = 'mag_' + band
#         err_col = 'err_' + band
#         xx = df['JD']
#         yy = df[mag_col]
#         yerr = df[err_col]
#         gloess.append(Gloess(xx, yy, yerr, period, smooth='auto', band=band))
#         fit_results[i] = gloess[i].fit_one_band()

    
#     gloess = np.array(gloess)

#     av_mags = np.ones(len(filters)) * np.nan
#     int_av_mags = np.ones(len(filters)) * np.nan
#     amps = np.ones(len(filters)) * np.nan
    

#     ref_phase = 0.0

#     if phase_ref_band != None:
#         band_num = np.argwhere(np.array(filters)==phase_ref_band)[0,0]
#         ref_phase = rising_branch_mean(fit_results, band_num)
#         print(f'ref_phase = {ref_phase}')
#         fake_phases = (fake_phases - ref_phase)
#         #print(f'min_fake_phases = {np.min(fake_phases)}')
#         #print(f'max_fake_phases = {np.max(fake_phases)}')
#         if np.min(fake_phases) < -1.5:
#             fake_phases = fake_phases + 1.
#         #print(f'min_fake_phases = {np.min(fake_phases)}')
#         #print(f'max_fake_phases = {np.max(fake_phases)}')
            
#     if phase_ref_TESS ==True:
#         jd_tess_max = df[df.mag_TESS == df.mag_TESS.min()].JD.values[0]
#         ref_phase = phase_jds(jd_tess_max, period)
#         print(f'ref_phase = {ref_phase}')
#         fake_phases = (fake_phases - ref_phase)
#         if np.min(fake_phases) < -1.5:
#             fake_phases = fake_phases + 1.

#     if plot==True:

#         if ax==None:
#             fig = plt.figure(figsize=(8,8))
#             ax1 = fig.add_subplot(1,1,1)
#         else:
#             ax1=ax
#         ax_min = 99.
#         ax_max = -99.
#         glo_cols = {'U' : 'Violet', 'B' : 'MediumSlateBlue', 'V' : 'DodgerBlue', 'R': 'Turquoise', 
#                     'I': 'LawnGreen', 'J': 'Gold', 'H': 'DarkOrange', 'Ks' :'HotPink', 'K': 'HotPink', 'IRAC1' : 'MediumVioletRed', 
#                     'IRAC2' : 'DeepPink', 'IRAC3' :'HotPink', 'IRAC4' : 'PeachPuff', 'G': 'Green', 'BP': 'Blue',
#                     'RP': 'Red', 'W1' : 'MediumVioletRed', 
#                     'W2' : 'DeepPink', 'W3' :'HotPink', 'W4' : 'PeachPuff', 'TESS': 'silver'}
#         #glo_offs = {'U' : 3, 'B' : 1.5, 'V' : 1.2, 'R': 0.7, 'I': 0.2, 'J': 0, 'H': -0.4, 'Ks' :-0.8, 'K': -0.8,
#                     #'IRAC1' : -1.4, 'IRAC2' : -1.8, 'IRAC3' :-2.2, 'IRAC4' : -2.6, 'G': 1, 'BP': 1, 'RP':1, 
#                     #'W1' : -1.4, 'W2' : -1.8, 'W3' :-2.2, 'W4' : -2.6}
#         glo_offs = {'U' : 0, 'B' : 0, 'V' : 0, 'R': 0, 'I': 0, 'J': 0, 'H': 0, 'Ks' :0, 'K': 0,
#                     'IRAC1' : 0, 'IRAC2' : 0, 'IRAC3' :0, 'IRAC4' : 0, 'G': 0, 'BP': 0, 'RP':0, 
#                     'W1' : 0, 'W2' : 0, 'W3' :0, 'W4' : 0, 'TESS': -0.5}
#         plot_symbols = {'U' : 'o', 'B' : 'o', 'V' : 'o', 'R': 'o', 'I': 'o', 'J': 'o', 'H': 'o', 'Ks' :'o', 'K': 'o',
#                         'IRAC1' : 'o', 'IRAC2' : 'o', 'IRAC3' :'o', 'IRAC4' : 'o', 'G': 'o', 'BP': 'o', 'RP': 'o', 
#                         'W1' : 'x', 'W2' : 'x', 'W3' :'x', 'W4' : 'x', 'TESS': 'o'}
        
#         for i in range(len(filters)):
#             f = filters[i]
#             print(f'plotting filter {f}, mag_col {mag_cols[i]} offset {glo_offs[f]}, marker {plot_symbols[f]}, colour {glo_cols[f]}, n_points = {len(df[mag_cols[i]].dropna())}')
#             if len(df[mag_cols[i]].dropna()) > 0:
#                 mag_col = mag_cols[i]
#                 err_col = 'err_' + f    
#                 xx = df['JD']
#                 yy = df[mag_col]
#                 yerr = df[err_col]
#                 gf_phases = np.concatenate((fake_phases[200:300]-2., fake_phases[200:300]-1, fake_phases[200:300], fake_phases[200:300] + 1))
#                 gf_mags = np.concatenate((fit_results[i, 200:300], fit_results[i, 200:300], fit_results[i, 200:300], fit_results[i, 200:300]))
#                 xp = (gloess[i].phase_jds(xx, period) - ref_phase) - np.floor(gloess[i].phase_jds(xx, period) - ref_phase)
#                 ax1.plot(np.concatenate((xp, xp+1, xp+2, xp+3)), np.concatenate((yy,yy,yy,yy))+glo_offs[f], marker=plot_symbols[f], ls='None', label=make_offset_label(f, glo_offs[f]), ms=5, color=glo_cols[f])
#                 ax1.plot(gf_phases, gf_mags+glo_offs[f], 'k-')
#                 if show_means==True:
#                     av_mags[i], int_av_mags[i], amps[i] = get_lc_stats(fit_results[i, 200:400])
#                     print(f'{f}: mean = {int_av_mags[i]}')

#                 band_min = fit_results[i, 200:300].min()+glo_offs[f]
#                 band_max = fit_results[i, 200:300].max()+glo_offs[f]

#                 if band_min < ax_min:
#                     ax_min = band_min
#                 if band_max > ax_max:
#                     ax_max = band_max
#             ax1.set_title(f'{target}, P = {period:.5f} d')
#             #handles, labels = ax1.get_legend_handles_labels() 
#             #ax1.legend(handles[::-1],labels[::-1],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., numpoints=1)
#             #ax1.legend(handles, labels, mode="expand", borderaxespad=0, ncol=6, bbox_to_anchor=(0, 1.1, 1, 0.2), loc="lower left")
#             ax1.axis([0,2.5,ax_max+0.2, ax_min-0.2])  

#             ax1.set_xlabel('Phase')
#             ax1.set_ylabel('Magnitude')
#             if save_pdf==True:
#                 fn = target.replace(' ', '_') + '_gloess_lc.pdf'
#                 plt.savefig(fn, bbox_inches="tight")
        
#     """ if save_fit==True:
#         fake_phases = -0.99 + 0.01*(np.arange(0,500))
#         df_fit = pd.DataFrame(columns=['phase', 'mag_G', 'mag_BP', 'mag_RP'])
#         df_fit['phase'] = fake_phases[200:400]-1.0
#         for i in range(len(filters)):
#             band = filters[i]
#             mag_col = 'mag_' + band
#             df_fit[mag_col] = fit_results[i, 200:400]
#         fit_csv =  target.replace(' ', '_') + '_gloess_fit.csv'
#         print(fit_csv)
#         df_fit.to_csv(fit_csv, index=False, float_format='%.4f')
#         print(f'Saved gloess fit to {fit_csv}') """
    
#     return(fit_results, period, df, gloess, filters, fig)

# def gloess_fit_plot_colours_df(df, period, target, colour_cols, err_cols, time_col='JD', labels=[], plot_colours = [], phase_ref_band=None, ax = None):
    
#     fit_results = np.zeros((len(colour_cols), 500))
#     ## previously would only keep last filter's gloess object. Not sure that's what I wanted....
#     ## initialising as a NaN filled array of len(filters)
#     gloess = []

#     for i in range(len(colour_cols)):
#         print(f'{colour_cols[i]}')
#         col = colour_cols[i]
#         err = err_cols[i]
#         xx = df[time_col]
#         yy = df[col]
#         yerr = df[err]
#         gloess.append(Gloess(xx, yy, yerr, period, smooth='auto', band=col))
#         fit_results[i] = gloess[i].fit_one_band()
#     gloess = np.array(gloess)

#     fake_phases = -0.99 + 0.01*(np.arange(0,500))
#     if phase_ref_band!=None:
#         phase_ref_mag = 'mag_' + phase_ref_band
#         err = 'err_' + phase_ref_band
#         xx = df[time_col]
#         yy = df[phase_ref_mag]
#         yerr = df[err]
#         gloess_phase_ref = (Gloess(xx, yy, yerr, period, smooth='auto', band=col))
#         phase_ref_fit = gloess_phase_ref.fit_one_band()

#         ref_phase_bin = np.argmin(phase_ref_fit[200:400])
#         ref_phase = fake_phases[ref_phase_bin + 200] % 1
#         print(f'{ref_phase} {ref_phase_bin}')
#     else:
#         ref_phase = 0

#     if len(labels) == 0:
#         labels = colour_cols

#     labels = np.array(labels)
    
#     if ax==None:
#         fig = plt.figure(figsize=(8,8))
#         ax1 = fig.add_subplot(1,1,1)
#     else:
#         ax1 = ax
    
#     ax_min = 1.2
#     ax_max = -0.2
    
#     for i in range(len(colour_cols)):
#         print(len(labels))
#         l = labels[i]
#         if len(df[colour_cols[i]].dropna()) > 0:
#             col = colour_cols[i]
#             err = err_cols[i]
#             xx = df[time_col]
#             yy = df[col]
#             yerr = df[err]
#             xp = gloess[i].phase_jds(xx, period)
#             ax1.errorbar(np.concatenate((xp, xp+1, xp+2, xp+3)) - ref_phase, np.concatenate((yy,yy, yy, yy)), yerr=np.concatenate((yerr, yerr, yerr, yerr)),ls='None', marker='o', ms=5, label=l, color=plot_colours[i])

#             #ax1.errorbar(np.concatenate((xp, xp+1)), np.concatenate((yy,yy)), yerr=np.concatenate((yerr, yerr)),ls='None', marker='o', label=str(labels[i]), ms=5)
#             ax1.plot(fake_phases[100:450]- ref_phase, fit_results[i, 100:450], 'k-')

#             band_min = fit_results[i, 200:300].min()
#             band_max = fit_results[i, 200:300].max()

#             if band_min < ax_min:
#                 ax_min = band_min
#             if band_max > ax_max:
#                 ax_max = band_max
#     ax1.set_title(f'{target}, P = {period:.5f} d')
#     handles, labels = ax1.get_legend_handles_labels() 
#     ax1.legend(handles[::-1],labels[::-1], numpoints=1)
#     #ax1.legend(handles[::-1],labels[::-1],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., numpoints=1)
#         #ax1.legend(handles, labels, mode="expand", borderaxespad=0, ncol=6, bbox_to_anchor=(0, 1.1, 1, 0.2), loc="lower left")
#     #ax1.axis([0,2.5,ax_max+0.2, ax_min-0.2])  
#     ax1.axis([0,2.5,1.0,-0.2])
#     ax1.set_xlabel('Phase')
#     ax1.set_ylabel('Colour')
        
    
#     return(fit_results, period, df, gloess)

# def make_offset_label(band, offset):
#     if offset==0:
#         label = f'{band}'
#     elif offset < 0:
#         label = f'{band} $-$ {np.abs(offset)}'
#     else:
#         label = f'{band} + {offset}'
#     return(label)




