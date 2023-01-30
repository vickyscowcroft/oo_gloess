import numpy as np
from numpy.linalg import inv
import os.path 
import pandas as pd

import astropy.io.votable as vot

from astroquery.gaia import Gaia
import matplotlib.pyplot as plt


np.seterr(divide='ignore')
np.seterr(over='ignore')

class Gloess(object):
    """ Gloess class
    initialise as gloess = Gloess(xx, yy, yerr, period, smooth='auto')
    where xx = mjds, yy = mags, yerr = photometric errors, period=period of star
    smooth = smoothing parameter for that band. if 'auto' then gloess finds the best smoothing paramters
    degree = 2 is the order of the fit. not implemented yet. 
    """
    @staticmethod
    def phase_mjds(jds, period):
        """ phases the JDs according to the period """
        phase = (jds / period) - np.floor(jds/ period)
        return(phase)
        
    def __init__(self, mjds, mags, errs, period, smooth='auto', degree=2):
        """ initialising the Gloess object
        sets up the phase, y, yerr arrays to repeat over 5 cycles
        finds the smoothing parameter if not fixed to user value
        gets the phase matrix
        """

        y_temp = mags[~mags.isna()]
        
        if len(y_temp) == 0:
            raise Exception('No data in this band.')
        self.y = np.concatenate((y_temp, y_temp, y_temp, y_temp, y_temp))

        phase_temp = self.phase_mjds(mjds[~mags.isna()], period)     
        self.phase = np.concatenate((phase_temp, phase_temp+1., phase_temp+2., phase_temp+3., phase_temp+4.))

        err_temp = errs[~mags.isna()]
        self.yerr = np.concatenate((err_temp, err_temp,err_temp,err_temp,err_temp))

        self.degree = degree

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

    def fit2(self, x1, wt): #x1,y1,n,wt):
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

""" Below here are convenience functions for reading data, plotting, saving output etc. """
def phase_mjds(jds, period):
        """ phases the JDs according to the period """
        phase = (jds / period) - np.floor(jds/ period)
        return(phase)

def old_gloess_to_df(filename, ref_col=False):
    """ takes .gloess_in formatted file as input 
    returns gloess_input, period, smoothing
    where gloess_input = dataframe
    period = float
    smoothing = np.array of smoothing params
    """
    """ check if file exists. if it doesn't, check if adding the .gloess_in extention works """
    if os.path.isfile(filename)==False:
        filename = filename + '.gloess_in'
        if os.path.isfile(filename)==False:
            print("input file doesn't exist")
            return(1)
    
    """ setting up the dataframe stuff """
    old_gloess_bands = ['U', 'B', 'V', 'R', 'I', 'J','H', 'Ks', 'IRAC1', 'IRAC2', 'IRAC3', 'IRAC4']
    filters = np.array(old_gloess_bands)
    n_cols = int(len(filters)*2.)
    print(ref_col)
    if ref_col==False:
        """ have to do this part because some old gloess files have a column with the reference, some don't
        might implement reference filtering later
        """
        cols_list = (np.linspace(0,n_cols, n_cols+1)).astype(int)
        names = set_up_dataframe_cols(filters)

    if ref_col==True:
        cols_list = (np.linspace(0,n_cols+1, n_cols+2)).astype(int)
        names = set_up_dataframe_cols(filters)
        names = np.append(names, 'Reference')

    """ gloess_input is dataframe with the mjds, mags, errs. 
    bad mags replaced with np.nan rather than 99.99"""
    gloess_input = pd.read_csv(filename, skiprows=4, names=names, delim_whitespace=True, usecols=cols_list)
    """ getting the period and the smoothing paramters from the file """
    with open(filename, 'r') as fp:
        period, smoothing = [x.strip() for ei,x in enumerate(fp) if ei in [1,3]]
    period = float(period)
    smoothing = np.fromstring(smoothing, sep=" ")
    """ fixing the 99 -> np.nans"""
    mag_cols = [c for c in names if 'mag' in c]
    for c in mag_cols:
        gloess_input[c].replace(99.99, np.NaN, inplace=True)
    
    return(gloess_input, period, smoothing)

def read_gaia_epoch_photometry_from_file(filename):
    """ read in the gaia epoch photometry files from datalink
    and convert to the right type of file for gloess
    """

    vot_df = vot.parse_single_table(filename).to_table().to_pandas()
    if vot_df.source_id.nunique() > 1:
        print('more than one source_id in this file.')
        return(1)
    vot_df.dropna(subset=['time'], inplace=True)
    piv_df = vot_df[['band', 'time', 'mag', 'flux_over_error', 'source_id']].pivot(index="time", columns="band", values=["mag", 'flux_over_error', 'source_id'])
    """ check it's just a single object """
    
    filters = vot_df.band.unique()
    """ times are JD-2455197.5"""
    names = set_up_dataframe_cols(filters)
    names = np.append(names, 'source_id')
    df = pd.DataFrame(columns=names, index=vot_df.time.dropna().values)
    df['Gaia_JD'] = df.index.copy()
    df['MJD'] = get_gaia_jds(df, jd_col='Gaia_JD')
    for filt in filters:
        mag_col = 'mag_' + filt
        err_col = 'err_' + filt
        df[mag_col] = piv_df[('mag', filt)]
        df[err_col] = piv_df.apply(lambda x: get_gaia_errs(x[('flux_over_error', filt)], filt), axis=1)
    df['source_id'] = vot_df['source_id'][0]
    df.reset_index(inplace=True, drop=True)

    return(df)


def set_up_dataframe_cols(filters):
    mags = np.array([])
    for i in range(0, len(filters)):
        mags = np.append(mags, 'mag_' + filters[i])  
    errs = np.array([])
    for i in range(0, len(filters)):
        errs = np.append(errs, 'err_' + filters[i])
    names = np.array(['MJD'])
    for i in range(0, len(filters)):
        names = np.append(names, mags[i])
        names = np.append(names, errs[i])
    return(names)

def get_gaia_errs(flux_over_error, filt):

    """ Need to update to get the correct zp_errs for DR3"""
    filters = ['G', 'BP', 'RP']
    zp_err = [0.0027553202, 0.0027901700, 0.0037793818]
    errs = np.sqrt((-2.5/(np.log(10)*flux_over_error))**2 + zp_err[filters==filt]**2)
    return(errs)

def get_gaia_jds(vot_df, jd_col = 'time', filt='all'):
    if filt!='all':
        """ times are JD-2455197.5"""
        times = vot_df[vot_df.band==filt][jd_col]
        jds = times + 2455197.5
    else:
        times = vot_df[jd_col]
        jds = times + 2455197.5
    return(jds)

def read_gaia_epoch_photometry_from_query(source_id):
    """ read in the gaia epoch photometry files from datalink
    and convert to the right type of file for gloess
    """
    retrieval_type = 'EPOCH_PHOTOMETRY'          # Options are: 'EPOCH_PHOTOMETRY', 'MCMC_GSPPHOT', 'MCMC_MSC', 'XP_SAMPLED', 'XP_CONTINUOUS', 'RVS', 'ALL'
    data_structure = 'INDIVIDUAL'   # Options are: 'INDIVIDUAL', 'COMBINED', 'RAW'
    data_release   = 'Gaia DR3'     # Options are: 'Gaia DR3' (default), 'Gaia DR2'


    datalink = Gaia.load_data(ids=source_id, data_release = data_release, retrieval_type=retrieval_type, data_structure = data_structure, verbose = False, output_file = None)

    dl_key = 'EPOCH_PHOTOMETRY-Gaia DR3 ' + str(source_id) + '.xml'
    vot_df = datalink[dl_key][0].to_table().to_pandas()
    #vot_df = vot.parse_single_table(filename).to_table().to_pandas()
    if vot_df.source_id.nunique() > 1:
        print('more than one source_id in this file.')
        return(1)
    vot_df.dropna(subset='time', inplace=True)
    piv_df = vot_df[['band', 'time', 'mag', 'flux_over_error', 'source_id']].pivot(index="time", columns="band", values=["mag", 'flux_over_error', 'source_id'])
    """ check it's just a single object """
    
    filters = vot_df.band.unique()
    """ times are JD-2455197.5"""
    names = set_up_dataframe_cols(filters)
    names = np.append(names, 'source_id')
    df = pd.DataFrame(columns=names, index=vot_df.time.dropna().values)
    df['Gaia_JD'] = df.index.copy()
    df['MJD'] = get_gaia_jds(df, jd_col='Gaia_JD')
    for filt in filters:
        mag_col = 'mag_' + filt
        err_col = 'err_' + filt
        df[mag_col] = piv_df[('mag', filt)]
        df[err_col] = piv_df.apply(lambda x: get_gaia_errs(x[('flux_over_error', filt)], filt), axis=1)
    df['source_id'] = vot_df['source_id'][0]
    df.reset_index(inplace=True, drop=True)

    return(df)

def fit_gloess_from_gaia_query(source_id, period='query', period_col='pf', vartype='cep'):
    df = read_gaia_epoch_photometry_from_query(source_id)
    if vartype=='cep':
        query = f"select source_id, {period_col} from gaiadr3.vari_cepheid where source_id in ({source_id})"
    elif vartype=='rrl':
        query = f"select source_id, {period_col} from gaiadr3.vari_rrlyrae where source_id in ({source_id})"

    job     = Gaia.launch_job_async(query)
    if period=='query':
        per = job.get_results()[period_col][0]
    else:
        per = period
    filters = ['BP', 'G', 'RP']
    fit_results = np.zeros((len(filters), 500))

    for i in range(len(filters)):
        band = filters[i]
        mag_col = 'mag_' + band
        err_col = 'err_' + band
        xx = df['MJD']
        yy = df[mag_col]
        yerr = df[err_col]
        gloess = Gloess(xx, yy, yerr, per, smooth='auto', degree=2)
        fit_results[i] = gloess.fit_one_band()

    return(fit_results, per, df, gloess)

def plot_gaia_gloess_fits(fit_results, df, gloess, source_id, period, save_pdf=True, alt_id='', show_means=False):
    fake_phases = -0.99 + 0.01*(np.arange(0,500))
    ax_min = 99.
    ax_max = -99.

    colours = ['Blue', 'Green', 'Red']
    filters = ['BP', 'G', 'RP']
    av_mags = np.ones(len(filters)) * np.nan
    int_av_mags = np.ones(len(filters)) * np.nan
    amps = np.ones(len(filters)) * np.nan


    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)
    for i in range(len(filters)):
        band = filters[i]
        mag_col = 'mag_' + band
        err_col = 'err_' + band
        xx = df['MJD']
        yy = df[mag_col]
        yerr = df[err_col]

        xp = gloess.phase_mjds(xx, period)
        ax.errorbar(np.concatenate((xp, xp+1)), np.concatenate((yy,yy)),yerr=np.concatenate((yerr,yerr)), marker='o',ls='None', label=band, ms=5, color=colours[i])
        ax.plot(fake_phases[200:400]-1, fit_results[i, 200:400], 'k-')

        if show_means==True:
            av_mags[i], int_av_mags[i], amps[i] = get_lc_stats(fit_results[i, 200:400])
            ax.axhline(av_mags[i], xmin=fake_phases.min(), xmax=fake_phases.max(), color='k', ls='--', label='Mean mag')
            ax.axhline(int_av_mags[i], xmin=fake_phases.min(), xmax=fake_phases.max(), color='k', ls='-.', label='Flux averaged mag')
    
        band_min = fit_results[i, 200:300].min()
        band_max = fit_results[i, 200:300].max()
    
        if band_min < ax_min:
            ax_min = band_min
        if band_max > ax_max:
            ax_max = band_max
    if len(alt_id) > 0:
        ax.set_title(f'{alt_id}, Gaia DR3 {source_id}, P = {period:.5f} d')
    else:
        ax.set_title(f'Gaia DR3 {source_id}, P = {period:.5f} d')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Magnitude')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    ax.axis([0,2,ax_max+0.2, ax_min-0.2])
    if save_pdf==True:
        fn = 'GaiaDR3' + str(source_id) + '_gloess_lc.pdf'
        plt.savefig(fn)
        plt.close()
    return(fn)

def gaia_gloess_fit_and_plot(source_id, period='query', period_col='pf', plot=True, save_pdf=True, save_fit=True, alt_id = '', vartype='cep', show_means=False, obs_dates=False, start_dates=np.NaN, end_dates=np.NaN):
    print(f'Fitting Gaia DR3 {source_id}')
    fit_results, gaia_period, df, gloess, = fit_gloess_from_gaia_query(source_id, period, period_col, vartype=vartype)
    if plot==True:
        if obs_dates==True:
            fn=plot_observing_dates(fit_results, df, gloess, source_id, period, start_dates, end_dates, save_pdf=True, alt_id='', show_means=False)
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
    
    return(fit_results, period, df, gloess)

def clean_old_gloess_file(filename):
    df, period, smoothing = gf.old_gloess_to_df(filename, ref_col=True)
    bad_refs = []
    bad_ref_files = ['bad_references_list.txt', 'johnson_system_references.txt', 'not_irac_references.txt']
    for file in bad_ref_files:
        with open('/Users/vs522/Dropbox/Python/oo_gloess/'+file) as fn:
            for line in fn:
                line = line.strip()
                bad_refs.append(line)
    df = df[~df['Reference'].isin(bad_refs)]
    df['MJD'].replace(-99.99, np.NaN, inplace=True)
    df.dropna(subset=['MJD'], inplace=True)
    return(df, period, smoothing)

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

def plot_observing_dates(fit_results, df, gloess, source_id, period, start_dates, end_dates, save_pdf=True, alt_id='', show_means=False):
    fake_phases = -0.99 + 0.01*(np.arange(0,500))
    ax_min = 99.
    ax_max = -99.

    colours = ['Blue', 'Green', 'Red']
    filters = ['BP', 'G', 'RP']
    av_mags = np.ones(len(filters)) * np.nan
    int_av_mags = np.ones(len(filters)) * np.nan
    amps = np.ones(len(filters)) * np.nan


    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)
    for i in range(len(filters)):
        band = filters[i]
        mag_col = 'mag_' + band
        err_col = 'err_' + band
        xx = df['MJD']
        yy = df[mag_col]
        yerr = df[err_col]

        xp = gloess.phase_mjds(xx, period)
        ax.errorbar(np.concatenate((xp, xp+1)), np.concatenate((yy,yy)),yerr=np.concatenate((yerr,yerr)), marker='o',ls='None', label=band, ms=5, color=colours[i])
        ax.plot(fake_phases[200:400]-1, fit_results[i, 200:400], 'k-')

        if show_means==True:
            av_mags[i], int_av_mags[i], amps[i] = get_lc_stats(fit_results[i, 200:400])
            ax.axhline(av_mags[i], xmin=fake_phases.min(), xmax=fake_phases.max(), color='k', ls='--', label='Mean mag')
            ax.axhline(int_av_mags[i], xmin=fake_phases.min(), xmax=fake_phases.max(), color='k', ls='-.', label='Flux averaged mag')
    
        band_min = fit_results[i, 200:300].min()
        band_max = fit_results[i, 200:300].max()
    
        if band_min < ax_min:
            ax_min = band_min
        if band_max > ax_max:
            ax_max = band_max
    if len(alt_id) > 0:
        ax.set_title(f'{alt_id}, Gaia DR3 {source_id}, P = {period:.5f} d')
    else:
        ax.set_title(f'Gaia DR3 {source_id}, P = {period:.5f} d')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Magnitude')
    
    gaia_jd0 = 2455197.5

    for i in range(len(start_dates)):
        start_time = start_dates[i].to_value('jd') - gaia_jd0
        end_time = end_dates[i].to_value('jd') - gaia_jd0

        start_phase = (start_time / period) - np.floor(start_time/ period)
        end_phase = (end_time / period) - np.floor(end_time/ period)
        ax.axvspan(start_phase, end_phase, alpha=.5, color='blue')


    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    ax.axis([0,2,ax_max+0.2, ax_min-0.2])
    if save_pdf==True:
        fn = 'GaiaDR3' + str(source_id) + '_gloess_lc.pdf'
        plt.savefig(fn)
        plt.close()
    return(fn)



 