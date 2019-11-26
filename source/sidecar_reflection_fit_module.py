import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from uncertainties import umath
from uncertainties import unumpy
from uncertainties import ufloat

def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def lin_to_dB(y):
    return 10*np.log10(y)

def guess_fo(f, gamma2):
    ind_fo = np.argmin(gamma2) #find index of resonant frequency
    return f[ind_fo]

def guess_offset(y):
    low_filt_perc = 0.33
    y_filtered = stats.trim1(y, low_filt_perc, tail = 'left') #cut out bottom low_filt_perc of y values. Basically want to filter out the notch a bit.
    return np.median(y_filtered)

def guess_dy(y):
    return guess_offset(y) - np.min(y) 

def guess_q(f, y):
    ind_fc = np.argmin(y) #find index of resonant frequency
    fc = f[ind_fc] #obtain resonant frequency

    #look at the left of the resonance
    left_f = f[:ind_fc] 
    left_y = y[:ind_fc] 
    dy = guess_dy(y)
    ind_fwhm = find_nearest_idx(left_y, dy/2)

    #find distance between fwhm and resonance
    f1 = f[ind_fwhm]
    #guess bandwidth as twice that distance
    del_f = 2*(fc-f1)
    Q_guess = fc/del_f
    return Q_guess


def guess_reflection_fit_params(f, gamma2):
    fo_guess = guess_fo(f, gamma2)
    Q_guess = guess_q(f, gamma2)
    dy_guess = guess_dy(gamma2)
    C_guess = guess_offset(gamma2)
    return fo_guess, Q_guess, dy_guess, C_guess

 
def func_pow_reflected(f, fo, Q, del_y, C):
    return -(fo/(2*Q))**2*del_y/((f-fo)**2+(fo/(2*Q))**2)+C

def plot_mag_phase( x, ymag, yphase):
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Frequency (MHz)')
    ax1.set_ylabel(r'$|\Gamma|$', color=color)
    ax1.plot(x, ymag, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel(r'$\angle \Gamma$ ', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, yphase, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    

def get_arr_ends(x, n_end_elements):
    return np.concatenate([x[:n_end_elements],x[-n_end_elements:]])

def deconvolve_transmission(f, gamma_mag, gamma_phase, C_fit):
    gamma_cav_mag = gamma_mag*np.sqrt(1/C_fit)

    interp_phase = interp1d(f, gamma_phase, kind='cubic')
    f_ends = get_arr_ends(f, 5)
    phase_ends = get_arr_ends(gamma_phase, 5)
#    interp_phase_wo_notch = interp1d(f_ends, phase_ends, kind='linear')
    interp_phase_wo_notch = np.poly1d(np.polyfit(f_ends, phase_ends, 1))
    delay_phase = interp_phase_wo_notch(f)
    gamma_cav_phase = interp_phase(f) - delay_phase

    return gamma_cav_mag, gamma_cav_phase, delay_phase

#    interp_mag = interp1d(f, deconvolved_mag, kind='cubic')
#    interp_sig_mag = interp1d(f, deconvolved_sig_mag, kind='cubic')
#    interp_phase = interp1d(f, deconvolved_phase, kind='cubic')
#

def calculate_coupling(gamma_mag_fo, gamma_phase_fo):
    beta = (1+np.sign(gamma_phase_fo - np.pi)*np.abs(gamma_mag_fo))/(1-np.sign(gamma_phase_fo - np.pi)*np.abs(gamma_mag_fo))
    return beta

def estimate_power_uncertainty(power):
    """Assume fractional uncertainty in power is constant. This is not a correct assumption, but probably good enough for now."""
    pow_fractional_std = np.std(get_arr_ends(power, 5))/np.mean(get_arr_ends(power, 5))
    sig_power = power*pow_fractional_std
    return sig_power

def calc_red_chisq(x, y, sigma_y, func, fit_param):
    resid = y-func(x, *fit_param)
    chisq = np.sum(resid/sigma_y)
    dof = len(y)-len(fit_param)
    red_chisq = chisq/dof
    return red_chisq

def fit_shape_database_hack(x, func, fit_param):
    #This takes the sidecar fit, which is in power, and returns it into the fit shape that 
    gamma_mag = np.sqrt(func(x, *fit_param))
    gamma_dummy = np.zeros_like(ymag)
    fit_shape = np.concatenate(gamma_mag, gamma_dummy)
    return fit_shape

