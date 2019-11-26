import math
import sys
import yaml
import numpy as np
import cmath
import json
from scipy.optimize import least_squares

from dripline.core import calibrate
from dripline.core import Spime
import dripline
import dragonfly
from scipy.interpolate import interp1d

#import functions for sidecar fitting.
from sidecar_reflection_fit_module import estimate_power_uncertainty
from sidecar_reflection_fit_module import guess_reflection_fit_params
from sidecar_reflection_fit_module import func_pow_reflected
from sidecar_reflection_fit_module import deconvolve_transmission
from sidecar_reflection_fit_module import calculate_coupling
from sidecar_reflection_fit_module import calc_red_chisq
from sidecar_reflection_fit_module import fit_shape_database_hack


import logging
logger = logging.getLogger('dragonfly.implementations.custom')

_all_calibrations = []


def iq_packed2powers(iq_data):
    """Turn iq data in [r,i,r,i,r,i...] format into an array of powers"""
    powers=np.zeros(int(len(iq_data)/2))
    for i in range(int(len(iq_data)/2)):
        powers[i]=iq_data[2*i]*iq_data[2*i]+iq_data[2*i+1]*iq_data[2*i+1]
    return powers

def unpack_iq_data(iq_data):
    """takes iq data in [r,i,r,i,r,i] format and unpacks into two arrays [r,r,r],[i,i,i]"""
    ret_r=np.zeros(len(iq_data)//2)
    ret_i=np.zeros(len(iq_data)//2)
    for i in range(0, len(iq_data),2):
        ret_r[i//2]=iq_data[i]
        ret_i[i//2]=iq_data[i+1]
    return ret_r,ret_i

def repack_iq_data(data_r, data_i):
    """takes iq data seperated into 2 arrays [r,r,r], [i,i,i] and combines them into one array [r,i,r,i,r,i]. This method was made to test certain functions that takes iq data with data that had already been unpacked."""
    if not (data_r.size == data_i.size):
        raise ValueError("Real and imaginary vectors should be the same length")
    iq_series = np.empty((data_r.size + data_i.size,), dtype=data_r.dtype)
    iq_series[0::2] = data_r
    iq_series[1::2] = data_i
    return iq_series

def transmission_power_shape(f,norm,f0,Q,noise):
    """returns the expected power from a transmission measurement at frequency f with parameters
        f0 - resonant frequency
        Q - resonant quality
        norm - normalization of the resonance
        noise - estimate of the noise background"""
    delta=Q*(f-f0)/f0
    return norm*(1/(1+4*delta*delta)+noise)

def reflection_iq_shape(f,norm,phase,f0,Q,beta,delay_time):
    """returns the expected [i,q] values from a reflection measurement
        """
#beta=2*atan(beta)/3.14159+1.0 only if we have trouble keeping beta within bounds
    delta=Q*(f-f0)/f0
    denom=1/(1+4*delta*delta)
    response=norm*complex(denom*((beta-1)-4*delta*delta),-denom*2*beta*delta)
    phase=cmath.exp(complex(0,phase+delay_time*(f-f0)))
    return response*phase


def fit_transmission(powers,frequencies):
    """
        Performs a least-squares fit on a transmission measurement, an array of powers and frequencies
        ASSUMPTIONS: (these go in as priors)
        - center frequency is within band
        - band is 1-10 
        - noise is the mean value of outer 10% of band
        - uncertainty is standard devation of outer 10% of band
    """
    if len(frequencies)!=len(powers):
        raise ValueError("point count not right nfreqs {} npows {}".format(len(frequencies),len(powers)))
    if len(frequencies)<16:
        raise ValueError("not enough points to fit transmission, need 16, got {}".format(len(powers)))

    f0_guess=frequencies[int(math.floor(len(frequencies)/2))]
    f_band=frequencies[-1]-frequencies[0]
    norm_guess=max(powers)
    Q_min=f0_guess/f_band
    Q_max=20*Q_min
    Q_guess=0.5*(Q_max+Q_min)
    ten_percent_mark=int(math.ceil(0.1*len(frequencies)))
    noise_guess=0.5*(np.mean(powers[0:ten_percent_mark]+np.mean(powers[len(powers)-ten_percent_mark:len(powers)])))/norm_guess
    uncertainty=0.5*(np.std(powers[0:ten_percent_mark]+np.std(powers[len(powers)-ten_percent_mark:len(powers)])))
    norm_guess=max(powers)-noise_guess
    p0=[norm_guess,f0_guess,Q_guess,noise_guess]
    def fit_fcn(x):
        #calculate the residuals of the fit as an array
        nfreq=len(frequencies)
        npriors=3
        norm=x[0]
        f0=x[1]
        Q=x[2]
        noise=x[3]
        resid=np.zeros(nfreq+npriors)
        #add priors
        #Prior 1: frequency must be within bounds
        if f0<frequencies[0]:
            resid[nfreq]=(f0-frequencies[0])/f0_guess
            f0=frequencies[0]
        if f0>frequencies[-1]:
            resid[nfreq]=(frequencies[-1]-f0)/f0_guess
            f0=frequencies[-1]
        #Prior 2: Q must be neither too small nor too large
        if Q<Q_min:
            resid[nfreq+1]=10*nfreq*(Q-Q_min)/Q_min
            Q=Q_min
        if Q>Q_max:
            resid[nfreq+1]=10*nfreq*(Q_max-Q)/Q_min
            Q=Q_max
        #Prior 3: noise level not too big
        if noise<0:
            resid[nfreq+2]=-10*nfreq*noise
            noise=0
        if noise>0.1:
            resid[nfreq+2]=(noise-0.1)
            #note I do not clamp noise, it is possible it gets this high, jsut undesirable
        for i in range(nfreq):
            yp=transmission_power_shape(frequencies[i],norm,f0,Q,noise)
            resid[i]=(yp-powers[i])/uncertainty
        return resid
    #actual fit done here
    res=least_squares(fit_fcn,p0,xtol=1e-12) #things like df/f are super small, so set xtol extra low
    chisq=res.cost/len(powers)
    #contsruct the fit shape
    fit_shape=[ transmission_power_shape(f,res.x[0],res.x[1],res.x[2],res.x[3]) for f in frequencies ]
    #return norm,f0,Q,noise, chi square, fit shape
    return [res.x[0],res.x[1],res.x[2],res.x[3],chisq,fit_shape]

def fit_reflection(iq_data,frequencies):
    """
        Performs a least-squares fit on a reflection measurement, an array of powers and frequencies
        ASSUMPTIONS: (these go in as priors)
        - center frequency is within band
        - NA band is 1-10 times q width
        - the phase from line length does not wrap around within the band
        - uncertainty is standard devation of outer 10% of band
    """
    if 2*len(frequencies)!=len(iq_data):
        raise ValueError("point count not right nfreqs {} npows {}".format(len(frequencies),len(powers)))
    if len(frequencies)<16:
        raise ValueError("not enough points to fit transmission, need 16, got {}".format(len(powers)))

    powers=iq_packed2powers(iq_data)
#print("powers {}".format(powers))
    min_loc=np.argmin(powers)
    f0_guess=frequencies[min_loc]
    f_band=frequencies[-1]-frequencies[0]
    norm_guess=np.sqrt(max(powers))
    Q_min=f0_guess/f_band
    Q_max=100*Q_min
    Q_guess=0.5*(Q_max+Q_min)
    print("Q_min {} Q_max {}".format(Q_min,Q_max))
    beta_guess=1.0
    ten_percent_mark=int(math.ceil(0.1*len(frequencies)))

    power_mean=0.5*(np.mean(powers[0:ten_percent_mark]+np.mean(powers[len(powers)-ten_percent_mark:len(powers)])))
    power_stdev=0.5*(np.std(np.concatenate([powers[0:ten_percent_mark],powers[len(powers)-ten_percent_mark:len(powers)]])))
    uncertainty=power_stdev/(2*np.sqrt(power_mean))
    dip_depth=powers[min_loc]/(power_mean)
    #make a guess at the overall phase and phase slope of the whole thing
    left_phase=complex(-iq_data[0],-iq_data[1])
    right_phase=complex(-iq_data[-2],-iq_data[-1])
    phase_guess=cmath.phase(left_phase+right_phase)
    delay_time_guess=-(cmath.phase(right_phase)-cmath.phase(left_phase))/f_band
    p0=[norm_guess,phase_guess,f0_guess,Q_guess,beta_guess,delay_time_guess]
    print("p0 is {}".format(p0))
    def fit_fcn(x):
        #calculate the residuals of the fit as an array
        nfreq=2*len(frequencies)
        npriors=4
        norm=x[0]
        phase=x[1]
        f0=x[2]
        Q=x[3]
        beta=x[4]
        delay_time=x[5]
        resid=np.zeros(nfreq+npriors)
        #Prior 1: frequencie must be within bounds
        if f0<frequencies[0]:
            resid[nfreq]=(f0-frequencies[0])/f0_guess
            f0=frequencies[0]
        if f0>frequencies[-1]:
            resid[nfreq]=(frequencies[-1]-f0)/f0_guess
            f0=frequencies[-1]
        #Prior 2: Q must be neither too small nor too large
        if Q<Q_min:
            resid[nfreq+1]=nfreq*10*(Q-Q_min)/Q_min
            Q=Q_min
        if Q>Q_max:
            resid[nfreq+1]=nfreq*10*(Q-Q_max)/Q_min
            Q=Q_max
        #Prior 3: beta is between 0 and 2
        if beta<0:
            resid[nfreq+2]=nfreq*10*beta
            beta=0
        if beta>2:
            resid[nfreq+2]=nfreq*10*(beta-2)
            beta=2
        #Prior 4: delay_time is positive and small
        if delay_time<0:
            resid[nfreq+3]=-delay_time
            delay_time=0
        max_delay_time=3e-5 #this corresponds to nearly a kilometer of cable
        if delay_time>3.0e-5: 
            resid[nfreq+3]=max_delay_time-delay_time
            delay_time=max_delay_time
        for i in range(int(nfreq/2)):
            yp=reflection_iq_shape(frequencies[i],norm,phase,f0,Q,beta,delay_time)
            resid[2*i]=(yp.real-iq_data[2*i])/uncertainty
            resid[2*i+1]=(yp.imag-iq_data[2*i+1])/uncertainty
        return resid
    res=least_squares(fit_fcn,p0,xtol=1e-14)
    chisq=res.cost/len(powers)
    #calculate shape
    fit_shape=[]
    for i in range(len(frequencies)):
        yp=reflection_iq_shape(frequencies[i],res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5])
        fit_shape.append(yp.real)
        fit_shape.append(yp.imag)
    #TODO at this point change to dict
    #return norm,phase,f0,Q,beta,delay_time,chi-square of fit
    return [res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5],chisq,fit_shape,dip_depth]

def sidecar_fit_reflection(iq_data,frequencies):
    """fits sidecar reflection data. For now, it is separate function from the main experiment so as not to disturb it."""
    #TODO Estimate uncertainty appropriately

    if 2*len(frequencies)!=len(iq_data):
        raise ValueError("point count not right nfreqs {} npows {}".format(len(frequencies),len(powers)))
    if len(frequencies)<16:
        raise ValueError("not enough points to fit transmission, need 16, got {}".format(len(powers)))

    gamma_mag_sq=iq_packed2powers(iq_data)
    sig_gamma_mag_sq = estimate_power_uncertainty(gamma_mag_sq)
    gamma_mag = np.sqrt(gamma_mag_sq)
    gamma_phase = np.unwrap(np.angle(gamma_complex))

    po_guess = guess_reflection_fit_params(frequencies, gamma_mag_sq)
    pow_fit_param, pow_fit_cov = curve_fit(func_pow_reflected, frequencies, 
                                           p0=po_guess, sigma= sig_gamma_mag_sq)
    fo_fit, Q_fit, del_y_fit, C_fit = pow_fit_param
    red_chisq = calc_red_chisq(f, gamma_mag_sq, sig_gamma_mag_sq, func_pow_reflected, pow_fit_param)

    gamma_cav_mag, gamma_cav_phase = deconvolve_transmission(frequencies, gamma_mag, gamma_phase, C_fit) #this is from data, not from the fitted function.

    gamma_cav_mag_fo_from_fit = np.sqrt(func_pow_reflected(fo_fit, *pow_fit_param)*1/C_fit)
    
    interp_phase = interp1d(f, gamma_cav_phase, kind='cubic')
    gamma_cav_phase_fo_from_interp = interp_phase(fo_fit)
    
    beta = calculate_coupling(gamma_cav_mag_fo_from_fit, gamma_cav_phase_fo_from_interp)
    
    delay_time = None

    fit_shape = fit_shape_database_hack(f, func_pow_reflected, pow_fit_param)

    return [C_fit, gamma_cav_phase, fo_fit, Q_fit, beta, delay_time, red_chisq, fit_shape, del_y_fit]


def semicolon_array_to_json_object(data_string,label_array):
    #Convert a bunch of values separated by semicolons into a json object
    #make a best guess as to whether the values are supposed to be arrays, numbers, or strings
    #it might crash if 
    split_strings=data_string.split(";")
    data_object={}
    if len(split_strings)<len(label_array):
        raise dripline.core.DriplineValueError("not enough values given to fill semicolon_array")
    for i in range(len(label_array)):
        if "," in split_strings[i]: 
            #we assume that if there are commas, it must mean an array
            elems=split_strings[i].split(',')
            my_array=[]
            for x in elems:
                try:
                    my_array.append(float(x))
                except ValueError: #otherwise it must be a string
                    my_array.append(x)
            data_object[ label_array[i] ]=my_array
        else:
            try: #if it acts like a float, assume its a number
                data_object[ label_array[i] ]=float(split_strings[i])
            except ValueError: #otherwise it must be a string
                data_object[ label_array[i] ]=split_strings[i]
    return json.dumps(data_object)
_all_calibrations.append(semicolon_array_to_json_object)

def debug_calibration(data_object):
    logger.info("data string zero is {}".format(data_object["start_frequency"]))
    return data_object
_all_calibrations.append(debug_calibration)

def transmission_calibration(data_object):
    """takes a network analyzer output of format 
            {
        start_frequency: <number>
        stop_frequency: <number>
        iq_data: <array of numbers, packed i,r,i,r>
            }
        and augments it with a transmission fit
          {
        fit_f0: <number>
        fit_Q: <number>
        fit_norm: <number>
        fit_noise: <number>
        fit_chisq: <number>
          }
    """
    freqs=np.linspace(data_object["start_frequency"],data_object["stop_frequency"],int(len(data_object["iq_data"])/2))
    powers=iq_packed2powers(data_object["iq_data"])
    fit_norm,fit_f0,fit_Q,fit_noise,fit_chisq,fit_shape=fit_transmission(powers,freqs)
    data_object["fit_norm"]=fit_norm
    data_object["fit_f0"]=fit_f0
    data_object["fit_Q"]=fit_Q
    data_object["fit_noise"]=fit_noise
    data_object["fit_chisq"]=fit_chisq
    data_object["fit_shape"]=fit_shape
    return data_object
#return data
_all_calibrations.append(transmission_calibration)
    
def reflection_calibration(data_object):
    """takes a network analyzer output of format 
            {
        start_frequency: <number>
        stop_frequency: <number>
        iq_data: <array of numbers, packed i,r,i,r>
            }
        and augments it with a transmission fit
          {
        fit_f0: <number>
        fit_Q: <number>
        fit_norm: <number>
        fit_noise: <number>
        fit_chisq: <number>
          }
    """
    freqs=np.linspace(data_object["start_frequency"],data_object["stop_frequency"],int(len(data_object["iq_data"])/2))
    fit_norm,fit_phase,fit_f0,fit_Q,fit_beta,fit_delay_time,fit_chisq,fit_shape,dip_depth=fit_reflection(data_object["iq_data"],freqs)
    data_object["fit_norm"]=fit_norm
    data_object["fit_phase"]=fit_phase
    data_object["fit_f0"]=fit_f0
    data_object["fit_Q"]=fit_Q
    data_object["fit_beta"]=fit_beta
    data_object["fit_delay_time"]=fit_delay_time
    data_object["fit_chisq"]=fit_chisq
    data_object["fit_shape"]=fit_shape
    data_object["dip_depth"]=dip_depth
    return data_object
_all_calibrations.append(reflection_calibration)

def sidecar_reflection_calibration(data_object):
    """takes a network analyzer output of format 
            {
        start_frequency: <number>
        stop_frequency: <number>
        iq_data: <array of numbers, packed i,r,i,r>
            }
        and augments it with a transmission fit
          {
        fit_f0: <number>
        fit_Q: <number>
        fit_norm: <number>
        fit_noise: <number>
        fit_chisq: <number>
          }
    """
    freqs=np.linspace(data_object["start_frequency"],data_object["stop_frequency"],int(len(data_object["iq_data"])/2))
    fit_norm,fit_phase,fit_f0,fit_Q,fit_beta,fit_delay_time,fit_chisq,fit_shape,dip_depth=sidecar_fit_reflection(data_object["iq_data"],freqs)
    data_object["fit_norm"]=fit_norm
    data_object["fit_phase"]=fit_phase
    data_object["fit_f0"]=fit_f0
    data_object["fit_Q"]=fit_Q
    data_object["fit_beta"]=fit_beta
    data_object["fit_delay_time"]=fit_delay_time
    data_object["fit_chisq"]=fit_chisq
    data_object["fit_shape"]=fit_shape
    data_object["dip_depth"]=dip_depth
    return data_object
_all_calibrations.append(reflection_calibration)



def find_peaks(vec,fraction,start,stop):
#examine the fraction*number top values in vec and return an array contiguous sections
#which are centroids of clusters interpolated between start and stop
    count=int(math.floor(fraction*len(vec)))
    vec=np.array(vec)
    max_indices=vec.argsort()[-count:]
    sorted_max_indices=sorted(max_indices)

    peak_centroids=[]
    last_num=sorted_max_indices[0]
    peak_start=last_num
    for i in range(1,len(sorted_max_indices)):
        if sorted_max_indices[i]!=(last_num+1): #part of this peak
            peak_centroids.append(int(0.5*( peak_start+sorted_max_indices[i-1])))
            peak_start=sorted_max_indices[i]
        last_num=sorted_max_indices[i]
    peak_centroids.append(int(0.5*( peak_start+sorted_max_indices[-1])))
    return np.interp(peak_centroids,[0,len(vec)],[start,stop])

def widescan_calibration(data_object):
    """takes a network analyzer output of format 
            {
        start_frequency: <number>
        stop_frequency: <number>
        iq_data: <array of numbers, packed i,r,i,r>
            }
        and augments it with crude peak finding
          {
        peak_freqs: <array of frequencies>
          }
    """
    powers=iq_packed2powers(data_object["iq_data"])
    data_fraction=0.05 #5 percent seems to work, change as you please
    data_object["peaks"]=find_peaks(powers,data_fraction,data_object["start_frequency"],data_object["stop_frequency"]).tolist()
    return data_object
_all_calibrations.append(widescan_calibration)


class MultiFormatSpime(Spime):
    '''In standard SCPI, you should be able to send a bunch of requests separated by colons
       This spime does this and returns a json structure organized by label'''
    def __init__(self,
            get_commands=None,
            set_commands=None,
            **kwargs):
        Spime.__init__(self,**kwargs)
        self._get_commands=get_commands
        self._set_commands=set_commands

    @calibrate(_all_calibrations)
    def on_get(self):
        if self._get_commands is None:
            raise DriplineMethodNotSupportedError('<{}> has no get commands available'.format(self.name))
        to_send=""
        get_labels=[]
        for i in range(len(self._get_commands)):
            if i!=0:
                to_send=to_send+";"
            to_send=to_send+self._get_commands[i]["get_str"]
            get_labels.append(self._get_commands[i]["label"])
        result=self.provider.send([to_send])
        return semicolon_array_to_json_object(result,get_labels)
    
    def on_set(self,value):
        if self._set_commands is None:
            raise DriplineMethodNotSupportedError('<{}> has no set commands available'.format(self.name))
        try:
            value_structure=yaml.safe_load(value)
        except yaml.YAMLError as ecx:
            raise DriplineValueError('<{}> had error {}'.format(self.name,exc))
        to_send=""
        for command in self._set_commands:
            if command["label"] in value_structure:
                if len(to_send)>0:
                    to_send=to_send+";"
                to_send+="{} {}".format(command["set_str"],value_structure[command["label"]])
        to_send=to_send+";*OPC?"
        return self.provider.send([to_send])
