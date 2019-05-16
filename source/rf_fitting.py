import math
import numpy as np
import cmath
from scipy.optimize import least_squares
#import scipy.optimize


def iq_packed2powers(iq_data):
    """Turn iq data in [r,i,r,i,r,i...] format into an array of powers"""
    powers=np.zeros(int(len(iq_data)/2))
    for i in range(int(len(iq_data)/2)):
        powers[i]=iq_data[2*i]*iq_data[2*i]+iq_data[2*i+1]*iq_data[2*i+1]
    return powers

def unpack_iq_data(iq_data):
    """takes iq data in [r,i,r,i,r,i] format and unpacks into two arrays [r,r,r],[i,i,i]"""
    ret_r=np.zeros(len(iq_data)/2)
    ret_i=np.zeros(len(iq_data)/2)
    for i in range(len(iq_data),2):
        ret_r[i/2]=iq_data[i]
        ret_i[i/2]=iq_data[i+1]
    return ret_r,ret_i

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
#beta=2*atan(beta)/3.14159+1.0
    delta=Q*(f-f0)/f0
    denom=1/(1+4*delta*delta)
    response=complex(denom*((beta-1)-4*delta*delta),-denom*2*beta*delta)
    phase=cmath.exp(complex(0,phase+delay_time*(f-f0)))
    return response*phase
#return [norm*(my_r*math.cos(phase)+my_i*math.sin(phase)),norm*(-my_r*math.sin(phase)+my_i*math.cos(phase))]


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

    f0_guess=frequencies[math.floor(len(frequencies)/2)]
    f_band=frequencies[-1]-frequencies[0]
    norm_guess=max(powers)
    Q_min=f0_guess/f_band
    Q_max=20*Q_min
    Q_guess=0.5*(Q_max+Q_min)
    ten_percent_mark=int(math.ceil(0.1*len(frequencies)))
#print("ten percent mark is {}".format(ten_percent_mark))
    noise_guess=0.5*(np.mean(powers[0:ten_percent_mark]+np.mean(powers[len(powers)-ten_percent_mark:len(powers)])))/norm
    uncertainty=0.5*(np.std(powers[0:ten_percent_mark]+np.std(powers[len(powers)-ten_percent_mark:len(powers)])))
#print("uncertainty is {}".format(uncertainty))
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
#print("state {} {} {} {}".format(norm,f0,Q,noise))
        resid=np.zeros(nfreq+npriors)
        #add priors
        #Prior 1: frequencie must be within bounds
        if f0<frequencies[0]:
            resid[nfreq]=(f0-frequencies[0])/f0_guess
            f0=frequencies[0]
        if f0>frequencies[-1]:
            resid[nfreq]=(frequencies[-1]-f0)/f0_guess
            f0=frequencies[-1]
        #Prior 2: Q must be neither too small nor too large
        if Q<Q_min:
            resid[nfreq+1]=(Q-Q_min)/Q_guess
            Q=Q_min
        if Q>Q_max:
            resid[nfreq+1]=(Q-Q_max)/Q_guess
            Q=Q_max
        #Prior 3: noise level not too big
        if noise<0:
            resid[nfreq+2]=-noise
            noise=0
        if noise>0.1:
            resid[nfreq+2]=(noise-0.1)
            #note I do not clamp noise, it is possible it gets this high, jsut undesirable
        for i in range(nfreq):
            yp=transmission_power_shape(frequencies[i],norm,f0,Q,noise)
            resid[i]=(yp-powers[i])/uncertainty
        return resid
    #actual fit done here
    res=least_squares(fit_fcn,p0)
#res=scipy.optimize.least_squares(fit_fcn,p0)
    chisq=res.cost/len(powers)
    #return norm,f0,Q,noise, chi square
    return [res.x[0],res.x[1],res.x[2],res.x[3],chisq]

def fit_reflection(iq_data,frequencies):
    """
        Performs a least-squares fit on a reflection measurement, an array of powers and frequencies
        ASSUMPTIONS: (these go in as priors)
        - center frequency is within band
        - band is 1-10 times q width
        - the phase from line length does not wrap around within the band
        - uncertainty is standard devation of outer 10% of band
    """
    if 2*len(frequencies)!=len(iq_data):
        raise ValueError("point count not right nfreqs {} npows {}".format(len(frequencies),len(powers)))
    if len(frequencies)<16:
        raise ValueError("not enough points to fit transmission, need 16, got {}".format(len(powers)))

    powers=iq_packed2powers(iq_data)
#print("powers {}".format(powers))
    f0_guess=frequencies[np.argmin(powers)]
    f_band=frequencies[-1]-frequencies[0]
    norm_guess=np.sqrt(max(powers))
    Q_min=f0_guess/f_band
    Q_max=10*Q_min
    Q_guess=0.5*(Q_max+Q_min)
    beta_guess=1.0
    ten_percent_mark=int(math.ceil(0.1*len(frequencies)))

    power_mean=0.5*(np.mean(powers[0:ten_percent_mark]+np.mean(powers[len(powers)-ten_percent_mark:len(powers)])))/norm
    power_stdev=0.5*(np.std(powers[0:ten_percent_mark]+np.std(powers[len(powers)-ten_percent_mark:len(powers)])))
    uncertainty=power_stdev/(2*power_mean)
#uncertainty=np.sqrt(uncertainty_squared)
    #make a guess at the overall phase and phase slope of the whole thing
    left_phase=complex(-iq_data[0],-iq_data[1])
    right_phase=complex(-iq_data[-2],-iq_data[-1])
    phase_guess=cmath.phase(left_phase+right_phase)
#phase_guess=cmath.phase(left_phase)+cmath.phase(right_phase)
    delay_time_guess=(cmath.phase(right_phase)-cmath.phase(left_phase))/f_band
    p0=[norm_guess,phase_guess,f0_guess,Q_guess,beta_guess,delay_time_guess]
#print(p0)
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
            resid[nfreq+1]=(Q-Q_min)/Q_guess
            Q=Q_min
        if Q>Q_max:
            resid[nfreq+1]=(Q-Q_max)/Q_guess
            Q=Q_max
        #Prior 3: beta is between 0 and 2
        if beta<0:
            resid[nfreq+2]=beta
            beta=0
        if beta>2:
            resid[nfreq+2]=beta-2
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
    res=least_squares(fit_fcn,p0)
    chisq=res.cost/len(powers)
    #return norm,phase,f0,Q,beta,delay_time,chi-square of fit
    return [res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5],chisq]
            





if __name__=='__main__':
    #build some fake data
    norm=1.0
    f0=800e6
    band=1e5
    Q=3e4
    noise=0.1
    npoints=256
    freqs=np.linspace(f0-band/2,f0+band/2,npoints)
    powers=[ transmission_power_shape(f,norm,f0,Q,noise)+0.01*np.random.normal() for f in freqs ]
    fit_norm,fit_f0,fit_Q,fit_noise,fit_chisq=fit_transmission(powers,freqs)
#print("targets: {} {} {} {}".format(norm,f0,Q,noise))
#p#rint("fits   : {} {} {} {} {}".format(fit_norm,fit_f0,fit_Q,fit_noise,fit_chisq))
    
    beta=0.9
    delay_time=1e-5
    phase=0.5
    iq_complex=[ reflection_iq_shape(f,norm,phase,f0,Q,beta,delay_time) +0.01*np.random.normal()+0.01j*np.random.normal() for f in freqs ]
    iq_packed=[]
    for i in range(len(freqs)):
        iq_packed.append(iq_complex[i].real)
        iq_packed.append(iq_complex[i].imag)
#print("iq_packed comp {}".format(iq_packed))

    fit_norm,fit_phase,fit_f0,fit_Q,fit_beta,fit_delay_time,chisq=fit_reflection(iq_packed,freqs)
    print("#norm {} phase {} f0 {} Q {} beta {} delay_time {} chisq {}".format(fit_norm,fit_phase,fit_f0,fit_Q,fit_beta,fit_delay_time,chisq))
    fit_data=[reflection_iq_shape(f,fit_norm,fit_phase,fit_f0,fit_Q,fit_beta,fit_delay_time) for f in freqs ]

    for i in range(len(freqs)):
        print("{} {} {} {} {}".format(freqs[i],iq_complex[i].real,iq_complex[i].imag,fit_data[i].real,fit_data[i].imag))
#for i in range(len(powers)):
#        print("{} {} {}".format(freqs[i],powers[i],transmission_power_shape(freqs[i],fit_norm,fit_f0,fit_Q,fit_noise)))
