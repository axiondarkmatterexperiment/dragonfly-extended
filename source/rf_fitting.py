import math
import numpy as np
import cmath
import csv
from scipy.optimize import least_squares
#from scipy.signal import find_peaks
#from scipy.signal import find_peaks_cwt
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
#beta=2*atan(beta)/3.14159+1.0 only if we have trouble keeping beta within bounds
    delta=Q*(f-f0)/f0
    denom=1/(1+4*delta*delta)
    response=norm*complex(denom*((beta-1)-4*delta*delta),-denom*2*beta*delta)
    phase=cmath.exp(complex(0,phase-delay_time*(f-f0)))
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

    power_mean=0.5*(np.mean(powers[0:ten_percent_mark]+np.mean(powers[len(powers)-ten_percent_mark:len(powers)])))/norm_guess
    power_stdev=0.5*(np.std(np.concatenate([powers[0:ten_percent_mark],powers[len(powers)-ten_percent_mark:len(powers)]])))
    uncertainty=power_stdev/(2*np.sqrt(power_mean))
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
    #return norm,phase,f0,Q,beta,delay_time,chi-square of fit
    return [res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5],chisq,fit_shape]
 
def find_peaks(vec,fraction,start_freq,stop_freq):
#examine the fraction*number top values and return contiguous sections
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
    return np.interp(peak_centroids,[0,len(vec)],[start_freq,stop_freq])





if __name__=='__main__':
    freqs=[]
    pows=[]
    powslog=[]
    f=open("wide.csv","r")
    c=csv.reader(f)
    for row in c:
        freqs.append(float(row[0]))
        powslog.append(float(row[1]))
        pows.append(10.0**(float(row[1])/10))
    f.close()

#Q_target=10000
#bin_width=len(freqs)*((0.5*(freqs[0]+freqs[-1])/Q_target)/(freqs[-1]-freqs[0]))
#print("bin width {}".format(bin_width))
    pk_inds=find_peaks(pows,0.05,freqs[0],freqs[-1])
#pk_inds=find_peaks_cwt(pows,[3],min_snr=2)
    print("peaks at {}".format(pk_inds))

    f=open("x.x","w")
    for i in range(len(freqs)):
#        if i in pk_inds:
#            f.write("{} {} 1\n".format(freqs[i],powslog[i]))
#        else:
            f.write("{} {} 0\n".format(freqs[i],powslog[i]))
    f.close()
"""
    freqs=[]
    reals=[]
    imags=[]
    f=open("refl_real.csv","r")
    c=csv.reader(f)
    for row in c:
        freqs.append(float(row[0]))
        try:
            reals.append(float(row[1]))
        except ValueError:
            reals.append(reals[-1])
    f.close()
    freqs=[]
    f=open("refl_imag.csv","r")
    c=csv.reader(f)
    for row in c:
        freqs.append(float(row[0]))
        try:
            imags.append(float(row[1]))
        except ValueError:
            imags.append(imags[-1])
    f.close()
    iq_data=[]
    for i in range(len(freqs)):
        iq_data.append(reals[i])
        iq_data.append(imags[i])
    norm,phase,f0,Q,beta,delay_time,chisq,shape=fit_reflection(iq_data,freqs)
    print("chisq is {}".format(chisq))
    print("norm is {}".format(norm))
    print("phase is {}".format(phase))
    print("Q is {}".format(Q))
    print("f0 is {}".format(f0))
    print("beta is {}".format(beta))
    print("delay_time is {}".format(delay_time))
    f=open("x.x","w")
    for i in range(len(freqs)):
        f.write("{} {} {} {} {}\n".format(freqs[i],reals[i],imags[i],shape[2*i],shape[2*i+1]))
    f.close()
"""


