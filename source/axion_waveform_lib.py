import numpy as np
import scipy.constants as consts
import math

class axion_waveform_sampler():
    """
    Class for sampling one of several lineshapes for the axion dark matter halo frequency distribution function
    
    Child classes:
    ----------
    None
    """
    def __init__(self, line_shape='SHM', f_rest=650000000, f_spacing=50, \
                     bandwidth = 1.e4, v_bar = (230/0.85), r = 0.85, **kwargs):
        """
        Initialization method
        
        Arguments:
        ----------
        1) line_shape (string) optional: name of the waveform lineshape to be generated. 
        Currently supported 'SHM', 'n-body', 'bose', 'big flow'
        2) f_rest (float) optional: frequency in Hz of an axion at rest in observer reference frame
        3) f_spacing (int or float) optional: bin spacing for sampling of distribution, in Hz
        4) bandwidth (float) optional: frequency bandwith being sampled, in Hz, with lower endpoint at f_rest
        5) v_bar (float) optional: mean speed of virialized dark matter in MW halo measured in the galactic center frame, in km/s
        6) r (float) optional: the ratio of the observer's orbital speed around MW relative to v_bar

        Returns:
        -------- 
        """
        # attributes for spectral line shape generation 
        # prescribed attributes
        self.bandwidth = bandwidth
        self.f_stan = f_spacing
        self.f_rest = f_rest
        self.line_shape = line_shape
        self.h = consts.Planck #J/Hz
        self.h_eV = consts.physical_constants['Planck constant in eV s'][0] #eV/Hz
        self.eV = consts.physical_constants['joule-electron volt relationship'][0] #6.24e18
        self.c = (consts.c) *10**-3 #km/s
        self.v_bar = v_bar #km/s
        self.r = r
        self.alpha =  0.36 #+/- 0.13 from Lentz et al. 2017
        self.beta = 1.39 #+/- 0.28 from Lentz et al. 2017
        self.T = 4.7e-7 #+/-1.9e-7 from Lentz et al. 2017
        
        # derived attributes
        self.freqs = np.arange(self.f_rest-self.f_stan, self.f_rest+self.bandwidth,self.f_stan)
        self.v_boost_c = np.array([0,self.v_bar*self.r,0])/self.c # in fraction of c in galactic coordiate frame [vr,vt,vz]
        self.v_bar_c = self.v_bar/self.c
        return None


    def get_freq_spec(self):
        """
        Method for retrieving the power spectra and frequencies for the argued lineshape

        Returns
        -------
        spec : (np.ndarray) 1D array of spectral weights of argued lineshape
        freqs : (np.ndarray) 1D array of frequencies (in Hz) at which the spectrum was sampled
        """
        if "shm" in self.line_shape.lower():
            spec = self.standard_halo_model()
        elif "-body" in self.line_shape.lower():
            spec = self.n_body_cdm_halo_model()
        elif "bose" in self.line_shape.lower():
            spec = self.n_body_bose_halo_model(Corr=0.0)
        elif "flow" in self.line_shape.lower():
            spec = self.big_flow_caustic_model()
        else:
            raise AttributeError("The argued line_shape does not match a supported shape")
        
        return spec, self.freqs
    
    def normShape(self,sampleArray):
        """
        Method for normalizing lineshape weights to unity

        Returns
        -------
        weights : (np.ndarray) 1D array of normalized weights
        """
        total = sampleArray.sum()
        return sampleArray/total
    
    def standard_halo_model(self):
        """
        Method for returning sampled power spectra of the Standard Halo Model lineshape.
        For reference, see Turner 1990 (https://link.aps.org/doi/10.1103/PhysRevD.42.3572)

        Returns
        -------
        spec : (np.ndarray) 1D array of spectral weights of  lineshape

        """
        vb = np.linalg.norm(self.v_boost_c)
        beta = 1./self.v_bar_c**2
        Es = self.freqs/self.f_rest - 1.0
        amps = np.array([2*(beta/np.pi)**(0.5)/vb*np.sinh(2*beta*np.sqrt(2*E)*vb)*np.exp(-2*E*beta-beta*vb**2) \
                         if E>=0.0 else 0.0 for E in Es])
        
        spec = self.normShape(amps)
        
        return spec
    
    def n_body_cdm_halo_model(self):
        """
        Method for returning sampled power spectra of the lineshape of Lentz et al 2017 
        (https://iopscience.iop.org/article/10.3847/1538-4357/aa80dd/meta), 
        derived from the Romulus25 N-body cosmological simulation

        Returns
        -------
        spec : (np.ndarray) 1D array of spectral weights of  lineshape

        """
        Es = self.freqs/self.f_rest - 1.0
        amps = np.array([np.power(E/self.T,self.alpha)*np.exp(-np.power(E/self.T,self.beta)) \
                         if E>=0.0 else 0.0 for E in Es])
        
        spec = self.normShape(amps)
        
        return spec
    
    def n_body_bose_halo_model(self,Corr=0.0):
        """
        Method for returning sampled power spectra of the lineshape derived from the results of 
        Lentz et al 2020 (https://academic.oup.com/mnras/article/493/4/5944/5801026)
        , which simulated single halo infall using Bose-inspired DM physics
        
        Arguments:
        ----------
        1) Corr (float) optional: correlation parameter for the prominence of the secondary 'Bose lobe' feature

        Returns
        -------
        spec : (np.ndarray) 1D array of spectral weights of  lineshape

        """
        # primary maxwellian shape
        vVir = 130./self.c # fraction of c, virial speed of halo. Estimated by narrowed lineshape of Lentz et al 2017
        beta = 1/vVir**2# effective temperature of halo central feature
        vboost = self.v_boost_c
        vb = np.linalg.norm(vboost)
        
        Es = self.freqs/self.f_rest - 1.0
        
        centralGauss = np.array([2*(beta/np.pi)**(0.5)/vb*np.sinh(2*beta*np.sqrt(2*E)*vb)*np.exp(-2*E*beta-beta*vb**2) \
                         if E>=0.0 else 0.0 for E in Es])
        centralGauss = self.normShape(centralGauss)
        
        # the secondary 'Bose lobe' feature
        vMeanBose = np.array((2.5*vVir,0,0)) # mean speed for Bose lobe, along +/- radial direction
        betaBose = beta*4
        vbPlus = np.linalg.norm(vboost+vMeanBose)
        lobePlus = np.array([2*(betaBose/np.pi)**(0.5)/vbPlus*np.sinh(2*betaBose*np.sqrt(2*E)*vbPlus)*np.exp(-2*E*betaBose-betaBose*vbPlus**2) \
                         if E>=0.0 else 0.0 for E in Es])
        lobePlus = self.normShape(lobePlus)
        vbMinus = np.linalg.norm(vboost-vMeanBose)
        lobeMinus=np.array([2*(betaBose/np.pi)**(0.5)/vbMinus*np.sinh(2*betaBose*np.sqrt(2*E)*vbMinus)*np.exp(-2*E*betaBose-betaBose*vbMinus**2) \
                         if E>=0.0 else 0.0 for E in Es])
        lobeMinus = self.normShape(lobeMinus)
        BoseLobe = 0.25*(lobePlus + lobeMinus) # normalized Bose lobe feature, assuming two radial velocity contributions and lobes are well separated 
        
        # superpose two contributions, using a correlation quadratic weighting 
        PeakAmp = 0.45 + 0.55*Corr**2
        GaussContr = PeakAmp*centralGauss
        BoseContr = (1.-PeakAmp)*BoseLobe
        amps = GaussContr + BoseContr

        # normalize total shape
        spec = self.normShape(amps)
        
        return spec
            
    def big_flow_caustic_model(self):
        """
        Method for returning sampled power spectra of the Big Flows lineshape derived from 
        Pierre Sikivie et al's semi-analytic models of the MW axion halos.
        For reference, see (https://arxiv.org/abs/2007.10509v2)

        Returns
        -------
        spec : (np.ndarray) 1D array of spectral weights of  lineshape

        """
        # relative densities for each flow
        gram2GeV = 1.e-3*consts.physical_constants["kilogram-electron volt relationship"][0]*1.e-9
        rhoBig = 2.e-23*gram2GeV # GeV/cc
        rhoLittle = 2.e-24*gram2GeV
        rhoUp = 9.6e-24*gram2GeV
        rhoDown = 8.4e-24*gram2GeV
        # central velocities in Galactic center fixed frame
        # using galactic cylindrical coordinates (vr,vphi,vz)
        vBig = np.array((-104.4, 509.4, 6.1) )# km/s
        vLittle = np.array((-0.2, 530.0, 4.5))
        vUp = np.array((-115.3,505.1,44.8))
        vDown = np.array((-116.4,505.4,-38.1))
        # relative velocity of Sun to MW center, in galactic cylindrical basis
        vSun = np.array((12.9, 245.6, 7.78)) # km/s s.galcen_v_sun # note thqat the x,y,z frame of icrs identifies with the galactic cylindrical coordinates
        # speed dispersion of each flow
        dv = 70. #m/s
        # boosted central velocity into sun-stationary frame
        vBigSol = vBig - vSun
        vLittleSol = vLittle - vSun
        vUpSol = vUp - vSun
        vDownSol = vDown - vSun
        
        # determine central frequency for  each flow
        fBig = self.f_rest*(1. + 0.5*np.linalg.norm(vBigSol)**2/self.c**2)
        fLittle = self.f_rest*(1. + 0.5*np.linalg.norm(vLittleSol)**2/self.c**2)
        fUp = self.f_rest*(1. + 0.5*np.linalg.norm(vUpSol)**2/self.c**2)
        fDown = self.f_rest*(1. + 0.5*np.linalg.norm(vDownSol)**2/self.c**2)
    
        
        # find index of bin close to each flow
        f_start = self.freqs[0]
        idxBig = math.ceil((-f_start+fBig)/self.f_stan)
        idxLittle = math.ceil((-f_start+fLittle)/self.f_stan)
        idxUp = math.ceil((-f_start+fUp)/self.f_stan)
        idxDown = math.ceil((-f_start+fDown)/self.f_stan)
        
        # set 1-bin amplitude using density of flow
        amps = np.zeros(len(self.freqs))
        amps[idxBig] += rhoBig
        amps[idxLittle] += rhoLittle
        amps[idxUp] += rhoUp
        amps[idxDown] += rhoDown
        
        # normalize total shape
        spec = self.normShape(amps)

        return spec


if __name__ == "__main__":
    
    waveform_generator = axion_waveform_sampler()
    SHM_waveform, freqs = waveform_generator.get_freq_spec()
    print(f"SHM waveform {SHM_waveform}")
