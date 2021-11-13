import asteval
import six

import dripline
#import dragonfly

import logging
logger = logging.getLogger('dragonfly.custom.sag_interface')

class SAGCoordinator(dripline.core.Endpoint):
    '''
    Coordinated interactions with all instruments within the broader sag system.
    Provides a single point of contact and uniform interface to the SAG.
    '''
    def __init__(self, enable_output_sets=None, disable_output_sets=None, sag_injection_sets=None, switch_endpoint=None, extra_logs=[], state_extra_logs={}, f_stan=60, f_rest = 650000000, line_shape='maxwellian',**kwargs):
        '''
        enable_output_sets: (list) - a sequence of endpoints and values to set to configure the system to be ready to start output of a signal
        disable_output_sets: (list) - a sequence of endpoints and values to set to configure the system to not produce any output
        sag_injection_sets: (list) - a sequence of endpoints to set to create a particular injection; configuration determined from asteval of input values
        switch_endpoint: (string) - name of the endpoint used for switching the signal path into the weak port
        extra_logs: (list) - list of endpoint names to cmd `scheduled_action` (to trigger a log) whenever the SAG is configured
        state_extra_logs: (dict) - dict with keys being valid switch_endpoint states (string) and values being a list of extra sensors to log when entering that state.

        '''
        dripline.core.Endpoint.__init__(self, **kwargs)

        self.enable_output_sets = enable_output_sets
        self.disable_output_sets = disable_output_sets
        self.sag_injection_sets = sag_injection_sets
        self.switch_endpoint = switch_endpoint
        self.extra_logs = extra_logs
        self.state_extra_logs = state_extra_logs

        self.evaluator = asteval.Interpreter()

        self.msg = ""
        self.waveform_name = ""
        self.spectrum = []
        self.re_tseries = []
        self.scale = []
        self.f_stan = f_stan
        self.f_rest = f_rest
        self.line_shape = line_shape
        self.N = 65536
        self.n = int(self.N -(self.f_stan*10)) #if uneven spacing, will return list
        self.h = scipy.constants.Planck #J/Hz
        self.h_eV = scipy.constants.physical_constants['Planck constant in eV s'][0] #eV/Hz
        self.eV = scipy.constants.physical_constants['joule-electron volt relationship'][0] #6.24e18
        self.c = (scipy.constants.c) *10**-3 #km/s
        self.v_bar = (230/0.85) #km/s
        self.r = 0.85
        self.alpha =  0.36 #+/- 0.13 from Lentz et al. 2017
        self.beta = 1.39 #+/- 0.28 from Lentz et al. 2017
        self.T = 4.7*(10**(-7)) #+/-1.9 from Lentz et al. 2017

    def _do_set_collection(self, these_sets, values):
        '''
        A utility method for processing a list of sets
        '''
        set_list = []
        # first parse all string evaluations, make sure they all work before doing any actual setting
        for a_calculated_set in these_sets:
            logger.debug("dealing with calculated_set: {}".format(a_calculated_set))
            if len(a_calculated_set) > 1:
                raise dripline.core.DriplineValueError('all calculated sets must be a single entry dict')
            [(this_endpoint,set_str)] = six.iteritems(a_calculated_set)
            logger.debug('trying to understand: {}->{}'.format(this_endpoint, set_str))
            this_value = set_str
            if '{' in set_str and '}' in set_str:
                try:
                    this_set = set_str.format(**values)
                except KeyError as e:
                    raise dripline.core.DriplineValueError("required parameter, <{}>, not provided".format(e.message))
                logger.debug('substitutions make that RHS = {}'.format(this_set))
                this_value = self.evaluator(this_set)
                logger.debug('or a set value of {}'.format(this_value))
            set_list.append((this_endpoint, this_value))
        # now actually try to set things
        for this_endpoint, this_value in set_list:
            #logger.info("if I weren't a jerk, I'd do:\n{} -> {}".format(this_endpoint, this_value))
            self.provider.set(this_endpoint, this_value)

    #def _do_log_noset_sensors(self):
    def _do_extra_logs(self, sensors_list):
        '''
        Send a scheduled_action (log) command to configured list of sensors (this is for making sure we log everything
        that should be recorded on each injection, but which is not already/automatically logged by a log_on_set action)
        '''
        logger.info('triggering logging of the following sensors: {}'.format(sensors_list))
        for a_sensor in sensors_list:
            self.provider.cmd(a_sensor, 'scheduled_action')

    def update_state(self, new_state):
        # do universal extra logs
        self._do_extra_logs(self.extra_logs)
        # do state-specific extra logs
        if new_state in self.state_extra_logs:
            self._do_extra_logs(self.state_extra_logs[new_state])
        else:
            logger.warning('state <{}> does not have a state-specific extra logs list, please create one (it may be empty)'.format(new_state))
        # actually set to the new state
        if new_state == 'term':
            self.do_disable_output_sets()
            self.provider.set(self.switch_endpoint, "term")
        elif new_state == 'sag':
            self.do_enable_output_sets()
            self.provider.set(self.switch_endpoint, "sag")
        elif new_state == 'vna':
            # set the switch
            self.provider.set(self.switch_endpoint, "vna")
            # disable outputs
            self.do_disable_output_sets()
        elif new_state == 'locking':
            raise dripline.core.DriplineValueError('locking state is not currently supported')
        else:
            raise dripline.core.DriplineValueError("don't know how to set the SAG state to <{}>".format(new_state))

    def do_enable_output_sets(self):
        logger.info('enabling lo outputs')
        self._do_set_collection(self.enable_output_sets, {})

    def do_disable_output_sets(self):
        logger.info('disabling lo outputs')
        self._do_set_collection(self.disable_output_sets, {})

    def configure_injection(self, **parameters):
        '''
        parameters: (dict) - keyworded arguments are available to all sensors as named substitutions when calibrating
        '''
        logger.info('in configure injection')
        # set to state 'sag' (which enables output)
        self.update_state("sag")
        # to extra sets calculated from input parameters
        self._do_set_collection(self.sag_injection_sets, parameters)

    def make_waveform(self):
    
        def get_du(self):

            self.m_a = (self.h*self.f_rest)/(self.c**2) #J/c^2
            if self.line_shape == 'max_2017':
                self.m_a = self.m_a * self.eV #eV/c^2
                self.m_a = self.m_a * SAG.c**2
            else:
                pass
            self.du = (self.h*self.f_stan)/((self.m_a*(self.v_bar**2))/2) 

        
        def max_term(self,x,i):
            term = ((((i-self.n)*self.f_stan)*self.h_eV)/(self.rest_m*self.T))**x
            return term

        def SAG_Spec(self):
            '''
            This function generates the distribution function in terms of the axion kinetic energy   measured in the experiment's laboratory (Turner 1990 [5b]).

            u = (i-n)du where u dimensionless form of axion KE in lab frame
            r = ratio of the velocity of the Sun through the Galaxy to the rms halo velocity
            max_2017 = maxwellian form from Letz et al. 2017
            maxwellian = maxwellian form from Turner et al. 2015
            '''
            spec=np.zeros(self.N) 
            if self.line_shape == 'max_2017':
                for i in range(self.n, self.N):
                    spec[i] = self.max_term(self.alpha,i)*math.exp(-(self.max_term(self.beta,i)))

            else: 
                for i in range(self.n, self.N):
                    spec[i]=np.sqrt(np.sqrt(3/(2*np.pi))/self.r*np.exp(-1.5*(self.r*self.r+(i-self.n)*self.du))*np.sinh(3*self.r*np.sqrt((i-self.n)*self.du)))


            spec = preprocessing.normalize([spec],norm='l1')
            sns.set(style='white')
            plt.figure(figsize=(8,5))
            plt.title('Distribution Function')
            plt.xlabel('Frequency') #frequency steps
            plt.ylabel('Amplitude')
            plt.plot(spec[0])
            self.spectrum = spec[0]

        def FourierTrans(self):
            N=np.size(self.spectrum) 
            tseries=np.fft.ifft(self.spectrum) #compute the one-dimensional inverse discrete Fourier Transform from frequency domain to time domain
            
            tseries=tseries.real 
            
            plt.figure(figsize=(8,5))
            plt.title('Time Series Spectrum')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.figure(2)    
            plt.plot(tseries)
            
            re_tseries=np.zeros(self.N) 
            
            for i in range(0,self.N):
                re_tseries[i]=tseries[i-self.N//2] #rescaled

            plt.figure(figsize=(8,5))
            plt.title('Rescaled Time Series Spectrum')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.figure(3)
            plt.plot(re_tseries)
            self.re_tseries = re_tseries

        def reScale(self):
            '''
            this function rescales the tseries amplitude to -8191:8191
            '''
            maxVal=np.amax(self.re_tseries)
            minVal=np.amin(self.re_tseries)
            
            print (f'maxVal={maxVal}')
            print (f'minVal={minVal}')
            
            N=np.size(self.re_tseries)
            scale=np.zeros(N)
            #rescales amplitude to -8191:8191
            for i in range(0, N):
                scale[i]=int(round((16382*(self.re_tseries[i]-minVal)/(maxVal-minVal))-8191))
            

            plt.figure(figsize=(8,5))
            plt.title('Rescaled v.2 Frequency Spectrum')
            plt.xlabel('Frequency')
            plt.ylabel('Amplitude')
            plt.plot(scale)
            self.scale = scale

        
        def writeWF(self):
            '''
            This function reads out the tscaled spectrum and returns all values of tscaled as a string
            '''
            tseries=FourierTrans(SAG_Spec(self.n ,self.r, self.du))
            tscaled=reScale(tseries)
            self.msg="DATA:DAC VOLATILE, "
            
            N=np.size(tseries)
            
            for i in range(0, N):
                msg+=str(int(tscaled[i]))
                if i<N-1:
                    msg+=", "
            
            self.msg+="\n"

        def writeToAG():
            '''
            TCP_IP = a string representing a hostname in Internet domain notation or an IPv4 address
            TCP_PORT = int
            '''

            TCP_IP='10.95.101.64'
            TCP_PORT=5025
            BUFFER_SIZE=1024

            s=skt.socket(skt.AF_INET, skt.SOCK_STREAM) #creating a new socket
            s.connect((TCP_IP, TCP_PORT)) #connect to a remote socket at address

            msg3=writeWF()
            
            msg2="FREQ 50 \n" #set frequency [Hz]
            s.send(msg2) 
            

            s.send(msg3) #sends tscaled to the socket
            
            self.waveform_name = 'MY_AXION4'
            msg=f"DATA:COPY {self.waveform_name}\n" #this saves the name of the line shape make modular on the line shape
            s.send(msg) 
            s.close()
            return

        SAG = SAG_Maker(f_stan=self.f_stan, f_rest=self.f_rest, line_shape=self.line_shape)
        SAG.get_du()
        SAG.SAG_Spec()
        SAG.FourierTrans()
        SAG.reScale()
        SAG.writeWF()
        SAG.writeAG()

        print(f'\n--- Waveform of Type {self.line_shape} at Center Frequency {self.f_rest} Hz Saved as {self.waveform_name}---\n', flush=True)

        
            
        

            





