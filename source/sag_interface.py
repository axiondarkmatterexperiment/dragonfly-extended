import asteval
import six

import dripline
#import dragonfly

#import scipy
import scipy.constants as consts
import socket as skt
import numpy as np
import math

import axion_waveform_lib

import logging
logger = logging.getLogger('dragonfly.custom.sag_interface')

class SAGCoordinator(dripline.core.Endpoint):
    '''
    Coordinated interactions with all instruments within the broader sag system.
    Provides a single point of contact and uniform interface to the SAG.
    '''
    def __init__(self, enable_output_sets=None, disable_output_sets=None, sag_injection_sets=None, update_waveform_sets=None, dummy_message_sets=None, sag_arb_waveform_name=None, switch_endpoint=None, extra_logs=[], state_extra_logs={}, f_stan=50, f_rest=650000000, line_shape='maxwellian', **kwargs):
        '''
        enable_output_sets: (list) - a sequence of endpoints and values to set to configure the system to be ready to start output of a signal
        disable_output_sets: (list) - a sequence of endpoints and values to set to configure the system to not produce any output
        sag_injection_sets: (list) - a sequence of endpoints to set to create a particular injection; configuration determined from asteval of input values
        update_waveform_sets: (list) - a sequence of endpoints to set to update the wave form to be generated by the ARB
        switch_endpoint: (string) - name of the endpoint used for switching the signal path into the weak port
        extra_logs: (list) - list of endpoint names to cmd `scheduled_action` (to trigger a log) whenever the SAG is configured
        state_extra_logs: (dict) - dict with keys being valid switch_endpoint states (string) and values being a list of extra sensors to log when entering that state.
        f_stan: (float) - value of the frequency seperation between, in Hz
        f_rest: (float) - value of the axion rest mass frequency for time series generation, in Hz
        line_shape: (string) - name of the axion line shape model to be used
        '''
        dripline.core.Endpoint.__init__(self, **kwargs)

        self.enable_output_sets = enable_output_sets
        self.disable_output_sets = disable_output_sets
        self.sag_injection_sets = sag_injection_sets
        self.update_waveform_sets = update_waveform_sets
        self.dummy_message_sets = dummy_message_sets # this is for a dummy test message to confirm message passing to arb service
        self.sag_arb_waveform_name = sag_arb_waveform_name
        self.switch_endpoint = switch_endpoint
        self.extra_logs = extra_logs
        self.state_extra_logs = state_extra_logs
        

        self.evaluator = asteval.Interpreter()

        # attributes for line shape generation into the waveform generator (33220A)
        self.tscaled = []
        self.tseries = []
        self.spectrum = []
        self.re_tseries = []
        self.scale = []
        self.f_stan = f_stan
        self.f_rest = f_rest
        self.line_shape = line_shape
        self.N = 65536
        self.n = 65000
        self.h = consts.Planck #J/Hz
        self.h_eV = consts.physical_constants['Planck constant in eV s'][0] #eV/Hz
        self.eV = consts.physical_constants['joule-electron volt relationship'][0] #6.24e18
        self.c = (consts.c) *10**-3 #km/s
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
            logger.info("setting endpoint '"+str(this_endpoint)+"' with value: "+str(this_value)) 
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

    def update_waveform(self, **parameters):
        '''
        updates waveform of choice into the arb static memory slot of choice
        '''
        logger.info('in update waveform')
        self.f_rest = float(parameters['f_rest'])
        self.line_shape = str(parameters['shape_type'])
        self.sag_arb_waveform_name = str(parameters['arb_waveform_name'])
        self.send_thru_sag_arb_service = bool(parameters['use_sag_arb_service'])
        logger.info('send_thru_sag_arb_service set to {}'.format(self.send_thru_sag_arb_service))
    
        def SAG_Spec():
            '''
            This function generates the spectral distribution function of the axion waveform due to the distribution of 
            kinetic energy in the axion field as measured in the experiment's laboratory frame (taken from the supplied 
            keyword arguments to update_waveform).
            Four line shapes are supported:
            max_2017 = N-body maxwellian form from Lentz et al. 2017
            big_flows = Caustic ring flows (n=4?) from Pierre Sikivie halo model
            bose_nbody_2020 = Bose N-body halo model from Lentz et al. 2020
            maxwellian = isothermal sphere halo model form from Turner et al. 1990
            '''
            spec=np.zeros(self.N)
            if ("body" or "2017" or "max") in self.line_shape.lower():
                line_shape = 'n-body'
            elif ("flow" or "big" or "sikivie") in self.line_shape.lower():
                line_shape = 'big flows'
            elif ("bose" or "entangled" or "bec") in self.line_shape.lower():
                line_shape = 'bose_nbody_2020'
            else:  # maxwellian
                line_shape = 'SHM'
            bandwidth = self.f_stan*(self.N - self.n)
            aw = axion_waveform_lib.axion_waveform_sampler(line_shape=line_shape, f_rest=self.f_rest, f_spacing=self.f_stan, \
                     bandwidth = bandwidth)
            spec_short, freqs = aw.get_freq_spec()
            spec[-len(spec_short):] = spec_short
            self.spectrum = list(spec)
            return None

        def spectrum2Timeseries():
            '''
            This function takes the designated spectral function and converts it into a time series.
            As the input is a (power) spectral function, we accomplish this by taking the square root of the PSF, 
            give it a phase (from a uniform random distribution) and perform an inverse FFT.
            '''
            N=np.size(self.spectrum)
            print(N)
            sqrt_spectrum = np.sqrt(self.spectrum)
            phases = 0#np.random.uniform(low=0,high=2*np.pi,size=N)
            amplitudes_posf = (sqrt_spectrum*np.exp(1j*phases))[0:N//2]
            amplitudes_negf = [np.conjugate(amp) for amp in amplitudes_posf]
            if N%2==0:
                amps_conc = list(amplitudes_posf) + list(reversed(amplitudes_negf))
            else:
                amps_conc = list(amplitudes_posf) + [0] + list(reversed(amplitudes_negf))
            tseries=np.fft.ifft(amps_conc) 
            #tseries_trunc = tseries
            print(len(tseries))
            #tseries=tseries.real 
            
            #re_tseries=np.zeros(self.N) 
            
            #for j in range(0,self.N):
            #    re_tseries[j]=tseries[j-self.N//2] #rescaled

            self.re_tseries = list(tseries.real) # though it should be real already
            return None

        def reScale():
            '''
            this function rescales the tseries amplitude to -8191:8191
            '''
            maxVal=np.amax(self.re_tseries)
            minVal=np.amin(self.re_tseries)
            
            #print('maxVal='+str(maxVal))
            #print('minVal='+str(minVal))
            
            N=np.size(self.re_tseries)
            scale=np.zeros(N)
            #rescales amplitude to -8191:8191
            for i in range(0, N):
                scale[i]=int(round((16382*(self.re_tseries[i]-minVal)/(maxVal-minVal))-8191))
            
            self.scale = scale
            return None

        
        def writeWF():
            '''
            This function reads out the tscaled spectrum and returns all values of tscaled as a string
            '''
            self.scale = [int(number) for number in self.scale] # redundant to reScale, yes, but necessary for unknown reasons
            logger.info('waveform element numbers of type: '+str(type(self.scale[0])))
            self.msg="DATA:DAC VOLATILE, "
            self.WFstr = ""
            
            N=np.size(self.scale)
            
            for i in range(0, N):
                self.WFstr+=str(int(self.scale[i]))
                if i<N-1:
                    self.WFstr+=", "
            
            self.msg+=self.WFstr 
            self.msg+="\n"
            
            # also partitioning the waveform string into J parts
            J = 4
            K = N//(J-1)     
            self.WFsegs = {"sag_waveform_array_"+str(i): self.scale[i*K:(i+1)*K] for i in range(0,J)} 
            return None
        
        def writeToAG():
            '''
            TCP_IP = a string representing a hostname in Internet domain notation or an IPv4 address
            TCP_PORT = int
            '''
            logger.info('In writeToAG')
            TCP_IP='10.95.101.64'
            TCP_PORT=5025
            BUFFER_SIZE=1024

            s=skt.socket(skt.AF_INET, skt.SOCK_STREAM) #creating a new socket
            s.connect((TCP_IP, TCP_PORT)) #connect to a remote socket at address

            msg3=self.msg
            
            msg2="FREQ 50 \n" #set frequency [Hz]
            s.send(msg2.encode()) # message needs to be encoded as a byte string

            s.send(msg3.encode()) #sends tscaled to the socket
            
            #self.waveform_name = 'MY_AXION4'
            msg="DATA:COPY "+str(self.sag_arb_waveform_name)+"\n" #this saves the name of the line shape to long term memory
            s.send(msg.encode())
            logger.info("messages passed to arb")
            s.close()
            return None
        
        def sendToAG():
            '''
            Iterates over messages to send to waveform generator to update line shape 
            '''
            logger.info('in send to AG')
            values = self.WFsegs
            logger.info('setting waveform array in '+str(len(values))+' segments: '+str(values))
            logger.info('update_waveform_sets: '+str(self.update_waveform_sets))
            self._do_set_collection(self.update_waveform_sets, values)
            logger.info('set complete')
            logger.info('pinging arb for current waveform stats')
            self.provider.cmd('sag_arb','retrieve_setting',value=[self.sag_arb_waveform_name],timeout=300)
            logger.info('settings retrieved')
            logger.info('instructing sag_arb service to save waveform to {}'.format(self.sag_arb_waveform_name))
            self.provider.cmd('sag_arb','send_waveform',value=[self.sag_arb_waveform_name],timeout=300)
            logger.info('waveform sent')
            return None

        # execute the in-method functions to generate the time series (and load to the waveform generator?)
        #SAG = SAG_Maker(f_stan=self.f_stan, f_rest=self.f_rest, line_shape=self.line_shape)
        SAG_Spec()
        spectrum2Timeseries()
        reScale()
        writeWF()
        if self.send_thru_sag_arb_service:
            sendToAG()
        else:
            writeToAG()

        return None

    def dummy_message(self, **parameters):
        """
        Method for sending dummy message N items long with entries b bits each to the sag arb service
        Will send both in stringified and list formats
        """
        logger.info('in dummy message')       
        b = int(parameters['b'])
        N = int(parameters['N'])
        Nlist = [2**b for i in range(N)]
        stringlist = ""
        for i,entry in enumerate(Nlist):
                stringlist+=str(int(entry))
                if i<N-1:
                    stringlist+=", "
        #print(stringlist)
        
        values = {"sag_dummy_message_0":stringlist}
        logger.info('sending dummy list string to SAG arb service')
        self._do_set_collection(self.dummy_message_sets, values)
        #self.provider.cmd('sag_arb','print_dummy_message',timeout=300)
        
        values = {"sag_dummy_message_0":Nlist}
        logger.info('sending explicit list to SAG arb service')
        self._do_set_collection(self.dummy_message_sets, values)
        #self.provider.cmd('sag_arb','print_dummy_message',timeout=300)
        return None






