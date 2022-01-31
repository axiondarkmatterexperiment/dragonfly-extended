import dripline
from dripline.core import MsgRequest, DriplineError, op_t
from dragonfly.implementations import EthernetProvider

import logging
logger = logging.getLogger('dragonfly.custom.sag_interface')

class EthernetProviderWaveform(EthernetProvider):
    '''
    An EthernetProvider class for interacting with the arb (Agilent 33220A), particularly for handling long waveform messages passed the the waveforme generator
    '''
    def __init__(self,write_waveform_prefix="DATA:DAC VOLATILE ",write_waveform_terminator=" \n",**kwargs):
        '''
        Initialize EthernetProvider parent

        '''
        EthernetProvider.__init__(self, **kwargs)
        self.write_waveform_prefix = write_waveform_prefix
        self.write_waveform_terminator = write_waveform_terminator
        return None
    
    def set_partial(self, endpoint, value, specifier=None, timeout=0):
        '''
        method for setting partial messages
        
        [kw]args:
        endpoint (string): routing key to which an OP_GET will be sent
        value : value to assign
        specifier (string|None): specifier to add to the message
        timeout (float|int): time to timeout, in seconds
        '''
        payload = {'values':[value]}
        result = self._generate_request_partial( msgop=op_t.set, target=endpoint, specifier=specifier, payload=payload )
        reply = self._receiver.wait_for_reply(result, timeout) # not sure how crucial the role of this step is
        return result
    
    def _generate_request_partial(self, msgop, target, specifier=None, payload=None, timeout=None, lockout_key=False):
        '''
        internal helper method to standardize generating requests for partial messages
        returns (partial) message request
        '''
        # import scarab just in time
        import sys
        sys.path.append('/usr/local/src/dripline-cpp/build/')
        import scarab
        # now to generate request
        a_specifier = specifier if specifier is not None else ""
        #a_request = MsgRequest.create(payload=scarab.to_param(payload), msg_op=msgop, routing_key=target, specifier=a_specifier)
        a_request = MsgRequest.create(payload=payload, msg_op=msgop, routing_key=target, specifier=a_specifier)
        #receive_reply = self.send(a_request)
        #if not receive_reply.successful_send:
        #    raise DriplineError('unable to send request')
        return a_request
    
    def send_waveform(self,commands,**parameters):
        '''
       Takes list of waveform strings, concatenates them, then sends tham to the Agilent 33220A as single message
       For use w/ waveform endpoints to be passed to arbitrary waveform generator
        '''
        logger.info('in send_waveform')
        # collect waveform writing endpoints, combine, and consolidate commands
        commands_dict = {}
        for endpoint in commands:
            commands_dict.update(endpoint) 
        waveform_write_command = commands_dict['sag_arb_write_waveform_0']
        waveform_write_command += commands_dict['sag_arb_write_waveform_1']+', ' # additional characters required for total waveform to be interpreted as list
        waveform_write_command += commands_dict['sag_arb_write_waveform_2']+', '
        waveform_write_command += commands_dict['sag_arb_write_waveform_3']+', '
        waveform_write_command += commands_dict['sag_arb_write_waveform_4']
        logger.info('structure of endpoint object: '+str(self._endpoints['sag_arb_store_waveform_0']))
        logger.info('structure of endpoint object: '+str(self._endpoints['sag_arb_store_waveform_0'].__dict__.keys()))
        waveform_store_endpoints = { k:v for k,v in self._endpoints.items() if 'sag_arb_store_waveform_' in k }
        N_endpoints = len(waveform_store_endpoints)
        waveform_list = []
        for i in range(0,N_endpoints):
                waveform_list.append(waveform_store_endpoints['sag_arb_store_waveform_'+str(i)].on_set)
        waveform_string = ', '.join([str(val) for val in waveform_list])
        write_waveform_cmd_string = self.write_waveform_prefix + waveform_string + self.write_waveform_terminator
        # execute send from EthernetProvider for waveform write to arb volitile memory   
        self.send(write_waveform_cmd_string, **parameters)
        # execute send from EthernetProvider for waveform copy and save to linshape 
        save_waveform_cmd_string = waveform_store_endpoints['sag_arb_save_waveform'].on_set
        self.send(save_waveform_cmd_string, **parameters)
        return None
    
    pass
