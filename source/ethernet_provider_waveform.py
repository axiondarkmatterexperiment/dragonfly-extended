import dripline
from dragonfly.implementations import EthernetProvider

import logging
logger = logging.getLogger('dragonfly.custom.sag_interface')

class EthernetProviderWaveform(EthernetProvider):
    '''
    An EthernetProvider class for interacting with the arb (Agilent 33220A), particularly for handling long waveform messages passed the the waveforme generator
    '''
    def __init__(self,write_waveform_prefix="DATA:DAC VOLATILE ",write_waveform_terminator=" \n",save_waveform_prefix="DATA:COPY ",save_waveform_terminator=", VOLATILE \n",**kwargs):
        '''
        Initialize EthernetProvider parent

        '''
        logger.info('in EthernetProviderWaveform')
        EthernetProvider.__init__(self, **kwargs)
        self.write_waveform_prefix = write_waveform_prefix
        self.write_waveform_terminator = write_waveform_terminator
        self.save_waveform_prefix = save_waveform_prefix
        self.save_waveform_terminator = save_waveform_terminator
        return None
    
    def send_waveform(self,*values,**parameters):
        '''
        Takes list of waveform strings, concatenates them, then sends tham to the Agilent 33220A as a single list
        For use w/ waveform endpoints to be passed to arbitrary waveform generator
       
        other_messages: (dict|None) contains location on the arb the new waveform should be saved to  
        '''
        logger.info('in send_waveform')
        # collect waveform storing endpoints, combine, and send
        store_waveform_endpoints = { k:v for k,v in self._endpoints.items() if 'sag_arb_store_waveform_' in k }
        N_endpoints = len(store_waveform_endpoints)
        waveform_list = []
        for i in range(0,N_endpoints):
                waveform_list.extend(store_waveform_endpoints['sag_arb_store_waveform_'+str(i)].get_value())
        waveform_string = ', '.join([str(val) for val in waveform_list])
        write_waveform_cmd_string = self.write_waveform_prefix + waveform_string + self.write_waveform_terminator
        # execute send from EthernetProvider for waveform write to arb volitile memory   
        self.send(write_waveform_cmd_string, **parameters)
        # execute send from EthernetProvider for waveform copy and save to linshape 
        save_location = values[0]
        logger.info('saving waveform to location {}'.format(save_location))
        save_waveform_cmd_string = self.save_waveform_prefix + save_location + self.save_waveform_terminator
        self.send(save_waveform_cmd_string, **parameters)
        return None
    
    def retrieve_setting(self,*values,**parameters):
        '''
        Retrieves a handful settings
        '''
        waveform_location = values[0]
        waveform_setting_1_string = "FUNC?"
        return1 = self.send(waveform_setting_1_string,**parameters)
        logger.info('function type set to: {}'.format(return1))
        waveform_setting_2_string = "FREQ?"
        return2 = self.send(waveform_setting_2_string,**parameters)
        logger.info('frequency set to: {}'.format(return2))
        waveform_setting_3_string = "MEM:STAT:NAME? 0"
        return3 = self.send(waveform_setting_3_string,**parameters)
        logger.info('Mem state name 0 set to: {}'.format(return3))
        waveform_setting_4_string = "MEM:STAT:NAME? 4"
        return4 = self.send(waveform_setting_4_string,**parameters)
        logger.info('Mem state name 4 set to: {}'.format(return4))
        return None
    
    def print_dummy_message(self,*values):
        '''
        Prints a dummy message passed between sag interface and arb services
        '''
        dummy_message = {k:v for k,v in self._endpoints.items() if 'sag_arb_store_dummy_message' in k}
        logger.info('dummy message stored as: {}'.format(dummy_message))
        return None
    
    pass
