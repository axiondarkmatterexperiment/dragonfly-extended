import dripline
from dragonfly.implementations import EthernetProvider

import logging
logger = logging.getLogger('dragonfly.custom.sag_interface')

class EthernetProviderWaveform(EthernetProvider):
    '''
    An EthernetProvider class for interacting with the arb (Agilent 33220A), particularly for handling long waveform messages passed the the waveforme generator
    '''
    def __init__(self,**kwargs):
        '''
        Initialize EthernetProvider parent

        '''
        EthernetProvider.__init__(self, **kwargs)
        return None
    
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
        #for command in commands:
        #    waveform_write_command += command
        # execute send from EthernetProvider for waveform write to arb volitile memory   
        #logger.info('sending command to arb starting with: "'+waveform_write_command[0:10]+'", and of length: '+str(len(waveform_write_command)))
        self.send(waveform_write_command, **parameters)
        # execute send from EthernetProvider for waveform copy and save to linshape 
        waveform_save_command = commands_dict['sag_arb_save_waveform']
        #logger.info('sending command to arb starting with: "'+waveform_save_command[0:10]+'", and of length: '+str(len(waveform_save_command)))
        self.send(waveform_save_command, **parameters)
        return None
    
    pass
