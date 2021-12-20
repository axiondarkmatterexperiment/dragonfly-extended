import dripline
from dragonfly.implementations import EthernetProvider

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
       Takes list of endpoint command strings and concatenates them before send as single message
       For use w/ waveform endpoints to be passed to arbitrary waveform generator
        '''
        # collect waveform writing endpoints, combine, and consolidate commands 
        waveform_write_command = ''
        for command in commands:
            waveform_write_command += command
        # execute send from EthernetProvider      
        self.send(waveform_write_command, **parameters)
        return None
    
    pass
