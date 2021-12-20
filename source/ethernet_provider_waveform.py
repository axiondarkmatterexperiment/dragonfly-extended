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
    
    def send_waveform(self,**parameters):
        '''
       Takes the waveform endpoint set strings and 
       Restructure of 'send' in Ethernetprovider
        '''
        # collect waveform writing endpoints, combine, and consolidate commands 


        # execute send from EthernetProvider        
        return None
    
    def send_waveform(self, commands, **kwargs):
        '''
        Takes the waveform endpoint set strings and 
        Restructure of 'send' in Ethernetprovider
        
        Standard provider method to communicate with instrument.
        NEVER RENAME THIS METHOD!
        commands (list||None): list of command(s) to send to the instrument following (re)connection to the instrument, still must return a reply!
                             : if impossible, set as None to skip
        '''
        if isinstance(commands, six.string_types):
            commands = [commands]
        self.alock.acquire()

        try:
            data = self._send_commands(commands)
        except socket.error as err:
            logger.warning("socket.error <{}> received, attempting reconnect".format(err))
            self._reconnect()
            data = self._send_commands(commands)
            logger.critical("Ethernet connection reestablished")
        except exceptions.DriplineHardwareResponselessError as err:
            logger.critical(str(err))
            try:
                self._reconnect()
                data = self._send_commands(commands)
                logger.critical("Query successful after ethernet connection recovered")
            except exceptions.DriplineHardwareConnectionError:
                logger.critical("Ethernet reconnect failed, dead socket")
                raise exceptions.DriplineHardwareConnectionError("Broken ethernet socket")
            except exceptions.DriplineHardwareResponselessError as err:
                logger.critical("Query failed after successful ethernet socket reconnect")
                raise exceptions.DriplineHardwareResponselessError(err)
        finally:
            self.alock.release()
        to_return = ';'.join(data)
        logger.debug("should return:\n{}".format(to_return))
        return to_return
    
    
    pass
