'''
All communication with instruments is performed via ethernet, or via an intermediate interface that inherits from EthernetProvider.  This class must be kept general, with specific cases handled in higher-level interfaces.

General rules to observe:
- All instruments should communicate with a response_terminator (typically \r and/or \n, possibly an additional prompt)
- All endpoints and communication must be configured to return a response, otherwise nasty timeouts will be incurred

Every instrument config file (hardware repo) must specify:
- socket_info, command_terminator, response_terminator
Optional config options:
- socket timeout   : if long timeouts expected (undesirable)
- cmd_at_reconnect : if instrument response buffer needs clearing or special configuration
- reply_echo_cmd   : if instrument response contains the query (see glenlivet)
- bare_response_terminator : special response terminator (undesirable)
'''
from __future__ import absolute_import
#import socket
import serial
import threading
import six

from dripline.core import Provider, exceptions, fancy_doc

import logging
logger = logging.getLogger(__name__)

__all__ = []


__all__.append('SerialProvider')
@fancy_doc
class SerialProvider(Provider):
    def __init__(self,
                 port='/dev/ttyUSB2',
                 baudrate=9600,
                 parity=serial.PARITY_NONE,
                 stopbits=serial.STOPBITS_ONE,
                 bytesize=serial.EIGHTBITS,
                 **kwargs
                 ):
        '''
        > Connection-related options:
        socket_timeout (float): time in seconds for the socket to timeout
        socket_info (tuple): (<network_address_as_str>, <port_as_int>)
        cmd_at_reconnect (list||None): list of command(s) to send to the instrument following (re)connection to the instrument, still must return a reply!
                                     : if impossible, set as None to skip
        reconnect_test (str): expected return value from final reconnect command
        > Query-related options:
        command_terminator (str): string to append to commands
        response_terminator (str||None): string to strip from responses, this MUST exist for get method to function properly!
        bare_response_terminator (str||None): abbreviated string to strip from responses containing only prompt
                                            : only used to handle non-standard lockin behavior
        reply_echo_cmd (bool): set to True if command+command_terminator or just command are present in reply
        '''
        Provider.__init__(self, **kwargs)

        self.serial = serial.Serial()
        self.port = port
        self.baudrate = baudrate
        self.parity = parity
        self.bytesize = bytesize
        
    def send(self, commands, **kwargs):
        '''
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


    def _send_commands(self, commands):
        '''
        Take a list of commands, send to instrument and receive responses, do any necessary formatting.

        commands (list||None): list of command(s) to send to the instrument following (re)connection to the instrument, still must return a reply!
                             : if impossible, set as None to skip
        '''
        all_data=[]

        for command in commands:
            command += self.command_terminator
            logger.debug("sending: {}".format(repr(command)))
            #self.socket.send(command.encode())
            # Chelsea added this line
            self.serial.write(command+'\r\n')
            if command == self.command_terminator:
                blank_command = True
            else:
                blank_command = False

            data = self._listen(blank_command)

            if self.reply_echo_cmd:
                if data.startswith(command):
                    data = data[len(command):]
                elif not blank_command:
                    raise exceptions.DriplineHardwareResponselessError("Bad ethernet query return: {}".format(data))
            logger.info("sync: {} -> {}".format(repr(command),repr(data)))
            all_data.append(data)
        return all_data


    def _listen(self, blank_command=False):
        '''
        Query socket for response.

        blank_comands (bool): flag which is True when command is exactly the command terminator
        '''
        data = ''
        try:
            while True:
                data += self.socket.recv(1024).decode(errors='replace')
                if data.endswith(self.response_terminator):
                    terminator = self.response_terminator
                    break
                # Special exception for lockin data dump
                elif self.bare_response_terminator and data.endswith(self.bare_response_terminator):
                    terminator = self.bare_response_terminator
                    break
                # Special exception for disconnect of prologix box to avoid infinite loop
                if data == '':
                    raise exceptions.DriplineHardwareResponselessError("Empty socket.recv packet")
        except socket.timeout:
            logger.warning("socket.timeout condition met; received:\n{}".format(repr(data)))
            if blank_command == False:
                raise exceptions.DriplineHardwareResponselessError("Unexpected socket.timeout")
            terminator = ''
        logger.debug(repr(data))
        data = data[0:data.rfind(terminator)]
        return data
