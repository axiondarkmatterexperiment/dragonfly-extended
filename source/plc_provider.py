'''
A basic provider for talking to the PLC
'''
from __future__ import absolute_import

import ctypes

import pyModbusTCP.client

from dripline.core import calibrate, Provider, Spime

import logging
logger = logging.getLogger('dragonfly.custom')

class modbus_provider(Provider):
    def __init__(self,
                 modbus_host=None,
                 modbus_port=502,
                 **kwargs):
        '''
        '''
        Provider.__init__(self, **kwargs)
        if modbus_host is None:
            raise ValueError("modbus_host is a required configuration parameter for <modbus_provider>")
        self.modbus_client = pyModbusTCP.client.ModbusClient(host=modbus_host, port=modbus_port, auto_open=True)

    def read_holding(self, register, n_registers):
        logger.debug('calling read_holding_registers({}, {})'.format(register, n_registers))
        return self.modbus_client.read_holding_registers(register, n_registers)

class plc_value(Spime):
    def __init__(self, register=None, n_registers=2, **kwargs):
        Spime.__init__(self, **kwargs)
        if register is None:
            raise ValueERror("register is a required configuration parameter for <plc_value>")
        self.register = register
        self.n_registers = n_registers

    @calibrate()
    def on_get(self):
        raw_bits_data = self.provider.read_holding(self.register, self.n_registers)
        logger.debug('raw bits are: ', raw_bits_data)
        # modbus puts the order in reverse...
        raw_bits_data.reverse()
        raw_bits = sum([d<<16*n for d,n in zip(raw_bits_data, range(self.n_registers-1, -1, -1))])
        typed_value = ctypes.c_float.from_buffer(ctypes.c_int(raw_bits))
        return typed_value.value
