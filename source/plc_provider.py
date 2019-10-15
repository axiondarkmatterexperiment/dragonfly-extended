'''
A basic provider for talking to the PLC
'''
from __future__ import absolute_import

import ctypes

import pyModbusTCP.client

from dripline.core import calibrate, Provider, Spime

import logging
logger = logging.getLogger('dragonfly.custom')

## base calibration function... this should live somewhere common...
#lin-log is log(x)
#log-lin is log(y)
def piecewise_cal(values_x, values_y, this_x, log_x=False, log_y=False):
    if log_x:
        logger.debug("doing log x cal")
        values_x = [math.log(x) for x in values_x]
        this_x = math.log(this_x)
    if log_y:
        logger.debug("doing log y cal")
        values_y = [math.log(y) for y in values_y]
    try:
        high_index = [i>this_x for i in values_x].index(True)
    except ValueError:
        high_index = -1
        logger.warning("raw value is above the calibration range, extrapolating")
        #raise dripline.core.DriplineValueError("raw value is likely above calibration range")
    if high_index == 0:
        high_index = 1
        logger.warning("raw value is below the calibration range, extrapolating")
        #raise dripline.core.DriplineValueError("raw value is below calibration range")
    m = (values_y[high_index] - values_y[high_index - 1]) / (values_x[high_index] - values_x[high_index - 1])
    to_return = values_y[high_index - 1] + m * (this_x - values_x[high_index - 1])
    if log_y:
        to_return = math.exp(to_return)
    return to_return

def mother_dewar_lhe(fraction):
    ''' converts linear position along the sensor to liquid liters of He '''
    #values_x = [0., 1., 3., 5., 6., 8., 10., 12., 13., 15., 17., 18., 20., 22., 24., 25., 27., 29., 31., 32., 34., 36., 37., 39., 41., 43., 44., 46., 48., 50., 51., 53., 55., 56., 58., 60., 62., 63., 65., 67., 68., 70., 72., 74., 75., 77., 79., 81., 82., 84., 86., 87., 89., 91., 93., 94., 96., 98., 100.]
    #values_y = [0., 2.17080717038854, 8.61458655758931, 19.228374975655, 33.9092092386384, 52.5541261605922, 75.060162555569, 101.324355237622, 131.243741020803, 164.715356719165, 201.636239146762, 241.903425117645, 285.413951445867, 332.064854945482, 381.753172430541, 434.375940715098, 489.830196613206, 548.012976938916, 607.525698416445, 667.038419893975, 726.551141371504, 786.063862849033, 845.576584326563, 905.089305804092, 964.602027281621, 1024.11474875915, 1083.62747023668, 1143.14019171421, 1202.65291319174, 1262.16563466927, 1321.6783561468, 1381.19107762433, 1440.70379910186, 1500.21652057939, 1559.72924205691, 1619.24196353444, 1678.75468501197, 1738.2674064895, 1797.78012796703, 1857.29284944456, 1916.80557092209, 1976.31829239962, 2034.50107272533, 2089.95532862344, 2142.57809690799, 2192.26641439305, 2238.91731789267, 2282.42784422089, 2322.69503019177, 2359.61591261937, 2393.08752831773, 2423.00691410091, 2449.27110678297, 2471.77714317794, 2490.4220600999, 2505.10289436288, 2515.71668278095, 2522.16046216815, 2524.33126933854]
    # x values are read in % but OEM provided as a function of linear position in inches
    #cryofab but bad values_x = [inches * ( 100. / 58. ) for inches in range(1,59)]
    #cryofab but values_y = [0.0, 3.4, 13.4, 29.6, 51.5, 78.7, 110.9, 147.5, 188.3, 232.7, 280.5, 331.1, 384.1, 439.2, 495.9, 553.8, 612.5, 671.6, 730.8, 790.0, 849.1, 908.3, 967.5, 1026.7, 1085.8, 1145.0, 1204.2, 1263.4, 1322.5, 1381.7, 1440.9, 1500.1, 1559.2, 1618.4, 1677.6, 1736.8, 1795.9, 1855.1, 1914.3, 1973.4, 2032.6, 2091.8, 2150.9, 2209.6, 2267.6, 2324.3, 2379.3, 2432.4, 2483.0, 2530.7, 2575.1, 2615.9, 2652.6, 2683.0, 2711.9, 2733.8, 2750.0, 2760.0, 2763.4]
    
    #already in percent now, calibration we made from measured level volume pairs
    values_x =[-2.,6.,9.4,10.4,13.3,14.6,17.2,17.5,19.7,20.,24.2,26.7,27.,27.5,28.4,29.5,29.8,31.4,32.7,33.2,33.5,36.7,37.4,38.2,39.5,39.6,42.,44.6,47.4,50.5,51,53.8,54.4,55.3,56.4,57.6]
    values_y = [50.,210.,230.,320.,380.,430.,475.,510.,570.,590.,600.,700.,710.,770.,790.,850.,910.,920.,1010.,1020.,1100.,1110.,1190.,1200.,1240.,1260.,1310.,1390.,1490.,1510.,1540.,1620.,1640.,1720.,1730.,1750.]
    return piecewise_cal(values_x, values_y, fraction)

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

    @calibrate([mother_dewar_lhe])
    def on_get(self):
        raw_bits_data = self.provider.read_holding(self.register, self.n_registers)
        logger.debug('raw bits are: ', raw_bits_data)
        # modbus puts the order in reverse...
        raw_bits_data.reverse()
        raw_bits = sum([d<<16*n for d,n in zip(raw_bits_data, range(self.n_registers-1, -1, -1))])
        typed_value = ctypes.c_float.from_buffer(ctypes.c_int(raw_bits))
        return typed_value.value

class plc_bool(Spime):
    # note well, the register here (and used in modbusTCP) is 0 indexed, but our PLC documentation is all
    # indexed from 1 (and have a preceeding 4, ie this_register = (PLC_code_register % 400000) -1 )
    def __init__(self, register=None, bit=None, **kwargs):
        Spime.__init__(self, **kwargs)
        if register is None:
            raise ValueERror("register is a required configuration parameter for <plc_value>")
        self.register = register
        self.bit = bit
        self.n_registers = 1

    @calibrate()
    def on_get(self):
        raw_bits_data = self.provider.read_holding(self.register, self.n_registers)
        logger.debug('raw bits are: ', raw_bits_data)
        this_state = bool(raw_bits_data[0] & 2**self.bit)
        return this_state
