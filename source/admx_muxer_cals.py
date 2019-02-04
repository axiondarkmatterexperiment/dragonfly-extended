
import math

from dripline.core import calibrate
import dripline
import dragonfly

import logging
logger = logging.getLogger('dragonfly.implementations.custom')

#lin-log is log(x)
#log-lin is log(y)
def piecewise_cal(values_x, values_y, this_x, log_x=False, log_y=False):
    if log_x:
        logger.info("doing log x cal")
        values_x = [math.log(x) for x in values_x]
        this_x = math.log(this_x)
    if log_y:
        logger.info("doing log y cal")
        values_y = [math.log(y) for y in values_y]
    try:
        high_index = [i>this_x for i in values_x].index(True)
    except ValueError:
        raise dripline.core.DriplineValueError("raw value is likely above calibration range")
    if high_index == 0:
        raise dripline.core.DriplineValueError("raw value is below calibration range")
    m = (values_y[high_index] - values_y[high_index - 1]) / (values_x[high_index] - values_x[high_index - 1])
    to_return = values_y[high_index - 1] + m * (this_x - values_x[high_index - 1])
    if log_y:
        to_return = math.exp(to_return)
    return to_return

def pt100_cal(resistance):
    '''Calibration for the (many) muxer pt100 temperature sensor endpoints'''
    values_x = [2.29, 9.39, 18.52,  39.72,  60.26,  80.31,   100.,  119.4, 138.51]
    values_y = [20.,   50., 73.15, 123.15, 173.15, 223.15, 273.15, 323.15, 373.15]
    return piecewise_cal(values_x, values_y, resistance)

# Cernox sensors
cernox_sensors = []
def x84971(resistance):
    '''Calibration for a cernox'''
    values_x = [82.2, 297.0, 3988.0]
    values_y = [305.0, 77.0, 4.2]
    return piecewise_cal(values_x, values_y, resistance, log_x=True, log_y=True)
cernox_sensors.append(x84971)

class ADMXMuxGetSpime(dragonfly.implementations.MuxerGetSpime):
    @calibrate([pt100_cal] + cernox_sensors)
    def on_get(self):
        # Note that these lines are a direct copy from the base class, I'm not sure how to undecorate
        #      when I try to change to the decorator here
        result = self.provider.send([self.get_str.format(self.ch_number)])
        logger.debug('very raw is: {}'.format(result))
        return result.split()[0]
