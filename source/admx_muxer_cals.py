

from dripline.core import calibrate
import dragonfly

import logging
logger = logging.getLogger('dragonfly.implementations.custom')

def lin_lin_cal(values_x, values_y, this_x):
    try:
        high_index = [i>this_x for i in values_x].index(True)
    except ValueError:
        raise dripline.core.DriplineValueError("raw value is likely above calibration range")
    if high_index == 0:
        raise dripline.core.DriplineValueError("raw value is below calibration range")
    m = (values_y[high_index] - values_y[high_index - 1]) / (values_x[high_index] - values_x[high_index - 1])
    return values_y[high_index - 1] + m * (this_x - values_x[high_index - 1])

def pt100_calibration(resistance):
    '''Calibration for the (many) muxer pt100 temperature sensor endpoints'''
    values_x = [2.29, 9.39, 18.52,  39.72,  60.26,  80.31,    100,  119.4, 138.51]
    values_y = [20,     50, 73.15, 123.15, 173.15, 223.15, 273.15, 323.15, 373.15]
    return lin_lin_cal(values_x, values_y, resistance)


class ADMXMuxGetSpime(dragonfly.implementations.MuxerGetSpime):
    @calibrate([pt100_calibration])
    def on_get(self):
        # Note that these lines are a direct copy from the base class, I'm not sure how to undecorate
        #      when I try to change to the decorator here
        result = self.provider.send([self.get_str.format(self.ch_number)])
        logger.debug('very raw is: {}'.format(result))
        return result.split()[0]
