
import math

from dripline.core import calibrate
import dripline
import dragonfly

import logging
logger = logging.getLogger('dragonfly.implementations.custom')

_all_calibrations = []

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
_all_calibrations.append(pt100_cal)

# Cernox sensors
def x84971(resistance):
    '''Calibration for a cernox'''
    values_x = [82.2, 297.0, 3988.0]
    values_y = [305.0, 77.0, 4.2]
    return piecewise_cal(values_x, values_y, resistance, log_x=True, log_y=True)
_all_calibrations.append(x84971)

def x76782(resistance):
    '''Calibration for a cernox'''
    values_x = [73.9, 251., 2896.]
    values_y = [305., 77., 4.2]
    return piecewise_cal(values_x, values_y, resistance, log_x=True, log_y=True)
_all_calibrations.append(x76782)

def x76779p2(resistance):
    '''Calibration for a cernox'''
    values_x = [73.2, 246., 2663.6]
    values_y = [305., 77., 4.2]
    return piecewise_cal(values_x, values_y, resistance, log_x=True, log_y=True)
_all_calibrations.append(x76779p2)

def x41840(resistance):
    '''Calibration for a cernox'''
    values_x = [58.8, 243., 5104.]
    values_y = [305., 77., 4.2]
    return piecewise_cal(values_x, values_y, resistance, log_x=True, log_y=True)
_all_calibrations.append(x41840)

def x41849(resistance):
    '''Calibration for a cernox'''
    values_x = [53.4, 215., 4337.]
    values_y = [305., 77., 4.2]
    return piecewise_cal(values_x, values_y, resistance, log_x=True, log_y=True)
_all_calibrations.append(x41849)

# RuOx sensors
def RuOx202a(resistance):
    '''Calibration for a RuOx'''
    values_x = [2008.5, 2130, 2243.1507919, 2247.82837008, 2252.67165521, 2257.69242943, 2262.90261077, 2268.31490773, 2273.94481798, 2279.80870753, 2285.92402049, 2292.31202231, 2298.99521874, 2305.99750346, 2313.34748079, 2321.07638638, 2329.22021822, 2337.81827044, 2346.91683612, 2356.56678496, 2366.82825705, 2377.76798468, 2389.46691367, 2395.62834041, 2402.01464403, 2408.63984847, 2415.51928432, 2422.6692313, 2430.10802159, 2437.85528088, 2445.93319361, 2454.36562977, 2463.17990092, 2472.40561454, 2482.07710006, 2492.23190591, 2502.91428312, 2514.17316164, 2526.06736559, 2538.66273378, 2552.04016269, 2566.29150584, 2581.5329411, 2597.89797142, 2615.55883889, 2634.71721236, 2655.64428995, 2678.65722169, 2704.16570341, 2732.66352706, 2764.84669842, 2778.92266862, 2793.7502015, 2801.4579521, 2809.36832643, 2825.86689284, 2843.52852376, 2862.62685027, 2883.18617288, 2905.22478212, 2928.99838467, 2954.81636122, 2968.5757862, 2982.95670854, 2998.00687716, 3013.77849689, 3030.32760197, 3047.71832163, 3066.02023264, 3085.31287362, 3105.68291885, 3127.23059199, 3150.06590596, 3174.31753284, 3200.12792484, 3227.66616756, 3257.12144397, 3288.72235472, 3322.72790766, 3359.45759811, 3399.27876097, 3442.65523479, 3490.12848945, 3542.39690771, 3600.2885489, 3664.89413383, 3737.51755467, 3819.87129619, 3914.02777317, 4022.75310485, 4083.59510837, 4149.45062312, 4220.96353588, 4298.96074351, 4384.44214969, 4478.77921405, 4583.74002105, 4701.99100529, 4836.59753612, 4990.611791, 5166.85835176, 5369.38032441, 5611.83504012, 5903.84561811, 6038.02877119, 6184.22615218, 6344.068314, 6519.43869339, 6712.75769848, 6926.90774683, 7165.50302782, 7432.9462211, 7735.1128401, 8079.35355643, 8270.27720363, 8475.69626238, 8697.52271613, 8937.92669063, 9199.42217175, 9485.27250376, 9799.25139812, 10145.8475355, 10531.1357652, 10962.3265779, 11448.087719, 11999.9339566, 12632.1906056, 13363.5119791, 14217.3470276, 15224.3774852, 16425.034128, 17877.7531888, 19665.2816715, 21927.1350297, 23308.1186795, 24920.4604245, 26839.0366817, 29185.9753819, 32137.3219905, 35958.554143, 40977.9945386, 47523.915336, 56177.874649, 69191.1003872]
    values_y = [305.0, 77.0, 40.0, 39.0, 38.0, 37.0, 36.0, 35.0, 34.0, 33.0, 32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.5, 19.0, 18.5, 18.0, 17.5, 17.0, 16.5, 16.0, 15.5, 15.0, 14.5, 14.0, 13.5, 13.0, 12.5, 12.0, 11.5, 11.0, 10.5, 10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.8, 5.6, 5.5, 5.4, 5.2, 5.0, 4.8, 4.6, 4.4, 4.2, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.15, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.48, 0.46, 0.44, 0.42, 0.4, 0.38, 0.36, 0.34, 0.32, 0.3, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.095, 0.09, 0.085, 0.08, 0.075, 0.07, 0.065, 0.06, 0.055, 0.05]
    return piecewise_cal(values_x, values_y, resistance, log_x=True, log_y=True)
_all_calibrations.append(RuOx202a)

# Hall Probe
def HGCA3020(resistance):
    '''Calibration for a Hall Probe'''
    values_x = [-0.02895958, -0.01930098, -0.00964152, 0., 0.00963079, 0.01929507, 0.02898745]
    values_y = [-30., -20., -10., 0., 10., 20., 30]
    return piecewise_cal(values_x, values_y, resistance)
_all_calibrations.append(HGCA3020)


class ADMXMuxGetSpime(dragonfly.implementations.MuxerGetSpime):
    @calibrate(_all_calibrations)
    def on_get(self):
        # Note that these lines are a direct copy from the base class, I'm not sure how to undecorate
        #      when I try to change to the decorator here
        result = self.provider.send([self.get_str.format(self.ch_number)])
        logger.debug('very raw is: {}'.format(result))
        return result.split()[0]
