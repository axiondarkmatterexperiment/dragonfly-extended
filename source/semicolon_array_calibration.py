from dripline.core import calibrate
import logging
logger = logging.getLogger('dragonfly.implementations.custom')


_all_calibrations = []

#suppose you're given a semicolon separated list of either floats or arrays of floats
#seperate them out and label them
def semicolon_array(data_string,label_array):
    to_return="{"
    split_strings=data_string.split(';')
    if len(split_strings)<len(label_array):
        raise dripline.core.DriplineValueError("not enough values given to fill semicolon_array")
    for i in range(len(label_array)):
        if i!=0:
            to_return+=','
        if "," in split_strings[i]: 
            #must be an array (TODO if you want to handle strings, this must be changed)
            to_return+='"'+label_array[i]+'": ['+split_strings[i]+']'
        else:
            #must just be a regular float, or maybe a string with no comma
            to_return+='"'+label_array[i]+'": '+split_strings[i]
    to_return+="}"
    return to_return
_all_calibrations.append(semicolon_array)
            


