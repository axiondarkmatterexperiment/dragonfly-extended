import math
from dripline.core import calibrate
from dripline.core import Spime
import dripline
import dragonfly
import yaml

import logging
logger = logging.getLogger('dragonfly.implementations.custom')

_all_calibrations = []


def semicolon_array(data_string,label_array):
#TODO I'm sure there is a real json library that will do this for me
    to_return="{"
    split_strings=data_string.split(';')
    if len(split_strings)<len(label_array):
        raise dripline.core.DriplineValueError("not enough values given to fill semicolon_array")
    for i in range(len(label_array)):
        if i!=0:
            to_return+=','
        if "," in split_strings[i]: 
            #must be an array (TODO if you want to handle strings with commas, this must be changed)
            to_return+='"'+label_array[i]+'": ['+split_strings[i]+']'
        else:
            #must just be a regular float, or maybe a string with no comma
            to_return+='"'+label_array[i]+'": '+split_strings[i]
    to_return+="}"
    return to_return

_all_calibrations.append(semicolon_array)

class MultiFormatSpime(Spime):
    def __init__(self,
            get_commands=None,
            set_commands=None,
            **kwargs):
        Spime.__init__(self,**kwargs)
        self._get_commands=get_commands
        self._set_commands=set_commands
        logger.debug("end here")

    @calibrate(_all_calibrations)
    def on_get(self):
        if self._get_commands is None:
            raise DriplineMethodNotSupportedError('<{}> has no get commands available'.format(self.name))
        to_send=""
        get_labels=[]
        for i in range(len(self._get_commands)):
            if i!=0:
                to_send=to_send+";"
            to_send=to_send+self._get_commands[i]["get_str"]
            get_labels.append(self._get_commands[i]["label"])
        result=self.provider.send([to_send])
        return semicolon_array(result,get_labels)
    
    def on_set(self,value):
        if self._set_commands is None:
            raise DriplineMethodNotSupportedError('<{}> has no set commands available'.format(self.name))
        try:
            value_structure=yaml.safe_load(value)
        except yaml.YAMLError as ecx:
            raise DriplineValueError('<{}> had error {}'.format(self.name,exc))
        to_send=""
        for command in self._set_commands:
            if command["label"] in value_structure:
                if len(to_send)>0:
                    to_send=to_send+";"
                to_send+="{} {}".format(command["set_str"],value_structure[command["label"]])
        to_send=to_send+";*OPC?"
        return self.provider.send([to_send])
