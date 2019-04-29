import dripline
import dragonfly

import logging
logger = logging.getLogger('dragonfly.implementations.custom')

class AG34980ARepeater(dragonfly.implementations.RepeaterProvider):
    '''
    '''

    def send(self, *args, **kwargs):
        full_result = dragonfly.implementations.RepeaterProvider.send(self, *args, **kwargs)
        return full_result['values'][0]
