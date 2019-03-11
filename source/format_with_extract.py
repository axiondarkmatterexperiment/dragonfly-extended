import re


import dripline
import dragonfly

import logging
logger = logging.getLogger('dragonfly.implementations.custom')

class FormatWithExtract(dragonfly.implementations.FormatSpime):
    '''
    An extension of the FormatSpime to support extracting the value_raw from a more complex reply,
    based on use of python's re library. The result from a get will be processed with re.search(extract_raw_regex, raw_result);
    the extract_raw_regex must be constructed with an extraction group keyed with the name "value_raw" (ie r'(?P<value_raw>)' )
    '''

    def __init__(self, extract_raw_regex=None, **kwargs):
        dragonfly.implementations.FormatSpime.__init__(self, **kwargs)
        self._extract_raw_regex = extract_raw_regex

    @dripline.core.calibrate()
    def on_get(self):
        if self._get_str is None:
            raise DriplineMethodNotSupportedError('<{}> has no get string available'.format(self.name))
        logger.info('level 0 get')
        first_result = self.provider.send([self._get_str])
        logger.debug('initial result is: {}'.format(first_result))
        matches = re.search(self._extract_raw_regex, first_result)
        if matches is None:
            logger.error('matching returned none')
            raise dripline.core.DriplineValueError('returned result [{}] has no match to input regex [{}]'.format(first_result, self._extract_raw_regex))
        logger.info("matches are: {}".format(matches.groupdict()))
        result = matches.groupdict()['value_raw']
        if self._get_reply_float:
            logger.debug('desired format is: float')
            formatted_result = map(float, re.findall("[-+]?\d+\.\d+",format(result)))
            # formatted_result = map(float, re.findall("[-+]?(?: \d* \. \d+ )(?: [Ee] [+-]? \d+ )",format(result)))
            logger.debug('formatted result is {}'.format(formatted_result[0]))
            return formatted_result[0]
        return result
