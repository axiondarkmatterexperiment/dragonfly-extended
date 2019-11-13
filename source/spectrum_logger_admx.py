'''
This file is a slight hack on the Project 8 version, which was previously insufficiently generic for our (ADMX's) needs.
That has been fixed and there are bugs here resolved by P8, we revert to their version.
'''

from __future__ import absolute_import

# standard libs
import logging
import re

# internal imports
from dripline.core import exceptions
from dragonfly.implementations.sensor_logger import SensorLogger

__all__ = []
logger = logging.getLogger('dragonfly.custom.sensor_logger_admx')

__all__.append('SensorLoggerADMX')
class SensorLoggerADMX(SensorLogger):
    '''
    Slow control table implementation
    '''

    def this_consume(self, message, basic_deliver):
        logger.debug("consuming message to: {}".format(basic_deliver.routing_key))
        ### Get the sensor name
        if not basic_deliver.routing_key.split('.')[0] == self.prefix:
            logger.warning("should not consume this message")
            return
        sensor_name = None
        if '.' in basic_deliver.routing_key:
            re_out = re.match(r'{}.(?P<from>\S+)'.format(self.prefix), basic_deliver.routing_key)
            sensor_name = re_out.groupdict()['from']
        # note that the following is deprecated in dripline 2.x, retained for compatibility
        else:
            raise exceptions.DriplineValueError('unknown sensor name')

        ### Get the type and table for the sensor
        this_type = None
        this_table = self.endpoints[self._sensor_type_map_table]
        this_type = this_table.do_select(return_cols=[self._sensor_type_column_name],
                                         where_eq_dict={self._sensor_type_match_column:sensor_name},
                                        )
        if not this_type[1]:
            logger.critical('endpoint with name "{}" was not found in database hence failed to log its value; might need to add it to the db'.format(sensor_name))
        else:
            this_table = self.endpoints[self._sensor_type_map_table]
            this_type = this_table.do_select(return_cols=[self._sensor_type_column_name],
                                             where_eq_dict={self._sensor_type_match_column:sensor_name},
                                            )
            self._sensor_types[sensor_name] = this_type[1][0][0]
            if not self._sensor_types[sensor_name] in self._data_tables:
                logger.critical('endpoint with name "{}" is not configured with a recognized type in the sensors_list table'.format(sensor_name))
                logger.critical('sensor type is {}'.format(self._sensor_types[sensor_name]))
                logger.critical('data tables: {}'.format(self._data_tables))
                return
            this_data_table = self.endpoints[self._data_tables[self._sensor_types[sensor_name]]]

            ### Log the sensor value
            insert_data = {'endpoint_name': sensor_name,
                           'timestamp': message['timestamp'],
                          }
            insert_data.update(message.payload)
            this_data_table.do_insert(**insert_data)
            logger.info('value logged for <{}>'.format(sensor_name))
