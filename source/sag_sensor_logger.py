import dripline
import dragonfly

import re
import six

import logging
logger = logging.getLogger("dragonfly.implementations.custom")

class GenSensorLogger(dragonfly.implementations.SensorLogger):
    def __init__(self, log_match_key='sensor_value.#', sensor_type_map_table=None, sensor_type_column_name='type', sensor_type_match_column='endpoint_name', data_tables_dict={}, **kwargs):
        '''
        log_match_key (string): routing key binding for messages to process
        '''
        #TODO the log_match_key should be treated more carefully; We need both a pattern for binding, and a pattern for extracting the <from> field, would be nice to do this all together in some clean way
        keys = kwargs.get('keys', [])
        if keys:
            logger.warning("GenSensorLogger is overriding CLI 'keys' option including default; must use 'log_match_key' config file value")
        #if isinstance(keys, six.string_types):
        #    keys = [keys]
        self.log_match_key = log_match_key
        #keys.append(log_match_key)
        keys = [log_match_key]
        kwargs.update({'keys': keys})
        logger.warning("about to recurse inits, initially keys are: {}".format(kwargs['keys']))
        dragonfly.implementations.PostgreSQLInterface.__init__(self, **kwargs)
        dripline.core.Gogol.__init__(self, **kwargs)
        logger.warning("after those inits, bindings are: {}".format(self._bindings))

        #sensor_type_map_conf['table_name'] = sensor_type_map_conf['table_name']
        #sensor_type_map_conf['name_column'] = sensor_type_map_conf.get('name_column', 'endpoint_name')
        #sensor_type_map_conf['type_column'] = sensor_type_map_conf.get('type_column', 'type')
        if sensor_type_map_table is None:
            raise dripline.core.DriplineValueError("sensor_type_map_table is required config value")
        self._sensor_type_map_table = sensor_type_map_table
        self._sensor_type_column_name = sensor_type_column_name
        self._sensor_type_match_column = sensor_type_match_column
        self._sensor_types = {}
        self._data_tables = data_tables_dict
        self.service = self

    def this_consume(self, message, basic_deliver):
        ### Get the sensor name
        sensor_name = None
        if '.' in basic_deliver.routing_key:
            re_out = re.match(r'{}.(?P<from>\S+)'.format(self.log_match_key.rstrip('.#')), basic_deliver.routing_key)
            if re_out is None:
                logger.warning("received routing key <{}> does not match sensor name extraction".format(basic_deliver.routing_key))
                return
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
            this_data_table = self.endpoints[self._data_tables[self._sensor_types[sensor_name]]]

            ### Log the sensor value
            insert_data = {'endpoint_name': sensor_name,
                           'timestamp': message['timestamp'],
                          }
            for key in ['value_raw', 'value_cal', 'memo']:
                if key in message.payload:
                    insert_data[key] = message.payload[key]
            this_data_table.do_insert(**insert_data)
            logger.info('value logged for <{}>'.format(sensor_name))
