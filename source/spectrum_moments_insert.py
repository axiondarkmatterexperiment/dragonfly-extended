'''
A service for computing statistical moments of spectra, and logging to the doubles table

Note: services using this module will require sqlalchemy (and assuming we're still using postgresql, psycopg2 as the sqlalchemy backend)
'''

from __future__ import absolute_import
__all__ = []

# std libraries
import json
import os
import types
import traceback

# 3rd party libraries
try:
    import sqlalchemy
except ImportError:
    pass
from datetime import datetime
from itertools import groupby
import numpy
import collections
import six

# local imports
from dripline.core import Provider, Endpoint, fancy_doc, constants
from dripline.core.exceptions import *
from dragonfly.implementations import SQLTable

import logging
logger = logging.getLogger('dragonfly.implementations.custom')
logger.setLevel(logging.DEBUG)

__all__.append("UpsertTable")


@fancy_doc
class SpectrumMomentsInsert(SQLTable):
    '''
    A class for making calls to _insert_with_return
    '''
    def __init__(self,
                 conditional_insert_field,
                 spectrum_field,
                 sensor_name_mean,
                 sensor_name_std,
                 do_upsert=False,
                 *args,
                **kwargs):
        '''
        do_upsert (bool): indicates if conflicting inserts should then update
        conditional_insert_field (string): name of an input kwarg field on inserts to cast to bool and determine if a log should be inserted
                                           default is true if the field is missing
        spectrum_field (string): name of an input kwarg field on inserts which contains an iterable of values for which moments will be calculated
        sensor_name_mean (string): name of the sensor to log with the mean of the spectrum
        sensor_name_std (string): name of the sensor to log with the STD of the spectrum
        '''
        if not 'sqlalchemy' in globals():
            raise ImportError('SQLAlchemy not found, required for SQLTable class')
        self.conditional_insert_field = conditional_insert_field
        self.spectrum_field = spectrum_field
        self.sensor_name_mean = sensor_name_mean
        self.sensor_name_std = sensor_name_std
        Endpoint.__init__(self, *args, **kwargs)
        SQLTable.__init__(self, *args, **kwargs)
        self.do_upsert = do_upsert

    def _insert_with_return(self, insert_kv_dict, return_col_names_list):
        try:
            ins = sqlalchemy.dialects.postgresql.insert(self.table).values(**insert_kv_dict)
            if return_col_names_list:
                logger.debug('adding return clause')
                ins = ins.returning(*[self.table.c[col_name] for col_name in return_col_names_list])
            if self.do_upsert:
                p_keys = [key.name for key in sqlalchemy.inspection.inspect(self.table).primary_key]
                update_d = {k:v for k,v in insert_kv_dict.items() if not k in p_keys}
                ins = ins.on_conflict_do_update(index_elements=p_keys, set_=update_d)
            insert_result = ins.execute()
            if return_col_names_list:
                return_values = insert_result.first()
            else:
                return_values = []
        except Exception as err:
            if str(err).startswith('(psycopg2.IntegrityError)'):
                raise DriplineDatabaseError(str(err))
            else:
                logger.critical('received an unexpected SQL error while trying to insert:\n{}'.format(str(ins) % insert_kv_dict))
                logger.info('traceback is:\n{}'.format(traceback.format_exc()))
                raise
        return dict(zip(return_col_names_list, return_values))

    def do_insert(self, *args, **kwargs):
        '''
        '''
        if kwargs.pop(self.conditional_insert_field, False):
            logger.info("not logging because power_measurement suppressed")
            return
        the_spectrum = kwargs.pop(self.spectrum_field)
        this_mean = numpy.mean(the_spectrum)
        this_std = numpy.std(the_spectrum)
        logger.info("other keys are: {}".format(kwargs.keys()))
        logger.info("timestamp: {}".format(kwargs['timestamp']))
        logger.info("computed a mean of: {}".format(this_mean))
        logger.info("computed an std of: {}".format(this_std))
        SQLTable.do_insert(self, *args, **{'timestamp': kwargs['timestamp'],
                                         'sensor_name': self.sensor_name_mean,
                                         'raw_value': this_mean,
                                         'calibrated_value': this_mean,
                                        })
        SQLTable.do_insert(self, *args, **{'timestamp': kwargs['timestamp'],
                                         'sensor_name': self.sensor_name_std,
                                         'raw_value': this_std,
                                         'calibrated_value': this_std,
                                        })

