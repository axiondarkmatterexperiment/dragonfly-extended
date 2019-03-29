'''
A service fo interfacing with the DAQ DB (the run table in particular)

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
import collections
import six

# local imports
from dripline.core import Provider, Endpoint, fancy_doc, constants
from dripline.core.exceptions import *
from dragonfly.implementations import SQLTable

import logging
#logger = logging.getLogger(__name__)
logger = logging.getLogger('dragonfly.implementations.custom')
logger.setLevel(logging.DEBUG)

__all__.append("UpsertTable")


@fancy_doc
class UpsertTable(SQLTable):
    '''
    A class for making calls to _insert_with_return
    '''
    def __init__(self,
                 do_upsert=False,
                 ignore_keys=[],
                 *args,
                **kwargs):
        '''
        do_upsert (bool): indicates if conflicting inserts should then update
        ignore_keys (list of strings): list of names of payload keys which should not be mapped to columns on an insert, even if present
        '''
        if not 'sqlalchemy' in globals():
            raise ImportError('SQLAlchemy not found, required for SQLTable class')
        self.ignore_keys = ignore_keys
        Endpoint.__init__(self, *args, **kwargs)
        SQLTable.__init__(self, *args, **kwargs)
        self.do_upsert = do_upsert

    def _insert_with_return(self, insert_kv_dict, return_col_names_list):
        try:
            #ins = self.table.insert().values(**insert_kv_dict)
            ins = sqlalchemy.dialects.postgresql.insert(self.table).values(**insert_kv_dict)
            if return_col_names_list:
                logger.debug('adding return clause')
                ins = ins.returning(*[self.table.c[col_name] for col_name in return_col_names_list])
            if self.do_upsert:
                p_keys = [key.name for key in sqlalchemy.inspection.inspect(self.table).primary_key]
                #update_d = {c.name:c for c in ins.excluded if not c.primary_key}
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
        kwargs = {k:v for k,v in kwargs.items() if not k in self.ignore_keys}
    #    print('in derived, but then do the parent version of insert')
    #    print("received the follow args:\n    {}".format(args))
    #    print('received the following keys:\n    {}'.format(kwargs.keys()))
        return SQLTable.do_insert(self, *args, **kwargs)
    #    if not isinstance(self.provider, PostgreSQLInterface):
    #        raise DriplineInternalError('InsertDBEndpoint must have a RunDBInterface as provider')
    #    # make sure that all provided insert values are expected
    #    for col in kwargs.keys():
    #        if not col in self._column_map.keys():
    #            #raise DriplineDatabaseError('not allowed to insert into: {}'.format(col))
    #            logger.warning('got an unexpected insert column <{}>'.format(col))
    #            kwargs.pop(col)
    #    # make sure that all required columns are present
    #    for col in self._required_insert_names:
    #        if not col['payload_key'] in kwargs.keys():
    #            raise DriplineDatabaseError('a value for <{}> is required!\ngot: {}'.format(col, kwargs))
    #    # build the insert dict
    #    this_insert = self._default_insert_dict.copy()
    #    this_insert.update({self._column_map[key]:value for key,value in kwargs.items()})
    #    return_vals = self._insert_with_return(this_insert, self._return_names,)
    #    return return_vals


