from __future__ import absolute_import
import logging

from dripline.core import Provider, Spime, calibrate, fancy_doc

__all__ = ['StoreWaveform']

logger = logging.getLogger(__name__)

# @fancy_doc
# class kv_store(Provider):
#     """
#     The KV store.  This is just a wrapper around a dict.
#     """
#     def __init__(self, **kwargs):
#         Provider.__init__(self, **kwargs)

#     def endpoint(self, endpoint):
#         """
#         Return the endpoint associated with some key.
#         """
#         return self.endpoints[endpoint]

#     def list_endpoints(self):
#         """
#         List all endpoints associated with this KV store.
#         This is the same as enumerating the keys in the
#         dict.
#         """
#         return self.keys()

#     def send(self, to_send):
#         logger.info('asked to send:\n{}'.format(to_send))


#@fancy_doc
class StoreWaveform(Spime):
    """
    Endpoint for storing SAG waveform segments
    A key in the KV store.
    """
    def __init__(self, initial_value=None, **kwargs):
        Spime.__init__(self, **kwargs)
        self._value = initial_value
        self.get_value = self.on_get

    def on_get(self):
        """
        Return the value associated with this
        key.
        """
        value = self._value
        return value

    def on_set(self, value):
        """
        Set the value associated with this key
        to some new value.
        """
        try:
            value = list(value)
            self._value = value
        except ValueError:
            self._value = value
        return self._value
