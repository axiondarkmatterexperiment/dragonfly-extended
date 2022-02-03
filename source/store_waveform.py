from __future__ import absolute_import
import logging

from dripline.core import Provider, Spime, calibrate, fancy_doc

__all__ = ['StoreWaveform']

logger = logging.getLogger(__name__)

@fancy_doc
class StoreWaveform(Spime):
    """
    Endpoint for storing SAG waveform segments
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
            self._value = []
        return self._value
