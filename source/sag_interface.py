import asteval

import dripline
import dragonfly

import logging
logger = logging.getLogger('dragonfly.custom.sag_interface')

class SAGCoordinator(dripline.core.Endpoint):
    '''
    Coordinated interactions with all instruments within the broader sag system.
    Provides a single point of contact and uniform interface to the SAG.
    '''
    def __init__(self, enable_output_sets=None, disable_output_sets=None, sag_injection_sets=None, switch_endpoint=None, extra_logs=[], state_extra_logs={}, **kwargs):
        '''
        enable_output_sets: (list) - a sequence of endpoints and values to set to configure the system to be ready to start output of a signal
        disable_output_sets: (list) - a sequence of endpoints and values to set to configure the system to not produce any output
        sag_injection_sets: (list) - a sequence of endpoints to set to create a particular injection; configuration determined from asteval of input values
        switch_endpoint: (string) - name of the endpoint used for switching the signal path into the weak port
        extra_logs: (list) - list of endpoint names to cmd `scheduled_action` (to trigger a log) whenever the SAG is configured
        state_extra_logs: (dict) - dict with keys being valid switch_endpoint states (string) and values being a list of extra sensors to log when entering that state.

        '''
        dripline.core.Endpoint.__init__(self, **kwargs)

        self.enable_output_sets = enable_output_sets
        self.disable_output_sets = disable_output_sets
        self.sag_injection_sets = sag_injection_sets
        self.switch_endpoint = switch_endpoint
        self.extra_logs = extra_logs
        self.state_extra_logs = state_extra_logs

        self.evaluator = asteval.Interpreter()

    def _do_set_collection(self, these_sets, values):
        '''
        A utility method for processing a list of sets
        '''
        set_list = []
        # first parse all string evaluations, make sure they all work before doing any actual setting
        for a_calculated_set in these_sets:
            if len(a_calculated_set) > 1:
                raise dripline.core.DriplineValueError('all calculated sets must be a single entry dict')
            this_endpoint,set_str = a_calculated_set.items()[0]
            logger.debug('trying to understand: {}->{}'.format(this_endpoint, set_str))
            this_set = set_str
            if '{' in set_str and '}' in set_str:
                try:
                    this_set = set_str.format(**values)
                except KeyError as e:
                    raise dripline.core.DriplineValueError("required parameter, <{}>, not provided".format(e.message))
            logger.debug('substitutions make that RHS = {}'.format(this_set))
            this_value = self.evaluator(this_set)
            logger.debug('or a set value of {}'.format(this_value))
            set_list.append((this_endpoint, this_value))
        # now actually try to set things
        for this_endpoint, this_value in set_list:
            #logger.info("if I weren't a jerk, I'd do:\n{} -> {}".format(this_endpoint, this_value))
            self.provider.set(this_endpoint, this_value)

    #def _do_log_noset_sensors(self):
    def _do_extra_logs(self, sensors_list):
        '''
        Send a scheduled_action (log) command to configured list of sensors (this is for making sure we log everything
        that should be recorded on each injection, but which is not already/automatically logged by a log_on_set action)
        '''
        logger.info('triggering logging of the following sensors: {}'.format(sensors_list))
        for a_sensor in sensors_list:
            self.provider.cmd(a_sensor, 'scheduled_action')

    def update_state(self, new_state):
        # do universal extra logs
        self._do_extra_logs(self.extra_logs)
        # do state-specific extra logs
        if new_state in self.state_extra_logs:
            self._do_extra_logs(self.state_extra_logs[new_state])
        else:
            logger.warning('state <{}> does not have a state-specific extra logs list, please create one (it may be empty)'.format(new_state))
        # actually set to the new state
        if new_state is 'term':
            self.do_disable_output_sets()
            self.provider.set(self.switch_endpoint, "term")
        elif new_state is 'sag':
            self.do_enable_output_sets()
            self.provider.set(self.switch_endpoint, "sag")
        elif new_state is 'vna':
            # set the switch
            self.provider.set(self.switch_endpoint, "vna")
            # disable outputs
            self.do_disable_output_sets()
        elif new_state is 'locking':
            raise dripline.core.DriplineValueError('locking state is not currently supported')
        else:
            raise dripline.core.DriplineValueErorr("don't know how to set the SAG state to <{}>".format(new_state))

    def do_enable_output_sets(self):
        logger.info('enabling lo outputs')
        self._do_set_collection(self.enable_output_sets, {})

    def do_disable_output_sets(self):
        logger.info('disabling lo outputs')
        self._do_set_collection(self.disable_output_sets, {})

    def configure_injection(self, **parameters):
        '''
        parameters: (dict) - keyworded arguments are available to all sensors as named substitutions when calibrating
        '''
        logger.info('in configure injection')
        # set to state 'sag' (which enables output)
        self.update_state("sag")
        # to extra sets calculated from input parameters
        self._do_set_collection(self.sag_injection_sets, parameters)
