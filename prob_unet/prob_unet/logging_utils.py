import logging
import time
from pathlib import Path
from pprint import pformat

logger = logging.getLogger(__name__)


def readable_delta_t(delta_t):
    """Return a human-readable string of a time delta.

    Args:
        delta_t (float): The time delta in seconds.

    Returns:
        str: A human-readable string of the time delta.
    """

    if delta_t < 0.001:
        return f'{round(delta_t * 1e6)} µs'
    elif delta_t < 1:
        return f'{round(delta_t * 1000)} ms'
    elif delta_t < 60:
        return f'{round(delta_t)} s'
    elif delta_t < 300:
        return f'{int(delta_t / 60)} min {round(delta_t % 60)} s'
    elif delta_t < 3600:
        return f'{round(delta_t / 60)} min'
    elif delta_t < 86400:
        return f'{int(delta_t / 3600)} h {round((delta_t % 3600) / 60)} min'
    else:
        return f'{int(delta_t / 86400)} d {round((delta_t % 86400) / 3600)} h'


def readable_value(value, expected_min=-1e38, expected_max=1e38):
    """Return a human-readable string of a value.

    Args:
        value (float): The value.
        expected_min (float): The expected minimum value.
        expected_max (float): The expected maximum value.

    Returns:
        str: A convenient human-readable string of the value.
    """

    if (value < expected_min) or (value > expected_max):
        return f'{value:.1e}'
    elif (int(value) == value) and abs(value) < 1000000:
        return f'{int(value)}'
    elif abs(value) < 0.01:
        return f'{value:.4e}'
    elif abs(value) < 100:
        return f'{value:.4f}'
    elif abs(value) < 10000:
        return f'{value:.2f}'
    elif abs(value) < 1000000:
        return f'{value:.0f}'
    else:
        return f'{value:.4e}'


class CustomLogging:
    def __init__(self, caller=None, queue=None, quick_repetition_tolerance=1):
        if caller is None:
            self.caller = logging
        else:
            self.caller = caller
        self.queue = queue
        self.last_log_messages = {}
        self.quick_repetition_tolerance = quick_repetition_tolerance
        self.quick_repetition_warning = False

    def log(self, caller, message, block_short_repetition_delay=0, identifier='', stacklevel=0,
            expected_nb_of_calls=1, add_eta=False):
        if (block_short_repetition_delay or add_eta) and (identifier not in self.last_log_messages):
            self.last_log_messages[identifier] = {'count': 1, 'first_time': time.time(),
                                                  'last_time': 0, 'quick_repetition_count': 0}
        elif add_eta:
            self.last_log_messages[identifier]['count'] += 1
        if (not block_short_repetition_delay) and (identifier in self.last_log_messages):
            self.last_log_messages[identifier]['last_time'] = 0
            self.last_log_messages[identifier]['quick_repetition_count'] = 0
        elif identifier in self.last_log_messages:
            if time.time() - self.last_log_messages[identifier]['last_time'] < block_short_repetition_delay:
                if self.last_log_messages[identifier]['quick_repetition_count'] >= self.quick_repetition_tolerance:
                    if not self.quick_repetition_warning:
                        self.caller.warning('Some logging messages are being blocked due to short time repetition')
                        self.quick_repetition_warning = True
                    return False
        if add_eta:
            if self.last_log_messages[identifier]['count'] == 1:
                eta_time_str = 'unknown'
            elif expected_nb_of_calls <= self.last_log_messages[identifier]['count']:
                eta_time_str = 'done'
            else:
                elapsed_time = time.time() - self.last_log_messages[identifier]['first_time']
                time_per_call = elapsed_time / (self.last_log_messages[identifier]['count'] - 1)
                time_left = time_per_call * (expected_nb_of_calls - self.last_log_messages[identifier]['count'])
                eta_time_str = readable_delta_t(time_left)
            message += f' (ETA: {eta_time_str})'
        caller(message, stacklevel=stacklevel + 3)
        if block_short_repetition_delay:
            self.last_log_messages[identifier]['last_time'] = time.time()
            self.last_log_messages[identifier]['quick_repetition_count'] += 1
        if self.queue is not None:
            self.queue.put((identifier, message))
        return True

    def debug(self, message, block_short_repetition_delay=0, identifier=None, stacklevel=0,
              expected_nb_of_calls=1, add_eta=False):
        return self.log(self.caller.debug, message, block_short_repetition_delay=block_short_repetition_delay,
                        identifier=identifier, stacklevel=stacklevel, expected_nb_of_calls=expected_nb_of_calls,
                        add_eta=add_eta)

    def info(self, message, block_short_repetition_delay=0, identifier=None, stacklevel=0,
             expected_nb_of_calls=1, add_eta=False):
        return self.log(self.caller.info, message, block_short_repetition_delay=block_short_repetition_delay,
                        identifier=identifier, stacklevel=stacklevel, expected_nb_of_calls=expected_nb_of_calls,
                        add_eta=add_eta)

    def warning(self, message, block_short_repetition_delay=0, identifier=None, stacklevel=0,
                expected_nb_of_calls=1, add_eta=False):
        return self.log(self.caller.warning, message, block_short_repetition_delay=block_short_repetition_delay,
                        identifier=identifier, stacklevel=stacklevel, expected_nb_of_calls=expected_nb_of_calls,
                        add_eta=add_eta)

    def error(self, message, block_short_repetition_delay=0, identifier=None, stacklevel=0,
              expected_nb_of_calls=1, add_eta=False):
        return self.log(self.caller.error, message, block_short_repetition_delay=block_short_repetition_delay,
                        identifier=identifier, stacklevel=stacklevel, expected_nb_of_calls=expected_nb_of_calls,
                        add_eta=add_eta)

    def critical(self, message, block_short_repetition_delay=0, identifier=None, stacklevel=0,
                 expected_nb_of_calls=1, add_eta=False):
        return self.log(self.caller.critical, message, block_short_repetition_delay=block_short_repetition_delay,
                        identifier=identifier, stacklevel=stacklevel, expected_nb_of_calls=expected_nb_of_calls,
                        add_eta=add_eta)


def default_basic_config_args(basic_config_args, show_logger_name=False, show_date=True):
    basic_config_args_with_defaults = {k: v for k, v in basic_config_args.items()}
    if 'format' not in basic_config_args_with_defaults:
        if show_logger_name:
            basic_config_args_with_defaults['format'] = \
                '%(asctime)s[%(levelname)s] (%(funcName)s, %(name)s) %(message)s'
        else:
            basic_config_args_with_defaults['format'] = '%(asctime)s[%(levelname)s] (%(funcName)s) %(message)s'
    if 'datefmt' not in basic_config_args_with_defaults:
        if show_date:
            basic_config_args_with_defaults['datefmt'] = '%Y-%m-%dT%H:%M:%S'
        else:
            basic_config_args_with_defaults['datefmt'] = '%H:%M:%S'
    if 'level' not in basic_config_args_with_defaults:
        basic_config_args_with_defaults['level'] = logging.DEBUG
    if 'force' not in basic_config_args_with_defaults:
        basic_config_args_with_defaults['force'] = True
    return basic_config_args_with_defaults


def start_root_logger(basic_config_args=None, show_logger_name=True, show_date=True, show_loggers_on_init=False,
                      disable_loggers=None, templates=None):
    basic_config_args = {} if basic_config_args is None else basic_config_args
    basic_config_args = default_basic_config_args(basic_config_args, show_logger_name, show_date)
    disable_loggers = [] if disable_loggers is None else disable_loggers
    delete_filename = False
    if 'filename' not in basic_config_args:
        basic_config_args['filename'] = templates['log_file']
        delete_filename = True
    path_log_file = basic_config_args['filename']
    Path(path_log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(**basic_config_args)
    if delete_filename:
        del basic_config_args['filename']
    logger.debug('Root logger started.')
    if show_loggers_on_init:
        formatted_logger_dict = pformat(sorted(list(logging.root.manager.loggerDict.keys())), compact=True)
        logger.debug(f'Root logger override. loggerDict keys: \n{formatted_logger_dict}')
    for disable_logger in disable_loggers:
        for name, known_logger in logging.root.manager.loggerDict.items():
            if (name[0:len(disable_logger)] == disable_logger) and (hasattr(known_logger, 'disabled')):
                known_logger.disabled = True
    return path_log_file


class FunctionWithOwnRootLogger:
    def __init__(self, fn, basic_config_args=None, templates=None, show_loggers_on_init=False,
                 show_logger_name=False, show_date=True, disable_loggers=None, substitutes_key=None,
                 identifier=None, queue=None):
        self.fn = fn
        if basic_config_args is None:
            basic_config_args = {}
        self.basic_config_args = default_basic_config_args(
            basic_config_args, show_logger_name=show_logger_name, show_date=show_date)
        self.templates = templates
        self.show_loggers_on_init = show_loggers_on_init
        if disable_loggers is None:
            disable_loggers = []
        self.disable_loggers = disable_loggers
        self.substitutes_key = substitutes_key
        self.identifier = identifier
        self.queue = queue
        self.log_file = None
        self.result = None

    def __call__(self, *args, **kwargs):
        delete_filename = False
        if 'filename' not in self.basic_config_args:
            if self.substitutes_key in kwargs:
                self.templates.add_substitutes(**kwargs[self.substitutes_key])
            self.basic_config_args['filename'] = self.templates['log_file']
            delete_filename = True
        Path(self.basic_config_args['filename']).parent.mkdir(parents=True, exist_ok=True)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        logging.basicConfig(**self.basic_config_args)
        self.log_file = self.basic_config_args['filename']
        if self.show_loggers_on_init:
            formatted_logger_dict = pformat(sorted(list(logging.root.manager.loggerDict.keys())), compact=True)
            logger.debug(f'Root logger override. loggerDict keys: \n{formatted_logger_dict}')
        if self.queue is not None:
            self.queue.put((self.identifier, self.basic_config_args['filename']))
        if delete_filename:
            del self.basic_config_args['filename']
        for disable_logger in self.disable_loggers:
            for name, known_logger in logging.root.manager.loggerDict.items():
                if (name[0:len(disable_logger)] == disable_logger) and (hasattr(known_logger, 'disabled')):
                    known_logger.disabled = True
        if (self.substitutes_key is not None) and (self.substitutes_key in kwargs):
            del kwargs[self.substitutes_key]
        self.result = self.fn(*args, **kwargs)
        return self


class LoggingFrequencySequence:
    def __init__(self, log_iterations=None, delays=None, use_default=False):
        self.log_iterations = [0] if log_iterations is None else log_iterations
        self.delays = [0] if delays is None else delays
        self.current_log_iteration = 0
        self.current_sequence_index = 0
        if use_default:
            self.set_default()
        if len(self.log_iterations) != len(self.delays):
            raise RuntimeError('log_iterations and delays must have the same length.')

    def set_default(self):
        self.log_iterations = [3, 10, 1000, 10000, 100000]
        self.delays = [0, 5, 10, 60, 300]

    def current_delay(self):
        return self.delays[self.current_sequence_index]

    def next_delay(self):
        self.current_log_iteration += 1
        if self.current_log_iteration > self.log_iterations[self.current_sequence_index]:
            if self.current_sequence_index < len(self.log_iterations) - 1:
                self.current_sequence_index += 1
        return self.current_delay()
