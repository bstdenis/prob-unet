import logging
import time
import traceback
from dataclasses import dataclass

from dask.distributed import CancelledError, Client, LocalCluster

from prob_unet.logging_utils import CustomLogging, FunctionWithOwnRootLogger, readable_delta_t

logger = CustomLogging(caller=logging.getLogger(__name__))


@dataclass(frozen=True, slots=True)
class DaskRunnerConfig:
    num_workers: int = 1
    worker_memory_limit: int | str | None = None
    max_submitted: int = 256
    check_status_time_delay: int = 0
    task_timeout: int | None = None
    log_repeat_thresholds: dict | None = None
    heartbeat_interval: int | None = None
    parallel: bool = True


class TimedFunction:
    def __init__(self, fn):
        self.fn = fn
        self.time_start = None
        self.time_end = None
        self.time_elapsed = None
        self.result = None

    def __call__(self, *args, **kwargs):
        self.time_start = time.time()
        self.result = self.fn(*args, **kwargs)
        self.time_end = time.time()
        self.time_elapsed = self.time_end - self.time_start
        return self


class DaskRunner:
    """A class to manage Dask parallel execution of tasks with a flexible configuration.

    Parameters
    ----------
    config : DaskRunnerConfig
        DaskRunner configurable parameters. See the `DaskRunnerConfig` dataclass for details.
    init_tasks_to_do_kwargs : dict
        Keyword arguments for the `init_tasks_to_do` method, which initializes the tasks to be executed.

    Attributes
    ----------
    n_workers : int
    """
    def __init__(self, config=None, init_tasks_to_do_kwargs=None):
        if config is None:
            config = DaskRunnerConfig()
        self.tasks_to_do = {}
        self.tasks_submitted = {}
        self.tasks_done = {}
        self.tasks_result = {}

        self.n_workers = config.num_workers
        self.worker_memory_limit = config.worker_memory_limit
        self.max_submitted = config.max_submitted
        self.task_timeout = config.task_timeout
        self.heartbeat_interval = config.heartbeat_interval
        self.parallel = config.parallel

        self.check_status_time_delay = config.check_status_time_delay

        self.total_execution_time = None
        self.use_timed_function = False
        self.delta_t_saved = None
        self.performance_gain = None

        self.log_repeat_thresholds = {} if config.log_repeat_thresholds is None else config.log_repeat_thresholds
        self.log_repeat_thresholds['max_submitted'] = self.log_repeat_thresholds.get('max_submitted', 86400)
        self.log_repeat_thresholds['nb_retrieved'] = self.log_repeat_thresholds.get('nb_retrieved', 10)
        self.log_repeat_thresholds['nb_stats'] = self.log_repeat_thresholds.get('nb_stats', 10)

        init_tasks_to_do_kwargs = {} if init_tasks_to_do_kwargs is None else init_tasks_to_do_kwargs
        self.init_tasks_to_do(**init_tasks_to_do_kwargs)

    @property
    def nb_of_tasks(self):
        return len(self.tasks_to_do) + len(self.tasks_done)

    def init_tasks_to_do(self, **kwargs):
        # To be implemented in child class, should populate self.tasks_to_do with a dictionary
        pass

    def next_submit(self):
        # To be implemented in child class, should be based on what's left in self.tasks_to_do
        for key, value in self.tasks_to_do.items():
            yield {'key': key,
                   'fn': lambda x: x,
                   'args': (),
                   'kwargs': {}}

    def collate_partial(self):
        # To be implemented in child class, will be called at the end of each status check
        pass

    def collate(self):
        # To be implemented in child class, will be called at the end of all tasks execution. Default
        # implementation is to call collate_partial once more.
        self.collate_partial()

    def submit_futures(self, client):
        submit_iterator = self.next_submit()
        while self.tasks_to_do:
            submit_args = next(submit_iterator, None)
            if submit_args is None:
                break
            if submit_args['key'] in self.tasks_submitted:
                continue
            self.tasks_submitted[submit_args['key']] = {
                'future': client.submit(submit_args['fn'], *submit_args['args'], **submit_args['kwargs']),
                'submission_time': time.time()}
            if len(self.tasks_submitted) >= self.max_submitted:
                break

    def check_task_status(self, key, task_submitted):
        retrieved = False
        future = task_submitted['future']
        if 'estimated_start_time' in self.tasks_submitted[key]:
            if (self.task_timeout is not None) and (not future.done()):
                if time.time() - self.tasks_submitted[key]['estimated_start_time'] > self.task_timeout:
                    for task_submitted in self.tasks_submitted.values():
                        task_submitted['future'].cancel()
                    logger.critical(f'Future execution timed out ({self.task_timeout} s).')
                    raise TimeoutError(f'Future execution timed out ({self.task_timeout} s).')
        elif ('estimated_start_time' not in self.tasks_submitted[key]) and (future.status != 'pending'):
            self.tasks_submitted[key]['estimated_start_time'] = time.time()
        if future.done():
            try:
                result = future.result()
            except Exception as e:
                logger.critical("".join(traceback.format_exception(e)))
                for task_submitted in self.tasks_submitted.values():
                    task_submitted['future'].cancel()
                raise e
            del self.tasks_to_do[key]
            submission_time = task_submitted['submission_time']
            estimated_start_time = self.tasks_submitted[key]['estimated_start_time']
            now = time.time()
            self.tasks_done[key] = {'retrieved_time': now}
            if isinstance(result, TimedFunction):
                self.tasks_done[key]['delta_t_fn_execution'] = result.time_elapsed
                self.tasks_done[key]['delta_t_parallel_overhead'] = now - estimated_start_time - result.time_elapsed
                self.use_timed_function = True
                result = result.result
            if isinstance(result, FunctionWithOwnRootLogger):
                self.tasks_done[key]['log_file'] = result.log_file
                result = result.result
            self.tasks_result[key] = result
            self.tasks_done[key]['delta_t_submission_to_start'] = estimated_start_time - submission_time
            self.tasks_done[key]['delta_t_start_to_retrieved'] = now - estimated_start_time

            log_msg = f'Task {key} done ({len(self.tasks_done)}/{self.nb_of_tasks})'
            logger.debug(log_msg, block_short_repetition_delay=self.log_repeat_thresholds['nb_retrieved'],
                         identifier='nb_retrieved', expected_nb_of_calls=self.nb_of_tasks, add_eta=True)

            retrieved = True
        return retrieved

    def check_tasks_status(self):
        keys_to_delete = []
        for key, task_submitted in self.tasks_submitted.items():
            retrieved = self.check_task_status(key, task_submitted)
            if retrieved:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del self.tasks_submitted[key]

    def parallel_execute(self):
        start_time = time.time()
        memory_limit_str = '' if self.worker_memory_limit is None else f' ({self.worker_memory_limit} per worker)'
        num_restart = 0
        while self.tasks_to_do:
            logger.debug(f"Starting LocalCluster Client with {self.n_workers} workers{memory_limit_str}")
            cluster = LocalCluster(n_workers=self.n_workers, threads_per_worker=1,
                                   memory_limit=self.worker_memory_limit)
            client = Client(cluster, heartbeat_interval=self.heartbeat_interval)
            while self.tasks_to_do or self.tasks_submitted:
                log_msg = f'Nb of pending tasks: {len(self.tasks_to_do)}'
                if len(self.tasks_to_do):
                    log_msg += f' ({len(self.tasks_to_do) - len(self.tasks_submitted)} to be submitted)'
                logger.debug(log_msg, block_short_repetition_delay=self.log_repeat_thresholds['nb_stats'],
                             identifier='nb_stats')
                if self.tasks_to_do and (len(self.tasks_submitted) < self.max_submitted):
                    self.submit_futures(client)
                if self.tasks_submitted:
                    try:
                        self.check_tasks_status()
                    except CancelledError:
                        num_restart += 1
                        logger.warning(f'CancelledError caught, retrying (count: {num_restart})...')
                        self.tasks_submitted = {}
                        client.close()
                        cluster.close()
                        break
                    except Exception as e:
                        # Fallback for cases where Dask wrapped it as a plain Exception.
                        # Detect the “stringified” cancellation.
                        msg = str(e)
                        if "FutureCancelledError" in msg or "CancelledError" in msg:
                            num_restart += 1
                            logger.warning(f'Wrapped CancelledError caught ({msg}), retrying (count: {num_restart})...')
                            self.tasks_submitted = {}
                            client.close()
                            cluster.close()
                            break
                    self.collate_partial()
                time.sleep(self.check_status_time_delay)
            self.collate()
        self.total_execution_time = time.time() - start_time

        if self.use_timed_function:
            total_fn_execution_time = sum([self.tasks_done[key]['delta_t_fn_execution'] for key in self.tasks_done])
            self.delta_t_saved = -(self.total_execution_time - total_fn_execution_time)
            self.performance_gain = total_fn_execution_time / self.total_execution_time
            logger.debug(f'Performance gain: {self.performance_gain:.2f}, '
                         f'Time saved: {readable_delta_t(self.delta_t_saved)}, '
                         f'Total time: {readable_delta_t(self.total_execution_time)}')

    def serial_execute(self):
        # For debugging purposes
        start_time = time.time()
        for submit_args in self.next_submit():
            key = submit_args['key']
            task_start_time = time.time()
            result = submit_args['fn'](*submit_args['args'], **submit_args['kwargs'])
            self.tasks_done[key] = {'estimated_start_time': task_start_time, 'retrieved_time': time.time()}
            if isinstance(result, TimedFunction):
                self.tasks_done[key]['delta_t_fn_execution'] = result.time_elapsed
                self.tasks_done[key]['delta_t_parallel_overhead'] = 0
                self.tasks_result[key] = result.result
                self.use_timed_function = True
            else:
                self.tasks_result[key] = result
            self.tasks_done[key]['delta_t_submission_to_start'] = 0
            self.tasks_done[key]['delta_t_start_to_retrieved'] = time.time() - task_start_time

            log_msg = f'Task {key} done ({len(self.tasks_done)}/{self.nb_of_tasks})'
            logger.debug(log_msg, block_short_repetition_delay=self.log_repeat_thresholds['nb_retrieved'],
                         identifier='nb_retrieved', expected_nb_of_calls=self.nb_of_tasks, add_eta=True)
        self.tasks_to_do = {}
        self.collate()
        self.total_execution_time = time.time() - start_time

    def execute(self, parallel=True):
        if parallel is None:
            parallel = self.parallel
        if parallel:
            self.parallel_execute()
        else:
            self.serial_execute()
