import random
import time

import distributed

from prob_unet import dask_utils


def custom_sleep(time_delay):
    time.sleep(time_delay)
    return time_delay


def test_serial_execution():
    class CustomDaskRunner(dask_utils.DaskRunner):
        def init_tasks_to_do(self):
            self.tasks_to_do = {j: None for j in range(100)}

        def next_submit(self):
            for key, _ in self.tasks_to_do.items():
                yield {'key': key,
                       'fn': custom_sleep,
                       'args': (0.001,),
                       'kwargs': {}}
    runner = CustomDaskRunner()
    runner.serial_execute()
    assert len(runner.tasks_done) == 100
    assert len(runner.tasks_result) == 100
    assert len(runner.tasks_to_do) == 0
    assert len(runner.tasks_submitted) == 0
    assert runner.tasks_result == {j: 0.001 for j in range(100)}
    assert runner.total_execution_time > 0.1


def test_parallel_execution():
    class CustomDaskRunner(dask_utils.DaskRunner):
        def init_tasks_to_do(self):
            self.tasks_to_do = {j: None for j in range(100)}

        def next_submit(self):
            for key, _ in self.tasks_to_do.items():
                yield {'key': key,
                       'fn': custom_sleep,
                       'args': (0.1,),
                       'kwargs': {}}
    runner = CustomDaskRunner(dask_utils.DaskRunnerConfig(num_workers=4))
    runner.parallel_execute()
    assert len(runner.tasks_done) == 100
    assert len(runner.tasks_result) == 100
    assert len(runner.tasks_to_do) == 0
    assert len(runner.tasks_submitted) == 0
    assert runner.tasks_result == {j: 0.1 for j in range(100)}
    assert runner.total_execution_time < 10


def test_parallel_execution_timed():
    custom_sleep_timed = dask_utils.TimedFunction(custom_sleep)

    class CustomDaskRunner(dask_utils.DaskRunner):
        def init_tasks_to_do(self):
            self.tasks_to_do = {j: None for j in range(100)}

        def next_submit(self):
            for key, _ in self.tasks_to_do.items():
                yield {'key': key,
                       'fn': custom_sleep_timed,
                       'args': (0.1,),
                       'kwargs': {}}
    runner = CustomDaskRunner(config=dask_utils.DaskRunnerConfig(num_workers=4))
    runner.execute()
    assert len(runner.tasks_done) == 100
    assert len(runner.tasks_result) == 100
    assert len(runner.tasks_to_do) == 0
    assert len(runner.tasks_submitted) == 0
    assert runner.tasks_result == {j: 0.1 for j in range(100)}
    assert runner.total_execution_time < 10
    assert runner.delta_t_saved > 0
    assert runner.performance_gain > 1


def crash_fn(x):
    if x == random.randint(2, 9):
        import os
        getattr(os, '._exit')(1)
    return x ** 2


def test_parallel_execution_crash():
    # Dask automatically retries on crashes!
    class CustomDaskRunner(dask_utils.DaskRunner):
        def init_tasks_to_do(self):
            self.tasks_to_do = {j: None for j in range(100)}

        def next_submit(self):
            for key, _ in self.tasks_to_do.items():
                yield {'key': key,
                       'fn': crash_fn,
                       'args': (key,),
                       'kwargs': {}}
    runner = CustomDaskRunner(config=dask_utils.DaskRunnerConfig(num_workers=4))
    runner.execute()
    assert len(runner.tasks_done) == 100


def scheduler_connection_lost(x):
    if x == random.randint(2, 9):
        raise distributed.client.FutureCancelledError(key="task", reason="Dummy loss of connection")
    return x ** 2


def test_parallel_execution_scheduler_connection_lost():
    class CustomDaskRunner(dask_utils.DaskRunner):
        def init_tasks_to_do(self):
            self.tasks_to_do = {j: None for j in range(100)}

        def next_submit(self):
            for key, _ in self.tasks_to_do.items():
                yield {'key': key,
                       'fn': scheduler_connection_lost,
                       'args': (key,),
                       'kwargs': {}}
    runner = CustomDaskRunner(config=dask_utils.DaskRunnerConfig(num_workers=4))
    runner.execute()
    assert len(runner.tasks_done) == 100
