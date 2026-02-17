# task_runner.py

import threading
import queue

class BackgroundTaskRunner:
    """
    A generic class to run any target function in a background daemon thread.
    """
    def __init__(self, target_func, *args, **kwargs):
        """
        Initializes the runner with the function to execute and its arguments.
        
        Args:
            target_func: The function to run in the background.
            *args: Positional arguments for the target function.
            **kwargs: Keyword arguments for the target function.
        """
        self._target_func = target_func
        self._args = args
        self._kwargs = kwargs
        self._result_queue = queue.Queue()
        self._thread = None

    def _run_target(self):
        """
        Internal wrapper that executes the target function and puts the
        outcome (result or exception) into the queue.
        """
        try:
            result = self._target_func(*self._args, **self._kwargs)
            self._result_queue.put(("success", result))
        except Exception as e:
            # Store the actual exception object to preserve its type and traceback
            self._result_queue.put(("error", e))

    def start(self):
        """
        Starts the background thread.
        """
        self._thread = threading.Thread(target=self._run_target, daemon=True)
        self._thread.start()

    def is_running(self):
        """
        Returns True if the background thread is currently running.
        """
        return self._thread and self._thread.is_alive()

    def get_result(self):
        """
        Retrieves the result from the completed task.
        
        If the task completed successfully, returns its result.
        If the task raised an exception, this method re-raises that exception.
        
        Raises:
            RuntimeError: If called while the task is still running or if the
                          result queue is unexpectedly empty.
            Exception: The original exception from the background task.
        """
        if self.is_running():
            raise RuntimeError("Cannot get result while task is still running.")
        try:
            status, data = self._result_queue.get_nowait()
            if status == "success":
                return data
            else:  # 'error'
                raise data  # Re-raise the original exception
        except queue.Empty:
            raise RuntimeError("Task finished, but the result queue is empty.")