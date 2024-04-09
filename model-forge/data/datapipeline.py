import pandas as pd
from copy import deepcopy
import inspect

from functools import wraps
from my_logger.custom_logger import logger


def safe(fn):
    """
    A decorator that creates a safe version of the decorated function.
    The safe version of the function makes a deep copy of the arguments
    and keyword arguments before calling the original function.
    This ensures that the original arguments are not modified during the function call.

    Args:
        fn (function): The function to be decorated.

    Returns:
        function: The safe version of the decorated function.
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        cp_args = deepcopy(args)
        cp_kwargs = deepcopy(kwargs)
        res = fn(self, *cp_args, **cp_kwargs)
        return res

    return wrapper


class PandasDataPipeline:
    """
    A data pipeline class that applies a series of steps to a pandas DataFrame.

    Args:
        steps (list): A list of functions or tuples (description, function) representing the steps to be applied.
        name (str, optional): The name of the pipeline. Defaults to "pipeline".

    Attributes:
        steps (list): A list of functions or tuples (description, function) representing the steps to be applied.
        name (str): The name of the pipeline.

    Methods:
        apply(df: pd.DataFrame) -> pd.DataFrame:
            Applies the pipeline steps to the given DataFrame.

    """

    def __init__(
        self,
        steps,
        name: str = "pipeline",
    ) -> None:
        self.steps = steps
        self.name = name

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the pipeline steps to the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to apply the steps to.

        Returns:
            pd.DataFrame: The DataFrame after applying the steps.

        Raises:
            TypeError: If a step function does not accept a pandas DataFrame as an argument.

        """
        for step_number, step in enumerate(self.steps, start=0):
            if isinstance(step, tuple):
                # If step is a tuple, assume it's (description, function)
                _, step_func = step
            else:
                step_func = step

            # Check if step_func expects a pandas DataFrame as its argument
            if not self._function_accepts_dataframe(step_func):
                raise TypeError(
                    f"The step function at step {step_number} does not accept a pandas DataFrame as an argument."
                )
            logger.info(step_func)
            # Apply the step
            df = step_func(df)

        return df

    def _function_accepts_dataframe(self, func):
        """Check if first argument op function expects pd.DataFrame"""
        sig = inspect.signature(func)
        params = sig.parameters.values()
        first_param = next(iter(params), None)
        return first_param and first_param.annotation is pd.DataFrame

    @safe
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the pipeline steps to the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to apply the steps to.

        Returns:
            pd.DataFrame: The DataFrame after applying the steps.

        """
        return self._apply(df)
