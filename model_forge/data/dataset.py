import pandas as pd
import numpy as np
from typing import Any, Optional, Callable, Union
from my_logger.custom_logger import logger


class Dataset(dict):
    """
    A class representing a dataset.

    This class extends the built-in `dict` class and provides additional functionality for working with datasets.

    Attributes:
        data_splitter: An optional data splitter object used to split the data into train and test sets.
        target_column: The name of the target column in the data.
        name: The name of the dataset.
        _is_data_splitted: A flag indicating whether the data has been split.
        data: The input data for the dataset.
        _X: The feature matrix X.
        _y: The target variable array.
        splits: A dictionary containing the splits of the dataset.

    Methods:
        X: Returns the feature matrix X.
        y: Returns the target variable array.
        columns: Returns the list of column names.
        shape: Returns the shape of the feature matrix X.
        _split_data: Splits the data into train and test sets.
        _run_checks: Runs checks on the splits to ensure data integrity.
        load_split: Loads a specific split of the dataset.
        load_train_test: Loads the training and testing data splits from the dataset.
        create_from_pipeline: Creates a dataset from a data loading function and optional data pipeline.
        create_from_splits: Creates a dataset from splits.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        data_splitter=None,
        target_column: str = "y",
        name: str = "dataset",
        splits_columns: list = None
    ) -> None:
        """
        Initialize a Dataset object.

        Args:
            data (pd.DataFrame): The input data for the dataset.
            data_splitter (optional): An optional data splitter object used to split the data into train and test sets.
            target_column (str): The name of the target column in the data.
            name (str): The name of the dataset.

        Returns:
            None
        """
        
        self.data_splitter = data_splitter
        self.target_column = target_column
        self.splits_columns = splits_columns
        self.name = name
        self._is_data_splitted = False
        self.data = data

        self._split_data()
        super().__init__(self.splits)

    @property
    def X(self) -> pd.DataFrame:
        """
        Returns the feature matrix X.
        
        Returns:
            pd.DataFrame: The feature matrix X.
        """
        return self['ALL'][0]

    @property
    def y(self) -> np.array:
        """
        Returns the target variable array.
        
        Returns:
            np.array: The target variable array.
        """
        return self['ALL'][1]

    @property
    def columns(self):
        """
        Returns a list of column names in the dataset.
        
        Returns:
            list: A list of column names.
        """
        return list(self.splits.values())[0][0].columns.tolist()

    @property
    def shape(self):
        """
        Returns the shape of the dataset.
        
        Returns:
            tuple: A tuple representing the shape of the dataset.
        """
        return self.X.shape
    



    def _split_data(self) -> None:
        """
        Split the data into train and test sets.

        This method splits the data into train and test sets based on the provided data splitter.
        If no data splitter is provided, it assumes all the data is the train set.

        Returns:
            None
        """
        self.splits = {}
        self.splits['ALL'] =  [True] * len(self.data)
        if self.splits_columns is not None:
            for column in self.splits_columns:
                self.splits[column] = list(self.data[column] == 1)




        self._is_data_splitted = True
        self._run_checks()
    
    def __getitem__(self, key: Any) -> Any:
        """
        Retrieve an item from the dataset.

        Args:
            key (Any): The key used to retrieve the item.

        Returns:
            Any: The item corresponding to the given key.
        """
        indexes = super().__getitem__(key)
        return (
                self.data.drop(columns=self.target_column)[indexes], 
                self.data[self.target_column][indexes]
            )

    def _run_checks(self) -> None:
        """
        Run checks on the splits of the dataset.

        Raises:
            AssertionError: If any of the splits is None, not a list, or empty.
        """
        for split_name, indexes in self.splits.items():
            assert indexes is not None, f"Split '{split_name}' is None"
            assert isinstance(indexes, list), f"Split '{split_name}' is not a list"
            assert len(indexes) != 0, f"Split '{split_name}' is empty"

    def __getattr__(self, __name: str) -> Any:
        """
        Retrieves the attribute specified by __name.

        Args:
            __name (str): The name of the attribute to retrieve.

        Returns:
            Any: The value of the attribute.

        Raises:
            AttributeError: If the attribute specified by __name is not found.
        """
        if __name.startswith(("X_", "y_")):
            _, split_name = __name.split("_", 1)
            if split_name in self.splits.keys():
                return (
                    self.splits[split_name][0]
                    if __name.startswith("X_")
                    else self.splits[split_name][1]
                )
        raise AttributeError(f"Attribute '{__name}' not found")

    def load_split(
        self,
        split: str,
        return_X_y: bool = False,
        sample_n_rows: Optional[int] = None,
        random_state: int = 36,
    ) -> Union[tuple[pd.DataFrame, np.array], pd.DataFrame]:
        """
        Load a specific split of the dataset.

        Args:
            split (str): The name of the split to load.
            return_X_y (bool, optional): Whether to return X and y separately. Defaults to False.
            sample_n_rows (int, optional): Number of rows to sample from the split. Defaults to None.
            random_state (int, optional): Random state for sampling rows. Defaults to 36.

        Returns:
            Union[tuple[pd.DataFrame, np.array], pd.DataFrame]: The loaded split of the dataset.
                If return_X_y is True, returns a tuple of X and y.
                If return_X_y is False, returns a DataFrame with X and y as columns.
        """

        if not self._is_data_splitted:
            self._split_data()
        if split not in self.splits.keys():
            raise ValueError(
                f"Invalid Split: You requested split '{split}'. Valid splits are: {*list(self.splits.keys()),} "
            )
        X, y = self[split][0], self[split][1]
        if sample_n_rows is not None:
            X = X.sample(sample_n_rows, random_state=random_state)
            y = y[X.index]

        if return_X_y:
            return X, y
        else:
            return X.assign(**{self.target_column: y})


    @classmethod
    def create_from_pipeline(
        cls,
        data_loading_function: Callable[[], pd.DataFrame],
        data_pipeline=None,
        data_splitter=None,
        target_column="y",
        name: str = "dataset",
    ):
        """
        Create a dataset from a data loading function and optional data pipeline.

        Args:
            cls: The class of the dataset.
            data_loading_function: A function that loads the data and returns a pandas DataFrame.
            data_pipeline: An optional data pipeline to apply to the loaded data.
            data_splitter: An optional data splitter to split the data into train and test sets.
            target_column: The name of the target column in the dataset.
            name: The name of the dataset.

        Returns:
            An instance of the dataset class.

        """
        data = data_loading_function()
        if data_pipeline:
            data = data_pipeline.apply(data)
        return cls(
            data=data,
            data_splitter=data_splitter,
            target_column=target_column,
            name=name,
        )

    @classmethod
    def create_from_splits(
        cls,
        splits: dict[str, tuple[pd.DataFrame, np.array]],
        name: str = "dataset",
        target_column: str = "y",
    ):
        """
        Create a dataset from splits.

        Args:
            cls (class): The class of the dataset.
            splits (dict[str, tuple[pd.DataFrame, np.array]]): A dictionary containing the splits of the dataset.
                Each split is represented as a tuple of a pandas DataFrame (X) and a numpy array (y).
            name (str, optional): The name of the dataset. Defaults to "dataset".
            target_column (str, optional): The name of the target column. Defaults to "y".

        Returns:
            dataset (cls): The created dataset.
        """
        Xs = []
        for split_name, (X, y) in splits.items():
            assert (
                target_column not in X.columns
            ), f"Split {split_name} already has a target column ({target_column}), please drop or rename"
            Xs.append(X.assign(y=y))
        fullX = pd.concat(Xs, ignore_index=True)

        dataset = cls(
            data=fullX, data_splitter=None, target_column=target_column, name=name
        )
        dataset._is_data_splitted = True
        dataset.splits = splits
        dataset._run_checks()
        return dataset

    