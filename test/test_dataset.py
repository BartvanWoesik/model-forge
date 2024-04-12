import pytest
import pandas as pd
import numpy as np
from model_forge.data.dataset import Dataset

@pytest.fixture
def dataset():
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [6, 7, 8, 9, 10],
        'train': [0, 0, 1, 1, 1],
        'target': [0, 0, 1, 1, 1]
    })
    return Dataset(data=data, target_column='target', splits_columns=['train'])

def test_x_retrieval(dataset):
    expected_X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [6, 7, 8, 9, 10],
        'train': [0, 0, 1, 1, 1]
    })
    print(dataset.X)
    print(expected_X)
    assert isinstance(dataset.X, pd.DataFrame)
    assert dataset.X.equals(expected_X)

def test_y(dataset):
    expected_y = np.array([0, 0, 1, 1, 1])
    assert np.array_equal(dataset.y, expected_y)

def test_shape(dataset):
    expected_shape = (5, 3)
    assert dataset.shape == expected_shape

def test_load_split(dataset):
    split_name = 'train'
    X, y = dataset.load_split(split=split_name, return_X_y=True)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0]


def test_create_from_pipeline():
    def data_loading_function():
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10],
            'target': [0, 0, 1, 1, 1]
        })
    
    dataset = Dataset.create_from_pipeline(data_loading_function=data_loading_function, target_column='target')
    assert isinstance(dataset, Dataset)
    assert dataset.shape == (5, 2)

def test_x_train_attribute_retrival(dataset):
    expected_X_train = pd.DataFrame({
        'feature1': [3, 4, 5],
        'feature2': [8, 9, 10],
        'train': [1, 1, 1]
    })

    assert dataset.X_train.reset_index(drop = True).equals(expected_X_train.reset_index(drop = True))

def test_missing_split_attribute(dataset):
  
    with pytest.raises(AttributeError):
        dataset.X_test

def test_missing_general_attribute(dataset):
  
    with pytest.raises(AttributeError):
        dataset.v