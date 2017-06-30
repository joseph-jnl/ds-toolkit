import numpy as np
import pandas as pd
import dstk
import random

# Create test data

# Class creation test dataset
df_create = pd.DataFrame()
df_create['numeric1'] = np.random.randint(100, size=(100))
df_create['numeric2'] = np.random.uniform(-1, 1, size=(100))
df_create['numericNaN'] = np.random.uniform(-1, 1, size=(100))
df_create.loc[np.random.choice(100, 20), 'numericNaN'] = None
df_create['cat1'] = [random.choice(['a', 'b']) for i in range(100)]
df_create['catNaN'] = [random.choice(['a', 'b', None]) for i in range(100)]


def test_find_missingval_features():
    # Test whether all columns with missing values are found
    mv = dstk.MissingValues(df_create)
    assert set(list(mv.features.keys())) == set(['numericNaN', 'catNaN'])


def test_ignore():
    # Test features defined as ignored
    mv = dstk.MissingValues(df_create, ignore=['catNaN'])
    assert set(list(mv.features.keys())) == set(['numericNaN'])


def test_id():
    # Test features defined as identifiers
    mv = dstk.MissingValues(df_create, identifier=['numericNaN'])
    assert mv.features['numericNaN'].ftype == 'ID'


def test_categorical():
    # Test features defined as identifiers
    mv = dstk.MissingValues(df_create, categorical=['numericNaN'])
    assert mv.features['numericNaN'].ftype == 'Categorical'