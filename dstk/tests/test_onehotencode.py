import numpy as np
import pandas as pd
from dstk.preprocessing import onehotencode


# Create test data

df = pd.DataFrame()
df['numeric1'] = [0, 1, 2, 3, 4, 5]
df['numeric2'] = [1.0, 3.4, 5.4, 2.3, 3.1, 4.1]
df['numericNaN'] = [1, 2, 3, None, 3, None]
df['cat1'] = ['a', 'a', 'b', 'c', 'c', 'a']
df['catNaN'] = ['A', 'B', None, None, 'B', 'C']


def test_checkprefix():
    # Test whether prefixes are created correctly
    test = onehotencode(df)
    assert test.columns.tolist() == ['numeric1',
                                     'numeric2',
                                     'numericNaN',
                                     'binary#cat1_b',
                                     'binary#cat1_c',
                                     'binary#catNaN_B',
                                     'binary#catNaN_C',
                                     'binary#catNaN_nan']


def test_select_feature():
    # Test whether subselection of features is correct
    test = onehotencode(df, features=['cat1'])
    assert test.columns.tolist() == ['numeric1',
                                     'numeric2',
                                     'numericNaN',
                                     'catNaN',
                                     'binary#cat1_b',
                                     'binary#cat1_c']


def test_retainNaNs():
    # Test whether nans are retained
    test = onehotencode(df, impute='retain')
    assert np.isnan(test['binary#catNaN_B']).tolist() == [
        False, False, True, True, False, False]


def test_modeimputeNaNs():
    # Test mode imputing NaNs
    test = onehotencode(df, impute='mode')
    assert test['binary#catNaN_B'].tolist() == [0, 1, 1, 1, 1, 0]


def test_trackNaNs():
    # Test whether nans are tracked in separate column
    test = onehotencode(df)
    assert test['binary#catNaN_nan'].tolist() == [0, 0, 1, 1, 0, 0]


def test_drop_zerovar():
    # Test whether zero variance columns are dropped
    df['cat2'] = ['a', 'a', 'a', 'a', 'a', 'a']
    test = onehotencode(df)
    assert test.columns.tolist() == ['numeric1',
                                     'numeric2',
                                     'numericNaN',
                                     'binary#cat1_b',
                                     'binary#cat1_c',
                                     'binary#catNaN_B',
                                     'binary#catNaN_C',
                                     'binary#catNaN_nan']
