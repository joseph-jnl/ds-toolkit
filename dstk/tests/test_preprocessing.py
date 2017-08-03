import numpy as np
import pandas as pd
from dstk.preprocessing import (onehot_encode,
                                mark_binary,
                                nan_to_binary,
                                num_to_str)

# Create test data

df = pd.DataFrame()
df['numeric1'] = [0, 1, 0, 0, 1, 1]
df['numeric2'] = [1.0, 3.4, 5.4, 2.3, 3.1, 4.1]
df['numericNaN'] = [1, 2, 3, None, 3, None]
df['cat1'] = ['a', 'a', 'b', 'c', 'c', 'a']
df['catNaN'] = ['A', 'B', None, None, 'B', 'C']


# Test for num_to_str function

def test_numtostr():
    # Test for converting column type to object
    test = num_to_str(df, ['numeric1'])
    assert test['numeric1'].dtype == 'O'


def test_numtostr_inplace():
    # Test for converting column to object in place
    df2 = df.copy()
    num_to_str(df2, ['numeric1'], inplace=True)
    assert df2['numeric1'].dtype == 'O'


# Tests for nan_to_binary function

def test_nantobinary_inplaceTrue():
    # Test for converting dataframe in place
    df2 = df.copy()
    nan_to_binary(df2, ['numericNaN'], inplace=True)
    assert df2['binary#numericNaN'].tolist() == [0, 0, 0, 1, 0, 1]


def test_nantobinary_featureselect():
    # Test for converting specified features
    test = nan_to_binary(df, ['numericNaN'])
    assert test['binary#numericNaN'].tolist() == [0, 0, 0, 1, 0, 1]


def test_nantobinary_auto():
    # Test for auto converting columns with NaN > threshold
    test = nan_to_binary(df)
    assert test['binary#catNaN'].tolist() == [0, 0, 1, 1, 0, 0]


def test_nantobinary_threshold():
    # Test for auto converting columns with NaN > specified threshold
    test = nan_to_binary(df, threshold=0.5, inplace=False)
    assert test.loc[2, 'catNaN'] == None


# Tests for markbinary function

def test_markbinary_inplaceFalse():
    # Test for not transforming df in place
    test = mark_binary(df, inplace=False)
    assert test.columns.tolist()[0] == 'binary#numeric1'


def test_markbinary_inplaceTrue():
    # Test for transforming df in place
    df2 = df.copy()
    mark_binary(df2, inplace=True)
    assert df2.columns.tolist()[0] == 'binary#numeric1'


def test_markbinary_inplaceTrue_selectfeature():
    # Test for selecting specific features to mark 
    df2 = df.copy()
    mark_binary(df2, ['numeric1'], inplace=True)
    assert df2.columns.tolist()[0] == 'binary#numeric1'


# Tests for onehotencode wrapper

def test_onehot_checkprefix():
    # Test whether prefixes are created correctly
    test = onehot_encode(df)
    assert test.columns.tolist() == ['numeric1',
                                     'numeric2',
                                     'numericNaN',
                                     'binary#cat1_b',
                                     'binary#cat1_c',
                                     'binary#catNaN_B',
                                     'binary#catNaN_C',
                                     'binary#catNaN_nan']


def test_onehot_selectfeature():
    # Test whether subselection of features is correct
    test = onehot_encode(df, features=['cat1'])
    assert test.columns.tolist() == ['numeric1',
                                     'numeric2',
                                     'numericNaN',
                                     'catNaN',
                                     'binary#cat1_b',
                                     'binary#cat1_c']


def test_onehot_retainNaNs():
    # Test whether nans are retained
    test = onehot_encode(df, impute='retain')
    assert np.isnan(test['binary#catNaN_B']).tolist() == [
        False, False, True, True, False, False]


def test_onehot_modeimputeNaNs():
    # Test mode imputing NaNs
    test = onehot_encode(df, impute='mode')
    assert test['binary#catNaN_B'].tolist() == [0, 1, 1, 1, 1, 0]


def test_onehot_trackNaNs():
    # Test whether nans are tracked in separate column
    test = onehot_encode(df)
    assert test['binary#catNaN_nan'].tolist() == [0, 0, 1, 1, 0, 0]


def test_onehot_drop_zerovar():
    # Test whether zero variance columns are dropped
    df['cat2'] = ['a', 'a', 'a', 'a', 'a', 'a']
    test = onehot_encode(df)
    assert test.columns.tolist() == ['numeric1',
                                     'numeric2',
                                     'numericNaN',
                                     'binary#cat1_b',
                                     'binary#cat1_c',
                                     'binary#catNaN_B',
                                     'binary#catNaN_C',
                                     'binary#catNaN_nan']
