from warnings import warn
import pandas as pd
import numpy as np


def num_to_str(df, features, inplace=False):
    '''
    Convert numeric columns to str to mark them as categorical

    Parameters
    ----------
    df: dataframe
        Dataframe with columns to convert
    features: list
        List of column names to convert to str
    inplace: boolean, False (default)
        Modify dataframe in place   

    Return
    ------
    dfm = modified dataframe
    '''

    if inplace:
        dfm = df
    else:
        dfm = df.copy()

    for f in features:
        dfm[f] = dfm[f].astype(str)

    return dfm


def nan_to_binary(df, features=[], prefix=True, inplace=False, threshold=0.3):
    '''
    Convert columns with large amount of NaN's to a binary column
    indicating which rows were NaN.

    Parameters
    ----------
    df: dataframe
        Dataframe with columns to convert
    features: list
        List of column names to convert to binary, 1 if NaN, 0 any other
        If empty, auto convert columns where NaN > 30%
    prefix: boolean, True (default)
        rename column to binary#[feature]
    inplace: boolean, False (default)
        Modify dataframe in place   

    Return
    ------
    dfm = modified dataframe
    '''

    if inplace:
        dfm = df
    else:
        dfm = df.copy()

    if not features:
        features = dfm.loc[:, dfm.isnull().sum() > dfm.shape[0] * threshold].columns.tolist()

    for f in features:
        dfm[f] = dfm[f].isnull().astype(int)
        if prefix:
            dfm.rename(columns={f: 'binary#' + f}, inplace=True)
    return dfm


def mark_binary(df, features=[], inplace=False):
    '''
    Rename columns with only 1's and 0's as binary#[feature name]

    Parameters
    ----------
    df: dataframe
        Dataframe containing categorical variables to be onehot encoded
    features: list
        List containing column names to be marked as binary, empty will auto mark
    inplace: boolean
        Modify dataframe in place

    Return
    ------
    dfm = modified dataframe
    '''
    if inplace:
        dfm = df
    else:
        dfm = df.copy()

    if features:
        for f in features:
            dfm.rename(columns={f: 'binary#' + f}, inplace=True)
    else:
        # Auto mark columns with only 1 or 0
        for f in df.columns:
            if not f.startswith('binary#') and dfm[f].value_counts().index.isin([0, 1]).all():
                dfm.rename(columns={f: 'binary#' + f}, inplace=True)

    return dfm


def onehot_encode(df, features=[], impute='retain',
                  first=True, sparse=False, tracknan=True, dropzerovar=True):
    '''
    Wrapper function for one hot encoding categorical variables

    Parameters
    ----------
    df: dataframe
        Dataframe containing categorical variables to be onehot encoded
    features: list
        List containing column names to be encoded, empty will encode all 
        object and category dtype columns
    impute: str, 'retain' (default) or 'mode'
        Retain NaN's or impute with mode
    first: boolean, True (default)
        Drop first binary column for each categorical column
        to remove collinearity
    sparse: boolean False (default)
        Use sparse matrix
    tracknan: boolean, True (default)
        Include column tracking rows that were NaN
    dropzerovar: boolean, True (default)
        Drop columns with 0 variance

    Return
    ------- 
    Modified dataframe
    '''
    dfc = df.copy()

    # Create prefix: binary#[categorical level label]
    if features:
        prefixes = ['binary#' + s for s in features]
    else:
        features = df.select_dtypes(
            include=['object', 'category']).columns
        prefixes = ['binary#' + s for s in features]

    # Check how many categorical levels
    total_levels = sum([df[f].unique().size for f in features])
    if total_levels > 100 and not sparse:
        warn('{} categorical levels found, recommend using sparse matrix or feature selection'.format(
            total_levels))

    # Impute using mode if specified
    if impute == 'mode':
        dfc[features] = dfc[features].fillna(dfc[features].mode().iloc[0])

    # One hot encode with pd.get_dummies()
    dfc = pd.get_dummies(dfc,
                         prefix=prefixes,
                         drop_first=first,
                         columns=features,
                         sparse=sparse,
                         dummy_na=tracknan)

    if impute == 'retain':
        if not tracknan:
            raise ValueError('tracknan must be True to retain nans')
        for f in features:
            flabels = [s for s in list(dfc) if s.startswith(
                'binary#' + f) and not s.endswith('_nan')]
            fnanlabel = 'binary#' + f + '_nan'
            dfc.loc[dfc[fnanlabel] == 1, flabels] = np.nan

    if dropzerovar:
        zero_var_columns = dfc.var() == 0
        dfc.drop(zero_var_columns[zero_var_columns == True].index.tolist(),
                 axis=1,
                 inplace=True)

    return dfc
