from warnings import warn
import pandas as pd
import numpy as np


def markbinary(df, features=[]):
    '''
    Rename columns with only 1's and 0's as binary#[feature name]
    
    Parameters
    ----------
    df: dataframe
        Dataframe containing categorical variables to be onehot encoded
    features: list
        List containing column names to be marked as binary, empty will auto mark
    '''

    if features:
        for f in features:
            df.rename(columns={f: 'binary#' + f}, inplace=True)
    else:
        for f in df.columns:
            if not f.startswith('binary#') and df[f].value_counts().index.isin([0, 1]).all():
                df.rename(columns={f: 'binary#' + f}, inplace=True)


def onehotencode(df, features=[], impute='retain',
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
        Drop first binary column for each categorical column to remove collinearity
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
