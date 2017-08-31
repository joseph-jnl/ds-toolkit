from warnings import warn
import pandas as pd
import numpy as np


def normalize(df, features=[], inplace=False):
    '''
    Normalize columns to [0, 1]

    Parameters
    ----------
    df: dataframe
        Dataframe with columns to convert
    features: list
        List of column names to standardize, if empty
        use all non binary, non categorical columns
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

    # Standardize non binary columns
    if not features:
        features = [s for s in list(dfm) if not s.startswith(
            'binary#') and dfm[s].dtype not in ['object']]

    dfm[features] = (dfm[features] - dfm[features].min()) / (dfm[features].max() -dfm[features].min())

    return dfm


def standardize(df, features = [], inplace = False):
    '''
    Standardize columns to mean 0, unit variance

    Parameters
    ----------
    df: dataframe
        Dataframe with columns to convert
    features: list
        List of column names to standardize, if empty
        use all non binary, non categorical columns
    inplace: boolean, False (default)
        Modify dataframe in place

    Return
    ------
    dfm = modified dataframe
    '''

    if inplace:
        dfm=df
    else:
        dfm=df.copy()

    # Standardize non binary columns
    if not features:
        features=[s for s in list(dfm) if not s.startswith(
            'binary#') and dfm[s].dtype not in ['object']]

    dfm[features] = (dfm[features] - dfm[features].mean()) / dfm[features].std()

    return dfm


def num_to_str(df, features, inplace = False):
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
        dfm=df
    else:
        dfm=df.copy()

    for f in features:
        dfm.loc[:, f]=dfm.loc[:, f].astype(str)

    if not inplace:
        return dfm


def nan_to_binary(df, features = [], prefix = True, inplace = False, threshold = 0.3):
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
        dfm=df
    else:
        dfm=df.copy()

    if not features:
        features=dfm.loc[:, dfm.isnull().sum() > dfm.shape[0]
                                       * threshold].columns.tolist()

    for f in features:
        dfm.loc[: , f] = dfm.loc[: , f].isnull().astype(int)
        if prefix:
            dfm.rename(columns = {f: 'binary#' + f}, inplace =True)

    if not inplace:
        return dfm


def mark_binary(df, features = [], inplace =False):
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
            dfm.rename(columns = {f: 'binary#' + f}, inplace =True)
    else:
        # Auto mark columns with only 1 or 0
        for f in df.columns:
            if not f.startswith('binary#') and dfm[f].value_counts().index.isin([0, 1]).all():
                dfm.rename(columns = {f: 'binary#' + f}, inplace =True)

    if not inplace:
            return dfm


def onehot_encode(df, features = [], impute ='retain',
                  first = True, sparse =False, tracknan=True, dropzerovar=True):
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
    dfc=df.copy()

    # Create prefix: binary#[categorical level label]
    if features:
        prefixes=['binary#' + s for s in features]
    else:
        features=df.select_dtypes(
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


def impact_encode(df, target, features=[], type='', probs={}, dropzerovar=True):
    '''
    Wrapper function for impact/conditional probability encoding 
    categorical variables
    
    For categorical variable X
    minmaxscaled( P(Y=1 | X = x) )
    
    e.g. 
    color = ['green', red', 'red', 'blue', 'red', 'green', 'green']
    target = [0, 1, 1, 1, 0, 1, 0]
    
    P(target=1 | X=red ) -> 0.67 minmax-> 0.5
    color = [red', 'red', 'red]
    target = [1, 1, 0]

    P(target=1 | X=green ) -> 0.33 minmax-> 0 
    color = ['green', 'green', 'green']
    target = [0, 1, 0]

    P(target=1 | X=blue ) -> 1 minmax-> 1
    color = ['blue']
    target = [1]
        
    or

    todo: add later
    
    P(X<= x | Y = 1)
    omega = [a,b,..] where a, b.. is P(target=1 | X=x )
    
    e.g.
    X = [0.33, 0.67, 1]
    color = ['red', 'red', 'blue', 'green']
    target = [1,1,1,1]

    P(X<=red | target = 1)
    P(X<=2/4 | target = 1) -> 0.33

    P(X<=green | target = 1) 
    P(X<=1/4 | target = 1) -> 0

    P(X<=blue | target = 1)
    P(X<=1/4 | target = 1) -> 0



    Parameters
    ----------
    df: dataframe
        Dataframe containing categorical variables to be onehot encoded
    target: str or pandas series
        column name of target class, or series containing target class
    features: list
        List containing column names to be encoded, empty will encode all 
        object and category dtype columns
    probs: dict
        Dictionary containing previous probabilities of categorical vars,
        if empty, will use current data
    dropzerovar: boolean, True (default)
        Drop columns with 0 variance

    Return
    ------- 
    Modified dataframe
    '''
    dfc = df.copy()


    # Create prefix: normalized#[categorical level label]
    if features:
        prefixes = ['normalized#' + s for s in features]
    else:
        features = df.select_dtypes(
            include=['object', 'category']).columns
        prefixes = ['normalized#' + s for s in features]


    # Conditional probability encode
    for f in features:
        cmin = df.loc[df[target]==1, f].value_counts().divide(df.loc[:,f].value_counts()).fillna(1).min()
        cmax = df.loc[df[target]==1, f].value_counts().divide(df.loc[:,f].value_counts()).fillna(1).max()
        probs[f] = df.loc[df[target]==1, f].value_counts().divide(df.loc[:,f].value_counts()).fillna(1).subtract(cmin).divide(cmax-cmin).to_dict() 
        for c in probs[f]:
            dfc[f] = df[f].transform(lambda x: np.NaN if pd.isnull(x) else probs[f][x])

    # Add prefix to column names
    dfc = dfc.rename(columns={f: prefix for f, prefix in zip(features, prefixes)})
    # print({f: prefix for f, prefix in zip(features, prefixes)})

    if dropzerovar:
        zero_var_columns = dfc.var() == 0
        dfc.drop(zero_var_columns[zero_var_columns == True].index.tolist(),
                 axis=1,
                 inplace=True)

    return dfc
