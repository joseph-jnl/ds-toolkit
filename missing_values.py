"""
Summarize, plot, and impute missing values found in data 
"""

# Authors: Joseph Lee <joseph.nw.lee@gmail.com>
# License: MIT

import numpy as np
import pandas as pd


class Feature(object):
    """
    Feature with missing values

    Parameters
    ----------
    feature: Pandas series
    
    Attributes
    ----------
    name: string
    ftype: string >> 'Categorical', 'Numeric', 'Date', 'Unknown'
    data: numpy Series
    missing_index: int list >> [0, 1, 4, 100, 123]  
    """

    def __init__(self, feature):
        self.name = feature.name
        if feature.dtype in ['object', 'bool']:
            self.ftype = 'Categorical'
        elif str(feature.dtype).startswith(('float', 'int')):
            self.ftype = 'Numeric'
        elif str(feature.dtype).startswith(('dateime', 'timedelta')):
            self.ftype = 'Date'
        else:
            self.ftype = 'Unknown'
        self.data = feature
        self.missing_index = feature[feature.isnull()].index


class MissingValues(object):
    """
    Missing Values

    Summarize, plot and impute features with missing values

    Parameters
    ----------
    df: input pandas dataframe

    optional:
    categorical = manually specify string array of feature columns that are categorical
                  ex. >> categorical = ['browser', 'country']
    identifier = manually specify string array of feature columns that are identifiers
                  ex >> identifier = ['id', 'name']
    ignore = specify columns to ignore

    Attributes
    ----------
    features: dict containing feature class with missing values
    df: input pandas data frame
    """

    def __init__(self, df, **kwargs):
        self.df = df
        self.features = {}
        for feature in df.columns:
            if df[feature].isnull().any():
                self.features[feature] = Feature(df[feature])
