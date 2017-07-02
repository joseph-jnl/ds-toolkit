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
    ftype: string >> 'ID', Categorical', 'Numeric', 'Date', 'Unknown'
    data: numpy Series
    missing_index: int list >> [0, 1, 4, 100, 123]  
    """

    def __init__(self, feature):
        self.name = feature.name
        if feature.dtype in ['object', 'bool']:
            self.ftype = 'Categorical'
        elif str(feature.dtype).startswith(('float', 'int')):
            self.ftype = 'Numeric'
        elif str(feature.dtype).startswith(('datetime', 'timedelta')):
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

    categorical: list >> categorical = ['browser', 'country'] 
                    manually define features that are categorical

    identifier: list >> identifier = ['id', 'name']
                    manually define features that are identifiers
    ignore: list
            specify columns to ignore

    Attributes
    ----------
    features: dict containing feature class with missing values
    df: input pandas data frame
    """

    def __init__(self, df, categorical=[], identifier=[], ignore=[]):
        self.df = df
        self.features = {}
        for feature in df.columns:
            if df[feature].isnull().any() and feature not in ignore:
                self.features[feature] = Feature(df[feature])
                # Manually override feature type if defined
                if feature in categorical:
                    self.features[feature].ftype = 'Categorical'
                elif feature in identifier:
                    self.features[feature].ftype = 'ID'

    def summary(self, plot=False):
        '''
        Summarize features with missing values

        Parameters
        ----------
        plot: Show plot of missing values
              ex >> plot=True

        Returns
        -------
        Summary of features with missing NaN
        Including Name, type, number of missing NaN's, % Missing
        '''
        print('{0:15} {1:12} {2:7}       {3:}'.format('name', 'type', '# NaNs', '%'))
        print('------------------------------------------------')
        for name, f in self.features.items():
            print('{0:15} {1:12} {2:7}       {3:0.3f}'.format(name, f.ftype,
                                                              len(f.missing_index),
                                                              len(f.missing_index) / self.df.shape[0]))

        return None
