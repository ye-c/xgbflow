import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def ratio_satisfy(df, target='y', pos_rate=None, pos_div_neg=None, random_state=None):
    '''
    pos=1, neg=0
    adjust df's pos/neg = rate
    '''
    pos_df = df[df[target] == 1]
    neg_df = df[df[target] == 0]
    if pos_rate:
        var = pos_df.shape[0] / pos_rate - pos_df.shape[0]
        neg_sample_rate = var / neg_df.shape[0]
    elif pos_div_neg:
        neg_sample_rate = pos_df.shape[0] / (pos_div_neg * neg_df.shape[0])
    else:
        neg_sample_rate = 1
    df_neg = neg_df.sample(frac=neg_sample_rate, random_state=random_state)
    df_new = pd.concat([df_neg, pos_df])
    return df_new


def badrate_encoding(df, cols, target='y', axis=False):
    '''
    cols: categorical feature list
    '''
    if not isinstance(cols, list):
        cols = [cols]
    for feature in cols:
        bad_count = df.groupby(feature).sum()
        col_count = df.groupby(feature).size()
        badrate_dict = dict(zip(
            bad_count.index,
            bad_count[target] / col_count
        ))
        # print(badrate_dict)
        any_null = df[feature].isnull().any()
        assert not any_null, '{} has null'.format(feature)
        col_badrate = feature if axis else feature + '_badrate'
        df[col_badrate] = df[feature].map(lambda x: badrate_dict[x])
    return df
