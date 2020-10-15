import numpy as np
import pandas as pd


def formula_woe(df, col_pos='pos_rate', col_neg='neg_rate'):
    res = np.log(df[col_pos] * 1.0 / df[col_neg])
    # res = np.log((df[col_pos] * 1.0 + 0.1) / (df[col_neg] + 0.1))
    return res


def formula_iv(df, col_pos='pos_rate', col_neg='neg_rate'):
    '''
    IV<=0.02 : 无预测能力；
    0.02 - 0.1 ：弱预测能力；
    0.1 - 0.3 ：中预测能力；
    0.3 - 0.5 ：强预测能力；
    大于0.5的为超强预测能力；
    '''
    # res = (df[col_pos] - df[col_neg]) * np.log((df[col_pos] * 1.0 + 0.1)/ (df[col_neg] + 0.1))
    res = (df[col_pos] - df[col_neg]) * formula_woe(df, col_pos, col_neg)
    return res


def calc_woe_iv(df, col, target):
    '''
    col: 已分箱或类别变量
    return: dict{woe: , iv: }
    '''
    df_group = df.groupby([col])[target].count()
    df_group = pd.DataFrame({'total': df_group})
    df_group['pos'] = df.groupby([col])[target].sum()
    df_group['neg'] = df_group['total'] - df_group['pos']
    df_group.reset_index(inplace=True)
    pos_all = sum(df_group['pos'])
    neg_all = sum(df_group['neg'])
    df_group['pos_rate'] = df_group['pos'] / pos_all
    df_group['neg_rate'] = df_group['neg'] / neg_all
    df_group['woe'] = df_group.apply(formula_woe, axis=1)
    df_group['iv'] = df_group.apply(formula_iv, axis=1)
    woe_dict = dict(zip(df_group[col], df_group['woe']))
    return woe_dict, sum(df_group['iv'])
