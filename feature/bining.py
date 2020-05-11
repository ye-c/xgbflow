import numpy as np
import pandas as pd


def frequence(dfc, perc):
    '''
    等频分箱
    '''
    desc = dfc.describe(percentiles=perc if perc else None)
    # print(desc)
    res = desc.to_dict()
    res.pop('count')
    res.pop('mean')
    res.pop('std')
    # res.pop('min')
    # res.pop('max')
    return res


def distance(dfc, n=5):
    '''
    等宽（等距）分箱
    '''
    maxv = dfc.max()
    minv = dfc.min()
    step = int((maxv - minv) / n)
    bins = [i for i in np.arange(minv, maxv, step)]
    bins.insert(0, float('-inf'))
    bins.append(float('inf'))
    return bins
