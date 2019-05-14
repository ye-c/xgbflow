import numpy as np
import pandas as pd


def ratio(dfls=[], keys=['train', 'verify']):
    if not isinstance(dfls, list):
        dfls = [dfls]
    res = []
    for name, df in zip(keys, dfls):
        shape = df.shape
        rat = df.groupby('label').size()
        pos = rat[1] if 1 in rat.index else 0
        neg = rat[0] if 0 in rat.index else 0
        rp, rn = '%.2f%%' % (
            pos * 100 / shape[0]), '%.2f%%' % (neg * 100 / shape[0])
        pnr = '1:%.1f' % (neg / pos) if pos != 0 else 'nan'
        res.append([name, shape, '%s (%s)' %
                    (pos, rp), '%s (%s)' % (neg, rn), pnr])
    header = ['/', 'Shape', 'Pos', 'Neg', 'Pos:Neg']
    return header, res


def coverage(dfls=[], cols=[], dfname=['train', 'verify']):
    if not isinstance(dfls, list):
        dfls = [dfls]
    if not isinstance(cols, list):
        cols = [cols]
    res = []
    for c in cols:
        tmp = [c]
        for df in dfls:
            null_num = df[df[c] == -1].shape[0]
            tmp.append('%.0f%%' % (100 - null_num * 100 / df.shape[0]))
        res.append(tmp)
    header = ['column'] + dfname
    return header, res


def predict_pos_ratio(df, fscore='score', fpos='label', cutn=5, cutoff=[]):
    overnum = df[df[fpos] == 1].shape[0]
    if overnum == 0:
        return

    if not cutoff:
        maxv = round(df[fscore].max(), 4)
        ls1 = [round(i, 4) for i in np.arange(0, maxv, maxv / cutn)]
        ls2 = ls1[1:] + [maxv]
        cutoff = list(zip(ls1, ls2))

    res = []
    for v in cutoff:
        if isinstance(v, list) or isinstance(v, tuple):
            t1 = df.loc[(df[fscore] > v[0]) & (df[fscore] <= v[1])]
            n1 = t1.loc[t1[fpos] == 1].shape[0]
        else:
            t1 = df.loc[df[fscore] == v]
            n1 = t1.loc[t1[fpos] == 1].shape[0]
        res.append((v, round(n1 / overnum, 4)))
    return res


def distribution(coldata, colname, cutn=10, cutoff=[]):
    if not cutoff:
        maxv = round(coldata.max(), 4)
        ls1 = [round(i, 4) for i in np.arange(0, maxv, maxv / cutn)]
        ls2 = ls1[1:] + [maxv]
        cutoff = list(zip(ls1, ls2))

    df = coldata.to_frame()
    num = df.shape[0]
    res = []
    for v in cutoff:
        if isinstance(v, list) or isinstance(v, tuple):
            n1 = df.loc[(df[colname] > v[0]) & (df[colname] <= v[1])].shape[0]
        else:
            n1 = df.loc[df[colname] == v].shape[0]
        res.append((v, round(n1 / num, 4)))
    return res


def calc_feature_dist(train_df, verify_df, tar_fea, cutoff, cutn=10):
    cols = ['label', tar_fea]
    trdata = train_df[cols]
    trnum = trdata.shape[0]
    vedata = verify_df[cols]
    venum = vedata.shape[0]
    res = []

    if not cutoff:
        maxv = max(trdata[tar_fea].max(), vedata[tar_fea].max())
        ls1 = [int(i) for i in np.arange(0, maxv, maxv / cutn)]
        ls2 = ls1[1:] + [maxv]
        cutoff = [-1, 0] + list(zip(ls1, ls2))

    for v in cutoff:
        if isinstance(v, list) or isinstance(v, tuple):
            t1 = trdata.loc[(trdata[tar_fea] > v[0]) & (
                trdata[tar_fea] <= v[1])].shape[0]
            t2 = vedata.loc[(vedata[tar_fea] > v[0]) & (
                vedata[tar_fea] <= v[1])].shape[0]
        else:
            t1 = trdata.loc[trdata[tar_fea] == v].shape[0]
            t2 = vedata.loc[vedata[tar_fea] == v].shape[0]
        res.append((v, [round(t1 / trnum, 4), round(t2 / venum, 4)]))
    return res


def statistic_bins_cov_acc_rec(df, bins, col_score='score', col_label='label'):
    '''
    原 calc_score_pos_ratio 重构版
    '''
    tmpdf = df.copy()
    dfcount = df.shape[0]
    poscount = pd.value_counts(df[col_label])[1]
    cut = pd.cut(df[col_score], bins)
    resdf = pd.value_counts(cut, sort=False).to_frame()
    resdf.columns = ['Count']
    tmpdf['cut'] = cut
    tmpdf = tmpdf[tmpdf[col_label] == 1]
    resdf['PosCount'] = pd.value_counts(tmpdf['cut'], sort=False)
    resdf['Coverage'] = round(resdf['Count'] * 100 / dfcount, 2)
    resdf['Accuracy'] = round(resdf['PosCount'] * 100 / resdf['Count'], 2)
    resdf['Recall'] = round(resdf['PosCount'] * 100 / poscount, 2)
    return resdf


def calc_score_pos_ratio(data, label_score, cutoff, cutn=10):
    '''
    deprecated 弃用
    '''
    label, tar_fea = label_score
    trdata = data[label_score]
    trnum = trdata.shape[0]
    posnum = trdata[trdata[label] == 1].shape[0]
    res = []

    if not cutoff:
        maxv = trdata[tar_fea].max()
        ls1 = [round(i, 4) for i in np.arange(0, maxv, maxv / cutn)]
        ls2 = ls1[1:] + [maxv]
        cutoff = list(zip(ls1, ls2))

    for v in cutoff:
        if isinstance(v, list) or isinstance(v, tuple):
            t1 = trdata.loc[(trdata[tar_fea] > v[0]) &
                            (trdata[tar_fea] <= v[1])]
        else:
            t1 = trdata.loc[trdata[tar_fea] == v]
        n1 = t1.shape[0]
        p1 = t1[t1[label] == 1].shape[0]
        cov = round(n1 * 100 / trnum, 2) if n1 != 0 else 0
        acc = round(p1 * 100 / n1, 2) if n1 != 0 else 0
        rec = round(p1 * 100 / posnum, 2)
        res.append((v, trnum, p1, cov, acc, rec))
    return res
