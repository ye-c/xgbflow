from xgbflow.utils.markitdown import MarkitDown
import numpy as np


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
    header = ['/', 'shape', 'pos', 'neg', 'pos:neg']
    # md = MarkitDown()
    # md.table(header, res)
    # md.show()
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
    # md = MarkitDown()
    # md.table(header, res)
    # md.show()
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
