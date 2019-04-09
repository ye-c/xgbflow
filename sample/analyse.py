from xgbflow.utils.markitdown import MarkitDown


def ratio(dfls=[], keys=['train', 'verify']):
    if not isinstance(dfls, list):
        dfls = [dfls]
    md = MarkitDown()
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
    heads = ['/', 'shape', 'pos', 'neg', 'pos:neg']
    md.table(heads, res)
    md.show()


def coverage(dfls=[], cols=[], dfname=['train', 'verify']):
    if not isinstance(dfls, list):
        dfls = [dfls]
    if not isinstance(cols, list):
        cols = [cols]
    md = MarkitDown()
    res = []
    for c in cols:
        tmp = [c]
        for df in dfls:
            null_num = df[df[c] == -1].shape[0]
            tmp.append('%.0f%%' % (100 - null_num * 100 / df.shape[0]))
        res.append(tmp)
    heads = ['column'] + dfname
    md.table(heads, res)
    md.show()