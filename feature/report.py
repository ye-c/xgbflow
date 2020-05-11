import pandas as pd
import numpy as np


'''
feature = [...]
bench_man = {
    'nfcs_total_loan': [float('-inf'), -1, 0.0, 860000.0, 1720000.0, 2580000.0, 3440000.0, float('inf')],
    ...
}

# calc benchmark of all feature
df1 = pd.read_csv(csv_of_benchmark).fillna(-1)
bench = {}
for c in feature:
    bench[c] = report.benchmark_csi(
        df1[c], bench_man[c]) if c in bench_man else report.benchmark_csi(df1[c])

# calc csi
df2 = pd.read_csv(csv_of_actual).fillna(-1)
for c in feature:
    res = report.csi(df2[c], bench[c][0], bench[c][1])
    print(c)
    print(bench[c][0])
    print(res)
'''


def benchmark_psi(dfc, bins=[], cutoff=5):
    '''
    dfc:    df.col
    return: bins, dfc_bins_rate
    '''
    if not bins:
        maxv = dfc.max()
        step = round(maxv / cutoff, 3)
        bins = [
            round(i, 3)
            for i in np.arange(0, maxv, step)
        ]
        bins = bins + [1]

    cut = pd.cut(dfc, bins)
    dfc_bins_rate = pd.value_counts(cut, sort=False).to_frame('count')
    dfc_bins_rate['rate'] = dfc_bins_rate['count'] / dfc.count()

    rebench = 0
    for index, row in dfc_bins_rate.iterrows():
        if row['count'] == 0:
            # print(bins)
            # print(index.right)
            bins.remove(index.right)
            rebench = 1

    if rebench:
        bins, dfc_bins_rate = benchmark_psi(dfc, bins)
    return bins, dfc_bins_rate


def benchmark_csi(dfc, bins=[], cutoff=5):
    '''
    dfc:    df.col
    '''
    if not bins:
        maxv = dfc.max()
        step = round(maxv / cutoff, 3) if maxv > 10 else 1
        bins = [
            round(i, 3)
            for i in np.arange(0, maxv, step)
        ]
        bins = [float('-inf'), -1] + bins + [float('inf')]

    cut = pd.cut(dfc, bins)
    dfc_bins_rate = pd.value_counts(cut, sort=False).to_frame('count')
    dfc_bins_rate['rate'] = dfc_bins_rate['count'] / dfc.count()

    rebench = 0
    for index, row in dfc_bins_rate.iterrows():
        if row['count'] == 0:
            # print(bins)
            # print(index.right)
            bins.remove(index.right)
            rebench = 1

    if rebench:
        bins, dfc_bins_rate = benchmark_csi(dfc, bins)
    return bins, dfc_bins_rate


def bench_and_csi(bench_df_col, act_df_cols=[], bins=[]):
    '''
    benchcsv = 'm10_feature.csv'
    act_csvs = [
        'm2_feature.csv',
        'm3_feature.csv',
        'm4_feature.csv',
    ]
    c = 'nfcs_total_loan'
    bins = []
    bins = [float('-inf'), -1, 0.0, 860000.0, 1720000.0,
            2580000.0, 3440000.0, float('inf')]

    df0 = pd.read_csv(benchcsv).fillna(-1)
    acts = [pd.read_csv(csv).fillna(-1)[c] for csv in act_csvs]

    rescsi = report.bench_and_csi(df0[c], acts, bins)
    print(rescsi)
    '''
    bins, bins_rate = benchmark_csi(bench_df_col, bins, cutoff=5)
    res_csi = []
    for act_df_c in act_df_cols:
        res = calc_psi_csi(act_df_c, bins, bins_rate)
        res_csi.append(res)
    return res_csi


def calc_psi_csi(dfc, bench_bins, bench_bins_rate):
    '''
    dfc: df.col
    bench_bins, bench_bins_rate = benchmark_csi(dfc, bins=[], cutoff=5)
    bench_bins = bench_bins_rate.index.categories.to_tuples().values
    '''
    cut = pd.cut(dfc, bench_bins)
    dfc_bins_rate = pd.value_counts(cut, sort=False).to_frame('count')
    dfc_bins_rate['rate'] = dfc_bins_rate['count'] / dfc.count()
    psi_csi = (dfc_bins_rate['rate'] - bench_bins_rate['rate']) * \
        np.log(dfc_bins_rate['rate'] / bench_bins_rate['rate'])
    dfc_bins_rate['psi_csi'] = np.around(psi_csi, decimals=4)
    # print(dfc_bins_rate)

    psi_res = dfc_bins_rate['psi_csi'].sum()

    if psi_res == float('inf'):
        print('WARNING...')
        print('Column:', dfc.name)
        print('Rate:')
        print(dfc_bins_rate['rate'])
        print('Bins:', bench_bins)

    return dfc_bins_rate, psi_res
