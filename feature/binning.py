import numpy as np
import pandas as pd
from feature import indicator


def frequence(dfc, bin_num=5):
    '''
    等频分箱
    '''
    percent_value = 1.0 / bin_num
    percentile_rate = [i * percent_value for i in range(1, bin_num)]
    percentile_rate.append(1.0)
    cutlist = dfc.quantile(percentile_rate, interpolation="lower")
    cutlist = cutlist.drop_duplicates(keep="last")
    cutlist = [round(i, 6) for i in cutlist]
    return cutlist


def distance(dfc, bin_num=5):
    '''
    等宽（等距）分箱
    '''
    bin_num = bin_num - 2
    maxv = dfc.max()
    minv = dfc.min()
    step = float((maxv - minv) / bin_num)
    bins = [round(i, 6) for i in np.arange(minv, maxv, step)]
    bins.insert(0, float('-inf'))
    bins.append(float('inf'))
    return bins


def bin_map(dfc, init_splite_points):
    def package_bin(x, splite_points):
        if x <= splite_points[0]:
            return splite_points[0]
        if x > splite_points[len(splite_points) - 1]:
            return float('inf')
        for i in range(len(splite_points) - 1):
            if x > splite_points[i] and x <= splite_points[i + 1]:
                return splite_points[i + 1]

    return dfc.apply(package_bin, splite_points=init_splite_points)


# 定义一个卡方分箱（可设置参数置信度水平与箱的个数）停止条件为大于置信水平且小于bin的数目
def chiMerge(df, variable, flag, bins=10, confidenceVal=3.841, sample=None):
    '''
    运行前需要 import pandas as pd 和 import numpy as np
    df:传入一个数据框仅包含一个需要卡方分箱的变量与正负样本标识（正样本为1，负样本为0）
    variable:需要卡方分箱的变量名称（字符串）
    flag：正负样本标识的名称（字符串）
    confidenceVal：置信度水平（默认是不进行抽样95%）
    bins：最多箱的数目
    sample: 为抽样的数目（默认是不进行抽样），因为如果观测值过多运行会较慢
    '''

    def groupby_split_bin(df, var, tar, bins, N=100, init_bin='frequence'):
        print('[%s] is continuous variable ...' % var)
        if init_bin == 'frequence':
            init_splite_points = frequence(df[var], N)
        else:
            init_splite_points = distance(df[var], N)
        init_splite_points = init_splite_points[1:-1]
        df[var] = bin_map(df[var], init_splite_points)
        return df.groupby([var])[tar].count()

    def group_data(df, var, tar, N=100):
        df_group = df.groupby([var])[tar].count()
        res = df_group if len(df_group) <= N \
            else groupby_split_bin(df, var, tar, bins, N=100, init_bin='frequence')
        res = pd.DataFrame({'total_num': res})
        res['pos_class'] = df.groupby([var])[tar].sum()
        res['neg_class'] = res['total_num'] - res['pos_class']
        res.reset_index(inplace=True)
        res = res.drop('total_num', axis=1)
        np_regroup = np.array(res)
        return np_regroup

    def merge_zero_square(np_regroup):
        i = 0
        while (i <= np_regroup.shape[0] - 2):
            '''
            pos neg
            v1  v2
            v3  v4
            '''
            v0 = np_regroup[i + 1, 0]
            v1 = np_regroup[i, 1]
            v2 = np_regroup[i, 2]
            v3 = np_regroup[i + 1, 1]
            v4 = np_regroup[i + 1, 2]
            flag_col1 = (v1 == v3 == 0)
            flag_col2 = (v2 == v4 == 0)
            if (flag_col1 or flag_col2):
                np_regroup[i, 1] += v3  # 正样本
                np_regroup[i, 2] += v4  # 负样本
                np_regroup[i, 0] = v0
                np_regroup = np.delete(np_regroup, i + 1, 0)
                i = i - 1
            i = i + 1
        return np_regroup

    def calc_chi2(np_regroup):
        chi_table = np.array([])
        for i in np.arange(np_regroup.shape[0] - 1):
            '''
            pos neg
            v1  v2
            v3  v4
            '''
            v1 = np_regroup[i, 1]
            v2 = np_regroup[i, 2]
            v3 = np_regroup[i + 1, 1]
            v4 = np_regroup[i + 1, 2]
            chi = (v1 * v4 - v2 * v3) ** 2 * (v1 + v2 + v3 + v4) / \
                  ((v1 + v2) * (v3 + v4) * (v1 + v3) * (v2 + v4))
            chi_table = np.append(chi_table, chi)
        return chi_table

    def chi_merge(np_regroup, chi_table, bins, confidenceVal):
        while True:
            '''
            pos neg
            vf1 vf2
            v1  v2
            v3  v4
            '''
            if (len(chi_table) <= (bins - 1) and min(chi_table) >= confidenceVal):
                break
            # 找出卡方值最小的位置索引
            chi_min_index = np.argwhere(chi_table == min(chi_table))[0]
            # 合并最小卡方值区间
            np_regroup[chi_min_index, 1] += np_regroup[chi_min_index + 1, 1]
            np_regroup[chi_min_index, 2] += np_regroup[chi_min_index + 1, 2]
            np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
            np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

            v1 = np_regroup[chi_min_index, 1]
            v2 = np_regroup[chi_min_index, 2]
            vf1 = np_regroup[chi_min_index - 1, 1]
            vf2 = np_regroup[chi_min_index - 1, 2]

            if (chi_min_index == np_regroup.shape[0] - 1):  # 最小值是最后两个区间的时候
                # 计算合并后当前区间与前一个区间的卡方值并替换
                chi_table[chi_min_index - 1] = (vf1 * v2 - vf2 * v1) ** 2 * (vf1 + vf2 + v1 + v2) / \
                                               ((vf1 + vf2) * (v1 + v2) * (vf1 + v1) * (vf2 + v2))
                # 删除替换前的卡方值
                chi_table = np.delete(chi_table, chi_min_index, axis=0)
            else:
                v3 = np_regroup[chi_min_index + 1, 1]
                v4 = np_regroup[chi_min_index + 1, 2]
                # 计算合并后当前区间与前一个区间的卡方值并替换
                chi_table[chi_min_index - 1] = (vf1 * v2 - vf2 * v1) ** 2 * (vf1 + vf2 + v1 + v2) / \
                                               ((vf1 + vf2) * (v1 + v2) * (vf1 + v1) * (vf2 + v2))
                # 计算合并后当前区间与后一个区间的卡方值并替换
                chi_table[chi_min_index] = (v1 * v4 - v2 * v3) ** 2 * (v1 + v2 + v3 + v4) / \
                                           ((v1 + v2) * (v3 + v4) * (v1 + v3) * (v2 + v4))
                # 删除替换前的卡方值
                chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)

        return np_regroup

    def calc_woe_iv(np_result):
        df_woe_iv = pd.DataFrame(np_result, columns=['cutoff', 'pos', 'neg'])
        pos_all = sum(df_woe_iv['pos'])
        neg_all = sum(df_woe_iv['neg'])
        df_woe_iv['pos_rate'] = df_woe_iv['pos'] / pos_all
        df_woe_iv['neg_rate'] = df_woe_iv['neg'] / neg_all
        df_woe_iv['woe'] = df_woe_iv.apply(indicator.formula_woe, axis=1)
        df_woe_iv['iv'] = df_woe_iv.apply(indicator.formula_iv, axis=1)
        woe_dict = dict(zip(df_woe_iv['cutoff'], df_woe_iv['woe']))
        return woe_dict, sum(df_woe_iv['iv']), df_woe_iv

    def pack_result(variable, np_regroup, df_woe_iv):
        result_data = pd.DataFrame()
        # 结果表第一列：变量名
        result_data['variable'] = [variable] * np_regroup.shape[0]
        list_temp = []
        for i in np.arange(np_regroup.shape[0]):
            if i == 0:
                x = '(-inf, %s]' % str(np_regroup[i, 0])
            elif i == np_regroup.shape[0] - 1:
                x = '(%s, inf]' % str(np_regroup[i - 1, 0])
            else:
                x = '(%s, %s]' % \
                    (str(np_regroup[i - 1, 0]), str(np_regroup[i, 0]))
            list_temp.append(x)
        result_data['cutoff'] = np_regroup[:, 0]
        result_data['interval'] = list_temp  # 结果表第二列：区间
        result_data['count_0'] = np_regroup[:, 2]  # 结果表第三列：负样本数目
        result_data['count_1'] = np_regroup[:, 1]  # 结果表第四列：正样本数目
        result_data['ratio_1'] = np_regroup[:, 1] / \
                                 (np_regroup[:, 1] + np_regroup[:, 2])
        result_data = pd.merge(result_data[['variable', 'cutoff', 'interval']], df_woe_iv, how='left', on='cutoff')
        return result_data

    # 进行是否抽样操作
    if sample:
        df = df.sample(n=sample)

    # 进行数据格式化录入
    np_regroup = group_data(df, variable, flag)

    # 处理连续没有正样本或负样本的区间，并进行区间的合并（以免卡方值计算报错）
    np_regroup = merge_zero_square(np_regroup)

    # 对相邻两个区间进行卡方值计算
    chi_table = calc_chi2(np_regroup)

    # 把卡方值最小的两个区间进行合并（卡方分箱核心）
    np_result = chi_merge(np_regroup, chi_table, bins, confidenceVal)

    # 计算 woe iv
    woe_dict, iv, df_woe_iv = calc_woe_iv(np_result)

    # 把结果保存成一个数据框
    result_data = pack_result(variable, np_result, df_woe_iv)
    bin_cut = list(result_data.cutoff)

    return bin_cut, woe_dict, iv, result_data
