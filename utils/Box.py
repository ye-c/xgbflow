import numpy as np
import pandas as pd


class Bin(object):
	@staticmethod
	def __get_splite_points(data, bin=5, type='frequence'):
		"""
		:param df: 數據表
		:param col: 分箱列名
		:param bin: 分箱數量
		:param type: frequence 等频率切分  distance 等距离切分
		:return:
		"""
		# data = df[col].values
		if type == 'frequence':
			data = np.sort(data, 0)
			n = int(len(data) / bin)
			spliteIndex = [i * n for i in range(1, bin)]
			return [data[index] for index in spliteIndex]
		else:
			max_value, min_value = np.max(data), np.min(data)
			distance = (max_value - min_value) / bin
			return [(min_value + i * distance) for i in range(1, bin)]

	@staticmethod
	def make_bin(x, splite_points):
		# 该方法会将 小于 第一个切割点的
		if x <= splite_points[0]:  # 如果小于最小切割点
			return splite_points[0]
		if x > splite_points[len(splite_points) - 1]:  # 如果大于最大切割点
			return float('inf')
		for i in range(len(splite_points) - 1):
			if x > splite_points[i] and x <= splite_points[i + 1]:
				return splite_points[i + 1]

	@staticmethod
	def chi2merge(df, col, target, bin=5, init_bin=100, init_type='distance'):
		"""
		卡方分箱法
		:param df: 待分箱的DadaFrame
		:param col: 要分箱的列
		:param target: 标签行
		:param bin: 分箱的个数
		:param init_bin: 初始化分箱的数量
		:param init_type: 初始化分箱采用的类型 distance 等距离， frequence 等频率
		:return:
		"""
		data = df[[col, target]]
		col_have_null = data.isnull().any()
		if col_have_null[0] == True:
			raise ValueError('数据列 {} 有空值，请先处理'.format(col))
		if col_have_null[1] == True:
			raise ValueError('数据列 {} 有空值，请先处理'.format(col))

		col_values = data[col].values
		init_splite_points = Bin.__get_splite_points(col_values, init_bin, init_type)

		data[col] = data[col].apply(Bin.make_bin, splite_points=init_splite_points)

		# 进行初始化的统计信息
		regroups = pd.DataFrame({'count': data.groupby(by=col)[target].value_counts()})
		regroups = regroups.pivot_table(values='count', index=[col], columns=target, fill_value=0)
		np_regroups = regroups.reset_index().sort_values(by=col).values
		print("数据加载完毕开始进行分箱计算")

		# 处理连续为0值的情况以免计算报错
		i = 0
		while i < len(np_regroups) - 1:
			if ((np_regroups[i, 1]==0 and np_regroups[i + 1, 1]==0) or
					(np_regroups[i, 2] ==0 and np_regroups[i + 1,2] ==0)):
				np_regroups[i, 0] = np_regroups[i + 1, 0]
				np_regroups[i, 1] = np_regroups[i, 1] + np_regroups[i + 1, 1]
				np_regroups[i, 2] = np_regroups[i, 2] + np_regroups[i + 1, 2]
				np_regroups = np.delete(np_regroups, i + 1, 0)
			else:
				i = i + 1
		print('数据连续空值处理完毕，开始分箱核心计算步骤')

		while len(np_regroups) > bin:
			chi_table = []
			for i in range(len(np_regroups) - 1):
				chi_table.append(
					Bin.calc_chi2(np_regroups[i, 1], np_regroups[i, 2], np_regroups[i + 1, 1], np_regroups[i + 1, 2]))
			i = np.argwhere(chi_table == min(chi_table))[0]

			if i == len(np_regroups) - 1:
				# np_regroups[i,0] = np_regroups[i-1,0]
				np_regroups[i, 1] = np_regroups[i, 1] + np_regroups[i - 1, 1]
				np_regroups[i, 2] = np_regroups[i, 2] + np_regroups[i - 1, 2]
				np_regroups = np.delete(np_regroups, i - 1, 0)
			else:
				np_regroups[i, 0] = np_regroups[i + 1, 0]
				np_regroups[i, 1] = np_regroups[i, 1] + np_regroups[i + 1, 1]
				np_regroups[i, 2] = np_regroups[i, 2] + np_regroups[i + 1, 2]
				np_regroups = np.delete(np_regroups,i+1,0)

		#print(np_regroups)
		x = []
		for v in np_regroups:
			x.append([v[0],v[1],v[2],str(v[2]/(v[1]+v[2])*100)+'%'])
		print(x)
		return [v[0] for v in np_regroups]


	#	 高效率合并逻辑，卡方计算值会反复复用
	# 	计算初始化的卡方值
	# 	chi_table = []
	# 	for i in range(len(np_regroups) -1):
	# 		chi_table.append(Bin.calc_chi2(np_regroups[i,1],np_regroups[i,2],np_regroups[i+1,1],np_regroups[i+1,2]))
	# 	print('已经完成第一轮卡方值计算')
	# 	while len(np_regroups) >= bin:
	# 		#卡方值最小的一组
	# 		i = np.argwhere(chi_table== min(chi_table))[0]
	# 		if i == 0 : #如果最小的卡方在第一位
	# 			np_regroups[i,1] = np_regroups[i,1] + np_regroups[i+1,1]
	# 			np_regroups[i,2] = np_regroups[i,2] + np_regroups[i+1,2]
	# 			np_regroups[i,0] = np_regroups[i+1,0]
	# 			np_regroups = np.delete(np_regroups,i+1,0)
	#
	# 			chi_table[i] = Bin.calc_chi2(np_regroups[i,1],np_regroups[i,2],np_regroups[i+1,1],np_regroups[i+1,2])
	# 			chi_table = np.delete(chi_table,i+1,0)
	#
	# 		elif i == len(np_regroups)- 2 : # 如果是最后一位
	# 			np_regroups[i,1] = np_regroups[i,1] + np_regroups[i+1,1]
	# 			np_regroups[i,2] = np_regroups[i,2] + np_regroups[i+1,2]
	# 			np_regroups[i,0] = np_regroups[i+1,0]
	# 			np_regroups = np.delete(np_regroups,i+1,0)
	#
	# 			chi_table[i-1] = Bin.calc_chi2(np_regroups[i-1,1],np_regroups[i-1,2],np_regroups[i,1],np_regroups[i,2])
	# 			chi_table = np.delete(chi_table,i,0)
	#
	# 		else:
	# 			np_regroups[i,1] = np_regroups[i,1] + np_regroups[i+1,1]
	# 			np_regroups[i,2] = np_regroups[i,2] + np_regroups[i+1,2]
	# 			np_regroups[i,0] = np_regroups[i+1,0]
	# 			np_regroups = np.delete(np_regroups,i+1,0)
	#
	# 			chi_table[i-1] = Bin.calc_chi2(np_regroups[i-1,1],np_regroups[i-1,2],np_regroups[i,1],np_regroups[i,2])
	# 			chi_table[i] = Bin.calc_chi2(np_regroups[i,1],np_regroups[i,2],np_regroups[i+1,1],np_regroups[i+1,2])
	# 			chi_table = np.delete(chi_table,i+1,0)

	@staticmethod
	def calc_chi2(a, b, c, d):
		"""
		如下横纵标对应的卡方计算公式为： K^2 = n (ad - bc) ^ 2 / [(a+b)(c+d)(a+c)(b+d)]　其中n=a+b+c+d为样本容量
			y1   y2
		x1  a    b
		x2  c    d
		:return: 卡方值
		"""
		return (a + b + c + d) * (a * d - b * c) ** 2 / (a + b) * (c + d) * (b + d) * (a + c)
#测试代码
# if __name__ =="__main__":
# 	raw_data = pd.read_csv("../other_code/test_data.csv",header=0)
# 	df = raw_data[['zmxy_score','flag']].dropna(how='any')
# 	zmxy_bins = Bin.chi2merge(df,'zmxy_score','flag',4,init_bin=100)
# 	print(zmxy_bins)

# 该类中的方法守极端值影响很大，需要重构该设计
# class __Chimerge:
#     def get_bin_points2(self,df,col,target,bins=5,init_bins = 100,confidenceVal=3.841):
#         '''
#         :param df:
#         :param col:
#         :param target:
#         :param bins:
#         :param init_bins: 初始分区数量
#         :param confidenceVal: 卡方阈值
#         :return:
#         '''
#
#     def get_bin_points1(self, df, col, targe, bins=5, sample=None, confidenceVal=3.841):
#         """
#         输入的数据为DataFrame格式的
#         :param df: 对应的数据集
#         :param col: 需要分享的列名
#         :param targe: 对应的y值得列名
#         :param new_col: 对应的分好
#         :param make_up_nan: 空值的填充项目，如果为None则过滤掉空值，不做任何填充
#         :param bins 分箱的数量
#         :param merge_direction down ->左开右闭  up-> 左闭右开
#         :return:数据切割点数组[]
#         """
#         # 不处理空值
#         data = None
#         if sample != None:
#             data = df[[col, targe]].sample(n=sample)
#         data = df[[col, targe]]
#
#         col_have_null = data.isnull().any()
#         if col_have_null[0] == True:
#             raise ValueError('自变量有空值，请检查数据')
#         if col_have_null[1] == True:
#             raise ValueError('因变量有空值，请检查数据')
#
#         regrouped = data.groupby(by=col)[targe].value_counts()
#         regrouped = pd.DataFrame({'count': regrouped})
#         regrouped = regrouped.pivot_table(values='count', index=[col], columns=targe, fill_value=0)
#         regrouped = regrouped.reset_index().sort_values(by=col)
#
#         # 为了提升效率，使用array进行计算
#         np_regroup = regrouped.values
#         print("数据加载完毕，开始处理连续的空值")
#
#         # 处理连续没有正样本或负样本的区间，并进行区间的合并（以免卡方值计算报错）
#         i = 0
#         while i < np_regroup.shape[0] - 1:
#             if (np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or (
#                     np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0):
#                 np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]
#                 np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]
#                 np_regroup = np.delete(np_regroup, i + 1, 0)
#             else:
#                 i = i + 1
#
#         chi_table = np.array([])
#         for i in np.arange(np_regroup.shape[0] - 1):
#             chi = self.cal_chi2(np_regroup[i, 1], np_regroup[i, 2], np_regroup[i + 1, 1], np_regroup[i + 1, 2])
#             chi_table = np.append(chi_table, chi)
#         print("已经完成第一轮的卡方值计算，开始按照卡方最小值进行合并")
#
#         while ( (len(chi_table) > bins) ): #and (mean(chi_table) >= confidenceVal)
#             chi_min_index = np.argwhere(chi_table == np.min(chi_table))[0]
#             # np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
#             np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
#             np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
#             np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)
#
#             if chi_min_index == 0:  # 如果是第一个位置的数据
#                 chi_table[chi_min_index] = self.cal_chi2(np_regroup[chi_min_index, 1], np_regroup[chi_min_index, 2],
#                                                          np_regroup[chi_min_index +1, 1],
#                                                          np_regroup[chi_min_index + 1, 2])
#                 chi_table =np.delete(chi_table, chi_min_index + 1, 0)
#             elif chi_min_index == np_regroup.shape[0] - 1:  # 如果是末位的分组
#                 chi_table[chi_min_index - 1] = self.cal_chi2(np_regroup[chi_min_index - 1, 1],
#                                                              np_regroup[chi_min_index - 1, 2],
#                                                              np_regroup[chi_min_index, 1], np_regroup[chi_min_index, 2])
#                 chi_table =np.delete(chi_table, chi_min_index, 0)
#             else:
#                 chi_table[chi_min_index - 1] = self.cal_chi2(np_regroup[chi_min_index - 1, 1],
#                                                              np_regroup[chi_min_index - 1, 2],
#                                                              np_regroup[chi_min_index, 1], np_regroup[chi_min_index, 2])
#                 chi_table[chi_min_index] = self.cal_chi2(np_regroup[chi_min_index, 1], np_regroup[chi_min_index, 2],
#                                                          np_regroup[chi_min_index +1, 1],
#                                                          np_regroup[chi_min_index + 1, 2])
#                 chi_table =np.delete(chi_table, chi_min_index + 1, 0)
#         print("已经完成卡方合并的计算过程")
#         return [value[0] for value in np_regroup]
#
#     def cal_chi2(self, a, b, c, d):
#         """r
#         如下横纵标对应的卡方计算公式为： K^2 = n (ad - bc) ^ 2 / [(a+b)(c+d)(a+c)(b+d)]　其中n=a+b+c+d为样本容量
#             y1   y2
#         x1  a    b
#         x2  c    d
#         """
#         return (a + b + c + d)*(a * d - b * c) ** 2 / [(a + b) * (c + d) * (b + d) * (a + c)]
