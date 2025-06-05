# import numpy as np
# import matplotlib.pyplot as plt
#
# n = 100  # 总项数
# ranks = np.arange(1, n+1)
#
# # 计算各种系数
# linear = (n - ranks + 1) / n
# exp = np.exp(-0.5 * ranks)
#
# step = np.zeros_like(ranks, dtype=float)
# step[ranks <= n*0.1] = 1.0
# step[(ranks > n*0.1) & (ranks <= n*0.3)] = 0.8
# step[(ranks > n*0.3) & (ranks <= n*0.5)] = 0.5
# step[ranks > n*0.5] = 0.2
#
# log = np.log(n - ranks + 2) / np.log(n + 1)
# reciprocal = 1.0 / ranks
# sigmoid = 1 / (1 + np.exp(-0.5 * (ranks - n/2)))
#
# quantile = np.zeros_like(ranks, dtype=float)
# quantile[ranks <= n*0.25] = 1.0
# quantile[(ranks > n*0.25) & (ranks <= n*0.5)] = 0.8
# quantile[(ranks > n*0.5) & (ranks <= n*0.75)] = 0.5
# quantile[ranks > n*0.75] = 0.2
#
# custom = np.ones_like(ranks, dtype=float)
# custom[ranks == 1] = 3.0
# custom[ranks == 2] = 2.0
# custom[ranks == 3] = 1.5
#
# # 绘制图表
# plt.figure(figsize=(12, 8))
# plt.plot(ranks, linear, label='Linear')
# plt.plot(ranks, exp, label='Exponential')
# plt.plot(ranks, step, label='Step')
# plt.plot(ranks, log, label='Logarithmic')
# plt.plot(ranks, reciprocal, label='Reciprocal')
# #plt.plot(ranks, sigmoid, label='Sigmoid')
# plt.plot(ranks, quantile, label='Quantile')
# #plt.plot(ranks, custom, label='Custom')
#
# plt.xlabel('Rank')
# plt.ylabel('Coefficient Value')
# plt.title('Comparison of Different Ranking Coefficient Functions')
# plt.grid(True)
# plt.legend()
# plt.show()