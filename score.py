import os
import sys
import time

import scipy.spatial
from scipy.stats import entropy
import math
import json

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class OutputScoreInfo:
    def __init__(self, output_scores):
        self.output_scores = output_scores
        self.all_token_re = [] # 将每个layer转换为以为数组，并进行归一化处理
        self.all_token_max_re = [] # 存储每一个layer 的最大值
        for token in range(len(self.output_scores)):
            re = self.output_scores[token][0].tolist()
            re = F.softmax(torch.tensor(re).to(device), 0).cpu().tolist() # 归一化处理
            self.all_token_re.append(re)
            self.all_token_max_re.append(max(re))

    def compute_maxprob(self):
        # (3) Maximum Softmax Probability
        seq_prob_list = self.all_token_max_re
        max_prob = np.mean(seq_prob_list) # 平均值
        return max_prob

    def compute_ppl(self):
        # (4) Perplexity
        seq_ppl_list = [math.log(max_re) for max_re in self.all_token_max_re] # 求每个元素的对数
        ppl = -np.mean(seq_ppl_list) # 再对所有的对数求平均值
        return ppl

    def compute_entropy(self):
        # (5) entropy
        seq_entropy_list = [entropy(re, base=2) for re in self.all_token_re] # 对self.all_token_re 中的每个元素，以2为底计算熵
        seq_entropy = np.mean(seq_entropy_list) # 求平均数
        return seq_entropy


class CoEScoreInfo:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states

    def compute_CoE_Mag(self):
        hs_all_layer = self.hidden_states
        layer_num = len(hs_all_layer)

        norm_denominator = np.linalg.norm(hs_all_layer[-1] - hs_all_layer[0], ord=2)             # 依次计算二范数，即欧基里德范数，Zmag系数
        al_repdiff = np.array([hs_all_layer[i+1] - hs_all_layer[i] for i in range(layer_num - 1)]) # 依次计算每两层的差值
        al_repdiff_norm = [np.linalg.norm(item, ord=2) / norm_denominator for item in al_repdiff]  # 计算二范数除以系数
        al_repdiff_ave = np.mean(np.array(al_repdiff_norm)) # 计算均值。均值也称为平均数，它是一组数据的总和除以数据的个数。均值能够反映出这组数据的中心位置或典型水平。
        al_repdiff_var = np.var(np.array(al_repdiff_norm)) # 计算方差。方差用于衡量一组数据的离散程度，也就是数据相对于均值的分散情况。方差越大，说明数据越分散；方差越小，说明数据越集中在均值附近。
        return al_repdiff_norm, al_repdiff_ave, al_repdiff_var


    def compute_CoE_Ang(self):
        hs_all_layer = self.hidden_states
        layer_num = len(hs_all_layer)

        al_semdiff = []
        norm_denominator = np.dot(hs_all_layer[-1], hs_all_layer[0]) / (np.linalg.norm(hs_all_layer[-1], ord=2) * np.linalg.norm(hs_all_layer[0], ord=2)) # 计算向量夹角的余弦值
        norm_denominator = math.acos(norm_denominator) # # 计算向量夹角（弧度）
        for i in range(layer_num - 1):
            a = hs_all_layer[i + 1]
            b = hs_all_layer[i]
            dot_product = np.dot(a, b)  # 计算两个数组的点积
            norm_a, norm_b = np.linalg.norm(a, ord=2), np.linalg.norm(b, ord=2)  # 分别计算二范数
            similarity = dot_product / (norm_a * norm_b)
            similarity = similarity if similarity <= 1 else 1

            arccos_sim = math.acos(similarity) # 计算反余弦值
            al_semdiff.append(arccos_sim / norm_denominator) # 除以系数，并添加到列表中

        al_semdiff_norm = np.array(al_semdiff)
        al_semdiff_ave = np.mean(np.array(al_semdiff_norm))
        al_semdiff_var = np.var(np.array(al_semdiff_norm))
        
        return al_semdiff_norm, al_semdiff_ave, al_semdiff_var

    def compute_CoE_R(self):
        _, al_repdiff_ave, _ = self.compute_CoE_Mag()
        _, al_semdiff_ave, _ = self.compute_CoE_Ang()

        return al_repdiff_ave - al_semdiff_ave

    def compute_CoE_C(self):
        al_repdiff_norm, _, _ = self.compute_CoE_Mag()
        al_semdiff_norm, _, _ = self.compute_CoE_Ang()
        x_list = np.array([al_repdiff_norm[i] * math.cos(al_semdiff_norm[i]) for i in range(len(al_semdiff_norm))])
        y_list = np.array([al_repdiff_norm[i] * math.sin(al_semdiff_norm[i]) for i in range(len(al_semdiff_norm))])
        al_combdiff_x_ave = np.mean(x_list)
        al_combdiff_y_ave = np.mean(y_list)
        al_combdiff_x_var = np.mean(x_list)
        al_combdiff_y_var = np.mean(y_list)

        return math.sqrt(al_combdiff_x_ave ** 2 + al_combdiff_y_ave ** 2)
