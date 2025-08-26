import torch.nn.functional as F
import os
import pickle
import math
from multiprocessing import Pool
import traceback
import torch
import time
import numpy as np


device =  'cpu'
#project_root_path = "E:/GitHub/Chain-of-Embedding/"


def get_sample_info(id,model,dateset):
    """
    保存sample_info
    :return:
    """
    file_path = f"F:\\GitHub\\Chain-of-Embedding\\OutputInfo\\en\\Sample_info\\{model}\\{dateset}\\{dateset}_{id}.pkl"
    with open(file_path, 'rb') as file:
        sample_info = pickle.load(file)
    return  sample_info

def normalize_to_255(arr):
    arr_scaled = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return (arr_scaled * 255).astype(np.uint8)

def count_base(id, layers, model, dateset):

    # 这个值也就是coe_mag
    layers_by_id = sorted(layers, key=lambda x: x.id)
    # 提取各列数据
    # linear = [layer.linear_coeffs for layer in layers_by_id]
    # exp = [layer.exp_coeffs for layer in layers_by_id]
    # step = [layer.step_coeffs for layer in layers_by_id]
    # log = [layer.log_coeffs for layer in layers_by_id]
    # reciprocal = [layer.reciprocal_coeffs for layer in layers_by_id]
    # quantile = [layer.quantile_coeffs for layer in layers_by_id]
    important = [layer.IsImportant for layer in layers_by_id]

    sample_info = get_sample_info(id,model,dateset)
    start_time = time.time()
    print(start_time)
    hidden_states = sample_info["output"]["all_token_hidden_states"]
    output_len = sample_info["output"]["output_len"]
    layer_num = len(hidden_states[1])
    hs_all_layer = []

    ############################################################

    for j in range(layer_num):
        z_list = []
        # if j!= 0 and important[j-1] == 0:
        #     hs_all_layer.append(0)
        #     continue
        for i in range(0, output_len):
            z = np.array(hidden_states[i][j][0][0].cpu())
            z_list.append(z)
        #  all_pos_hs 内部为每一个层的列表
        all_pos_hs = np.array(z_list)
        # 定义h_l ，
        # T 为all_pos_hs 的长度
        h = np.mean(all_pos_hs, axis=0)
        hs_all_layer.append(h)
     # compute_CoE_Mag


    # 系数a,计算二范数，即欧基里德范数，Zmag系数，系数的范围都是最有一层与第一层的差值
    norm_denominator_a = np.linalg.norm(hs_all_layer[-1] - hs_all_layer[0], ord=2)
    # 系数b 计算向量夹角（弧度）,#ZAng 系数，系数的范围都是最有一层与第一层的差值
    norm_denominator_te = np.dot(hs_all_layer[-1], hs_all_layer[0]) / (
            np.linalg.norm(hs_all_layer[-1], ord=2) * np.linalg.norm(hs_all_layer[0], ord=2))  # 计算向量夹角的余弦值
    norm_denominator_b = math.acos(norm_denominator_te)
    # 系数c
    al_repdiff_norm = []
    al_semdiff = []

    for i in range(len(hs_all_layer) - 1):
        # if important[i] == 0:
        #     al_repdiff_norm.append(0)
        #     al_semdiff.append(0)
        #     continue
        a = hs_all_layer[i]
        b = hs_all_layer[i +1 ]
        diff = b - a
        # # 计算差值的二范数
        norm_2 = np.linalg.norm(diff, ord=2)
        al_repdiff_norm.append(norm_2 / norm_denominator_a) # 除以系数，并添加到列表中
        # # 计算夹角
        dot_product = np.dot(a, b)  # 计算两个数组的点积
        norm_a = np.linalg.norm(a, ord=2)   # 分别计算二范数
        norm_b = np.linalg.norm(b, ord=2) # 分别计算二范数
        similarity = dot_product / (norm_a * norm_b) # 点积除以二范数
        similarity = similarity if similarity <= 1 else 1
        arccos_sim = math.acos(similarity)  # 计算反余弦值
        al_semdiff.append(arccos_sim / norm_denominator_b)
    #
    Mag = np.mean(np.array(al_repdiff_norm))

    # 求余弦角平均数
    Ang = np.mean(np.array(al_semdiff))
    coe_r =  Mag - Ang
    # coe_c
    al_semdiff_norm = np.array(al_semdiff)
    x_list = np.array([al_repdiff_norm[i] * math.cos(al_semdiff_norm[i]) for i in range(len(al_semdiff_norm))])
    y_list = np.array([al_repdiff_norm[i] * math.sin(al_semdiff_norm[i]) for i in range(len(al_semdiff_norm))])
    al_combdiff_x_ave = np.mean(x_list)
    al_combdiff_y_ave = np.mean(y_list)
    coe_c =  math.sqrt(al_combdiff_x_ave ** 2 + al_combdiff_y_ave ** 2)

    #mse_value_ave = np.mean(np.array(mse_value_list))
    ############################################################
    end_time = time.time()
    print(end_time)
    elapsed_time = end_time - start_time
    print(f"耗时: {elapsed_time:.6f}秒")
    return elapsed_time
    # print(f"********** {id}:CoE Score Info: **********\nMag {al_repdiff_ave}; Ang {al_semdiff_ave}; R {coe_r}; C {coe_c}\n")
    # # print(f"********** {id}:CoE Score Info: **********\nMag {al_repdiff_ave}; Ang {al_semdiff_ave}; R {coe_r}; C {coe_c}\n")
    # print(f"************{id} finished cost {elapsed_time}****************************************")
    # return {
    #     "Mag": Mag,
    #     "Ang": Ang,
    #     "R": coe_r,
    #     "C": coe_c,
    # }


class Layer:
    def __init__(self, id, arccosine):
        self.id = id
        self.arccosine = arccosine
        self.IsImportant = 0
        self.linear_coeffs = None
        self.exp_coeffs = None
        self.step_coeffs = None
        self.log_coeffs = None
        self.reciprocal_coeffs = None
        self.quantile_coeffs = None
        self.rank = None  # 添加 rank 属性

    def calculate_coefficients(self, rank, n):
        """根据排名计算所有类型的系数"""
        self.rank = rank  # 保存排名信息
        # 线性映射
        self.linear_coeffs = (n - rank + 1) / n

        # 指数衰减 (α=0.5)
        self.exp_coeffs = np.exp(-0.5 * rank)

        # 阶梯函数
        if rank <= n * 0.1:
            self.step_coeffs = 1.0
        elif rank <= n * 0.3:
            self.step_coeffs = 0.8
        elif rank <= n * 0.5:
            self.step_coeffs = 0.5
        else:
            self.step_coeffs = 0.2

        # 对数缩放
        self.log_coeffs = np.log(n - rank + 2) / np.log(n + 1)

        # 倒数加权
        self.reciprocal_coeffs = 1.0 / rank

        # Sigmoid函数 (β=0.5, mid=n/2)
        #self.sigmoid_coeffs = 1 / (1 + np.exp(-0.5 * (rank - n / 2)))

        # 分位数映射 (4分位数)
        if rank <= n * 0.25:
            self.quantile_coeffs = 1.0
        elif rank <= n * 0.5:
            self.quantile_coeffs = 0.8
        elif rank <= n * 0.75:
            self.quantile_coeffs = 0.5
        else:
            self.quantile_coeffs = 0.2

def read_list_from_txt(file_path):
    """从文本文件读取余弦值列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [float(line.strip()) for line in f if line.strip()]


def format_execution_time(seconds):
    """将执行时间格式化为易读的字符串（自动适配秒、毫秒、分钟）"""
    if seconds < 0.001:
        # 小于1毫秒，显示微秒
        return f"{seconds * 1e6:.2f} μs"
    elif seconds < 1.0:
        # 小于1秒，显示毫秒
        return f"{seconds * 1e3:.2f} ms"
    elif seconds < 60.0:
        # 小于1分钟，显示秒
        return f"{seconds:.2f} s"
    else:
        # 大于1分钟，显示分钟和秒
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes:.0f}m {seconds:.2f}s"

def compute_layer_status(model_name,dateset_name,task_start,task_end):
    ImportantFilePath = f"D:\\GitHub\\Chain-of-Embedding\\{model_name}_important_layer.txt"
    # 读取余弦值列表
    cos_list = read_list_from_txt(ImportantFilePath)
    n = len(cos_list)
    # 创建Layer对象列表（保留原始ID）
    layers = [Layer(i + 1, arccos) for i, arccos in enumerate(cos_list)]
    # 根据cosine值降序排序
    sorted_layers = sorted(layers, key=lambda x: x.arccosine, reverse=True)
    # 为排序后的层分配新排名并计算系数
    for rank, layer in enumerate(sorted_layers, 1):
        layer.calculate_coefficients(rank, n)
        # 根据系数判断是否重要（示例：使用自定义权重 > 1.5）
        if layer.linear_coeffs > 0.5:
            layer.IsImportant = 1

    # 打印处理结果
    task_list = list(range(task_start,task_end))
    cost_time_list = []
    for task_id in task_list:
        cost_time = count_base(task_id, sorted_layers, model_name, dateset_name)
        cost_time_list.append(cost_time)

    print(cost_time_list)
    average = sum(cost_time_list) / len(cost_time_list)
    print(average)



if __name__ == '__main__':

    compute_layer_status("Qwen2.5-7B-Instruct","mgsm",task_start=0,task_end=5)


