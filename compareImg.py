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


def get_sample_info(id, model,dateset):
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

def count_base(id,layers,model,dateset):
    try:
        start_time = time.time()
        sample_info = get_sample_info(id,model,dateset)
        output_scores = sample_info["output"]["output_scores"] # 输出层，264，每个tensor 为(1,152064)
        all_token_re = []  # 将每个layer转换为以为数组，并进行归一化处理
        all_token_max_re = []  # 存储每一个layer 的最大值
        for token in range(len(output_scores)):
            re = output_scores[token][0].tolist()  # 每一层转换为一维的数组
            re = F.softmax(torch.tensor(re), 0).cpu().tolist()  # softmax 函数的作用是将输入张量的每个元素转换为一个概率值，使得所有元素的概率之和为 1。第二个参数 0 表示在第 0 维上进行 softmax 操作。例如，对于一个形状为 (n,) 的一维张量，softmax 操作会将每个元素转换为一个概率值，使得这些概率值的和为 1
            all_token_re.append(re)
            all_token_max_re.append(max(re))
        # (3) Maximum Softmax Probability
        maxprob = np.mean(all_token_max_re) # ，最后求平均值
        # (4) Perplexity
        seq_ppl_list = [math.log(max_re) for max_re in all_token_max_re]  # 求每个元素的对数
        ppl = -np.mean(seq_ppl_list)  # 再对所有的对数求平均值
        # (5) entropy
        from scipy.stats import entropy
        seq_entropy_list = [entropy(re, base=2) for re in all_token_re] # 对self.all_token_re 中的每个元素，以2为底计算熵
        entropy = np.mean(seq_entropy_list) # 求平均数
        print(f"**********{id}: Output Info: **********\nmaxprob {maxprob}; perplexity {ppl}; entropy {entropy}\n")

        # 在使用 transformers 库中的预训练模型（如 BERT、GPT 等）进行推理时，模型会对输入的文本进行一系列的计算和转换，这些计算过程会在不同的层产生隐藏状态。
        # all_token_hidden_states 就是一个包含了模型所有层（包括嵌入层）的隐藏状态的元组。
        # 每一个元素对应着模型某一层的隐藏状态，其形状通常为 (batch_size, sequence_length, hidden_size)，其中：
        # batch_size 表示一次处理的样本数量。
        # sequence_length 表示输入序列的长度。
        # hidden_size 表示模型隐藏层的维度。
        hidden_states = sample_info["output"]["all_token_hidden_states"]
        output_len = sample_info["output"]["output_len"]
        layer_num = len(hidden_states[1])
        print(f"layer_num {layer_num}")
        hs_all_layer = []
        #hs_all_layer_pic = []
        for j in range(layer_num):
            z_list = []
            for i in range(0, output_len):
                z = np.array(hidden_states[i][j][0][0].cpu())
                z_list.append(z)
            #all_pos_hs 内部为每一个层的列表
            all_pos_hs = np.array(z_list)
            # 定义h_l ，
            # T 为all_pos_hs 的长度
            h = np.mean(all_pos_hs, axis=0)
            #hs_all_layer_pic.append(all_pos_hs)
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
        # mse_value_list = []
        # glcm_distance_list = []

        # 这个值也就是coe_mag
        layers_by_id = sorted(layers, key=lambda x: x.id)
        # 提取各列数据
        linear = [layer.linear_coeffs for layer in layers_by_id]
        exp = [layer.exp_coeffs for layer in layers_by_id]
        step = [layer.step_coeffs for layer in layers_by_id]
        log = [layer.log_coeffs for layer in layers_by_id]
        reciprocal = [layer.reciprocal_coeffs for layer in layers_by_id]
        quantile = [layer.quantile_coeffs for layer in layers_by_id]
        important = [layer.IsImportant for layer in layers_by_id]

        for i in range(len(hs_all_layer) - 1):
            a = hs_all_layer[i]
            b = hs_all_layer[i +1 ]
            diff = b - a
            # # 计算差值的二范数
            norm_2 = np.linalg.norm(diff, ord=2)
            al_repdiff_norm.append(norm_2 / norm_denominator_a) # 除以系数，并添加到列表中
            # norm_2_list.append(norm_2)
            # norm_2_qz2_list.append((len(hs_all_layer) - i) * norm_2/ norm_denominator_a)
            # # 计算夹角
            dot_product = np.dot(a, b)  # 计算两个数组的点积
            norm_a = np.linalg.norm(a, ord=2)   # 分别计算二范数
            norm_b = np.linalg.norm(b, ord=2) # 分别计算二范数
            similarity = dot_product / (norm_a * norm_b) # 点积除以二范数
            similarity = similarity if similarity <= 1 else 1
            arccos_sim = math.acos(similarity)  # 计算反余弦值
            al_semdiff.append(arccos_sim / norm_denominator_b)

        # # 计算方差。方差用于衡量一组数据的离散程度，也就是数据相对于均值的分散情况。方差越大，说明数据越分散；方差越小，说明数据越集中在均值附近。
        # al_repdiff_var = np.var(np.array(al_repdiff_norm))
        #

        Mag = np.mean(np.array(al_repdiff_norm))
        important_layer = np.array(important) * np.array(al_repdiff_norm)
        Mag_linear = np.mean(important_layer * np.array(linear))
        Mag_exp = np.mean(important_layer * np.array(exp))
        Mag_step = np.mean(important_layer * np.array(step))
        Mag_log = np.mean(important_layer * np.array(log))
        # Mag_reciprocal = np.mean(important_layer * np.array(reciprocal))
        Mag_quantile = np.mean(important_layer * np.array(quantile))

        # 求余弦角平均数
        Ang = np.mean(np.array(al_semdiff))
        important_layer = np.array(important) * np.array(al_semdiff)
        Ang_linear = np.mean(important_layer * np.array(linear))
        Ang_exp = np.mean(important_layer * np.array(exp))
        Ang_step = np.mean(important_layer * np.array(step))
        Ang_log = np.mean(important_layer * np.array(log))
       # Ang_reciprocal = np.mean(important_layer * np.array(reciprocal))
        Ang_quantile = np.mean(important_layer * np.array(quantile))

        # # 计算方差，未使用
        # al_semdiff_var = np.var(np.array(al_semdiff_norm))
        # #coe_r ,夹角和二范数的结合

        coe_r =  Mag - Ang
        # coe_c
        al_semdiff_norm = np.array(al_semdiff)
        x_list = np.array([al_repdiff_norm[i] * math.cos(al_semdiff_norm[i]) for i in range(len(al_semdiff_norm))])
        y_list = np.array([al_repdiff_norm[i] * math.sin(al_semdiff_norm[i]) for i in range(len(al_semdiff_norm))])
        al_combdiff_x_ave = np.mean(x_list)
        al_combdiff_y_ave = np.mean(y_list)
        coe_c =  math.sqrt(al_combdiff_x_ave ** 2 + al_combdiff_y_ave ** 2)

        #mse_value_ave = np.mean(np.array(mse_value_list))
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"********** {id}:CoE Score Info: **********\nMag {al_repdiff_ave}; Ang {al_semdiff_ave}; R {coe_r}; C {coe_c}\n")
        print(f"************{id} finished cost {elapsed_time}****************************************")
        return {
            "Mag": Mag,
            "Mag_linear":Mag_linear,
            "Mag_exp":Mag_exp,
            "Mag_step":Mag_step,
            "Mag_log":Mag_log,
           # "Mag_reciprocal":Mag_reciprocal,
            "Mag_quantile":Mag_quantile,
            "Ang": Ang,
            "Ang_linear": Ang_linear,
            "Ang_exp": Ang_exp,
            "Ang_step": Ang_step,
            "Ang_log": Ang_log,
          #  "Ang_reciprocal": Ang_reciprocal,
            "Ang_quantile": Ang_quantile,
            "R": coe_r,
            "C": coe_c,
            "maxprob": maxprob,
            "ppl":ppl,
            "entropy":entropy,

        }
    except:
        stack_trace = traceback.format_exc()
        print(stack_trace)


def work(id,sorted_layers,model_name,dataset):
    pid = os.getpid()
    str_out = f"PID {pid} ----Processing task {id}  "
    print(f"***{str_out}***")
    base_dict = None
    try:
        base_line_re = count_base(id,sorted_layers,model_name,dataset)
        base_dict = {"id": id, "val": base_line_re}
    except:
        stack_trace = traceback.format_exc()
        print(stack_trace)
    return base_dict

#

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
        # 分位数映射 (4分位数)
        if rank <= n * 0.25:
            self.quantile_coeffs = 1.0
        elif rank <= n * 0.5:
            self.quantile_coeffs = 0.8
        elif rank <= n * 0.75:
            self.quantile_coeffs = 0.5
        else:
            self.quantile_coeffs = 0.2
        # 倒数加权
        # self.reciprocal_coeffs = 1.0 / rank
        # Sigmoid函数 (β=0.5, mid=n/2)
        # self.sigmoid_coeffs = 1 / (1 + np.exp(-0.5 * (rank - n / 2)))


def read_list_from_txt(file_path):
    """从文本文件读取余弦值列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [float(line.strip()) for line in f if line.strip()]


def compute_layer_status(model_name,dateset_name,task_start,task_end):
    start_time = time.time()
    # 读取总要层的信息
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
    print(f"处理完成，共 {n} 个层，耗时: {time.time() - start_time:.2f} 秒")
    # 打印所有层的详细参数
    print("\n所有层的详细参数:")
    print(f"{'原始ID':<8}{'ARCCosine':<10}{'新排名':<6}{'Linear':<10}{'Exp':<10}"
          f"{'Log':<10}{'Reciprocal':<12}{'Quantile':<10}{'Important'}")
    for layer in layers:
        print(f"{layer.id:<8}&{layer.arccosine:<10.4f}&{layer.rank:<6}&{layer.linear_coeffs:<10.4f}&"
              f"{layer.exp_coeffs:<10.4f}&{layer.log_coeffs:<10.4f}&"
           #   f"{layer.reciprocal_coeffs:<12.4f}&"
              f"{layer.quantile_coeffs:<10.4f}&{layer.IsImportant}" + "\\")
    # exit()
    start_time = time.time()
    task_list = list(range(task_start,task_end))
    res_l = []
    p = Pool(24)
    for task_id in task_list:
        res = p.apply_async(work, args=(task_id, layers,model_name,dateset_name))
        res_l.append(res)
    p.close()
    p.join()
    # print([res.get() for res in res_l]) #该结果已经传给回调函数处理了
    ssim_list = [res.get() for res in res_l]
    base_dict_list = {}
    for base_dict in ssim_list:
        id = base_dict["id"]
        val = base_dict["val"]
        base_dict_list[id] = val
    file_path = f'D:\\GitHub\\Chain-of-Embedding\\LayerState\\{model_name}\\{dateset_name}\\base_list.pickle'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    # 以二进制写入模式打开文件
    with open(file_path, 'wb') as file:
        img_list = {"base_list": base_dict_list}
        # 使用 pickle.dump 方法将数组写入文件
        pickle.dump(img_list, file)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(len(base_dict_list))
    print(f"数组已成功保存到 {file_path}")
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"程序运行时间: {minutes} 分 {seconds} 秒")


if __name__ == '__main__':

    model = ["Qwen2.5-7B-Instruct","Llama-3-8B-Instruct"]
    dataset = ["commonsenseqa","mgsm","commonsenseqa"]
    p = []


    m = "Qwen2.5-7B-Instruct"
    d = "mgsm"
    compute_layer_status(m,d,task_start=0,task_end=500)

