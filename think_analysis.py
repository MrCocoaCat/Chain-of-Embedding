import pickle
import time
import torch
import numpy as np
from numpy.lib.scimath import arccos
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# project_root_path = "E:/GitHub/Chain-of-Embedding/"
model_name = "Qwen2.5-7B-Instruct"
# model_name = "Qwen2.5-7B-Instruct"


# def generate_unique_index(start, end, length):
#     return torch.randperm(end - start)[:length].tolist()

def generate_unique_index(start, end, length):
    """
    生成指定范围内的唯一随机整数列表（纯Python实现）

    参数:
        start (int): 起始值（包含）
        end (int): 结束值（不包含）
        length (int): 需要生成的索引数量

    返回:
        list: 包含唯一随机整数的列表，范围为 [start, end)

    异常:
        ValueError: 当 length 大于可选择的范围时
    """
    if length > end - start:
        raise ValueError(f"无法生成 {length} 个唯一索引，范围 [{start}, {end}) 只有 {end - start} 个元素")
    return random.sample(range(start, end), length)



def get_cos_similar_matrix(v1, v2, device):
    """
    计算两个向量集合之间的余弦相似度矩阵，并将结果归一化到[0,1]范围

    参数:
    v1: 第一个向量集合，形状为[batch_size1, embedding_dim]
    v2: 第二个向量集合，形状为[batch_size2, embedding_dim]
    device: 计算设备（如'cuda'或'cpu'）

    返回:
    res: 归一化后的余弦相似度矩阵，形状为[batch_size1, batch_size2]
    """
    # 将输入张量移到指定设备
    v1 = v1.to(device)
    v2 = v2.to(device)
    # 计算分子：向量点积矩阵
    num = torch.mm(v1, v2.t())  # 形状为[batch_size1, batch_size2]
    # 计算分母：两个向量集合的L2范数乘积矩阵
    v1_norm = torch.norm(v1, dim=1).reshape(-1, 1)  # 形状为[batch_size1, 1]
    v2_norm = torch.norm(v2, dim=1)  # 形状为[batch_size2]
    denom = v1_norm * v2_norm  # 形状为[batch_size1, batch_size2]
    # 计算余弦相似度
    res = num / denom
    # 处理数值稳定性：将无穷大值（如除以零）替换为0
    res[torch.isinf(res)] = 0
    # 将余弦相似度从[-1,1]范围归一化到[0,1]范围
    # 公式：normalized = (original + 1) / 2 = 0.5 + 0.5 * original
    # 原始版本
    # res = 0.5 + 0.5 * res
    res = torch.arccos(res)
    # 将结果移回CPU以释放GPU内存
    res = res.cpu()
    # 删除中间变量以释放GPU内存
    del v1, v2, num, denom
    return res


# def conunt_attention(id):
#     sample_info = get_sample_info(id)
#     cos_list = []
#     hidden_states = sample_info["output"]["all_token_hidden_states"]
#     layer_num = len(hidden_states[1])
#     hidden_states_list = []
#     hidden_states = [h[0].cpu() for h in hidden_states]
#     hidden_states_list.append(hidden_states)
#     layer_intervals = 1
#     for i in range(len(hidden_states_list)):
#         for j in range(layer_num - layer_intervals + 1):
#             cosine = get_cos_similar_matrix(
#                 hidden_states_list[i][j][0],
#                 hidden_states_list[i][j + layer_intervals][0]
#             )
#             print(j, j + layer_intervals)
#             similarity = torch.trace(cosine) / cosine.size(0)
#             cos_list.append(similarity.item())
#             del cosine
#
#     return cos_list

def average_similarity(layer_cosine_similarity):
    return torch.tensor(layer_cosine_similarity).mean().item()

def get_cosine_similarity(device, layer_intervals, file_path):
    # assert len(dataset) > num_data
    # model = model.to(device)
    hidden_states_list = []
   # data_index = generate_unique_index(1, 100, 5)
   # data_index = [1, 15, 17, 19, 20,30,40,50,60,70,80,90]
    num_layer = 0

    #input_ids = torch.tensor(dataset[i]['input_ids'])
    #if len(input_ids.shape) != 2:
    #    input_ids = input_ids.reshape(1, -1)
    #input_ids = input_ids.to(device)

   # print(file_path_simple_think)
    print(file_path)
    with open(file_path, 'rb') as file:
        info = pickle.load(file)


    #cos_list = []
    hidden_states = info["output"]["all_token_hidden_states"][0]
    num_layer = len(info["output"]["all_token_hidden_states"][1])
    # print("num_layer", num_layer)
    # hidden_states = model(input_ids, output_hidden_states=True).hidden_states
    hidden_states = [h.cpu() for h in hidden_states]
    hidden_states_list.append(hidden_states)
        #del input_ids
     # cosine_similarity = [[] for _ in range(num_layer - layer_intervals + 1)]
    cosine_similarity = [[] for _ in range(num_layer - layer_intervals )]
    for i in range(len(hidden_states_list)):
        #for j in range(num_layer - layer_intervals + 1):
        for j in range(num_layer - layer_intervals ):
            # print(i,j,j+1)
            cosine = get_cos_similar_matrix(
                hidden_states_list[i][j][0],
                hidden_states_list[i][j + layer_intervals][0],
                device
            )
            similarity = torch.trace(cosine) / cosine.size(0)
            cosine_similarity[j].append(similarity.item())
            del cosine

    print('Calculating cosine similarity...')
    similarities = [average_similarity(layer_sim) for layer_sim in cosine_similarity]
    similarities_tensor = torch.tensor(similarities)
    best_layer = torch.argmax(similarities_tensor).item()
    best_cosine = similarities[best_layer]

    # for i, sim in enumerate(similarities):
    #
    #     print(f'The cosine similarity between hidden_states {i} and hidden_states {i + layer_intervals} is {sim:.4f}')

    print(
        f'The highest cosine similarity comes from hidden_states {best_layer} and hidden_states {best_layer + layer_intervals}, with a value of {best_cosine:.4f}')
    #model.cpu()
    #del hidden_states_list, model
    torch.cuda.empty_cache()
    return similarities


def save_list_to_txt(data_list, file_path):
    """将列表保存为文本文件"""
    with open(file_path, 'w') as f:
        for item in data_list:
            f.write(f"{item}\n")  # 每行一个元素

def read_list_from_txt(file_path):
    """从文本文件读取列表"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter


def visualize_data(data_list,
                   labels=None,
                   title="Data Trend Analysis",
                   xlabel="Data Index",
                   ylabel="Value",
                   ylim=None,
                   figsize=(10, 6),
                   use_log_scale=True,
                   log_base=10,
                   show_minor_ticks=True,
                   styles=None):
    """
    Visualize multiple data trends with points and connecting lines on one chart

    Args:
        data_list: 列表形式的多个1D数组或数值列表，如[[data1], [data2], ...]
        labels: 各组数据的标签列表，若为None则自动生成
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        ylim: Y-axis limits (lower, upper)
        figsize: Figure size tuple (width, height)
        use_log_scale: 是否使用对数刻度
        log_base: 对数底数，默认10
        show_minor_ticks: 是否显示次要刻度
        styles: 自定义样式列表，每个元素为包含线型、颜色、标记的字典
    """
    # 确保输入是列表形式
    if not isinstance(data_list, list):
        data_list = [data_list]

    # 处理标签
    if labels is None:
        labels = [f'Data {i + 1}' for i in range(len(data_list))]
    elif len(labels) != len(data_list):
        raise ValueError("标签数量必须与数据组数量一致")

    # 定义默认样式循环使用
    default_styles = [
        {'linestyle': '-', 'color': 'b', 'marker': 'o'},
        {'linestyle': '--', 'color': 'g', 'marker': 's'},
        {'linestyle': '-.', 'color': 'r', 'marker': '^'},
        {'linestyle': ':', 'color': 'c', 'marker': 'D'},
        {'linestyle': '-', 'color': 'm', 'marker': '*'},
        {'linestyle': '--', 'color': 'y', 'marker': '+'},
        {'linestyle': '-.', 'color': 'k', 'marker': 'x'}
    ]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制每组数据
    for i, data in enumerate(data_list):
        # Convert data to numpy array of floats
        data = np.asarray(data, dtype=float)

        # 获取样式
        if styles and i < len(styles):
            style = styles[i]
        else:
            style = default_styles[i % len(default_styles)]

        # Plot data with markers and lines
        ax.plot(data,
                linestyle=style['linestyle'],
                color=style['color'],
                marker=style['marker'],
                markersize=5,
                linewidth=1.5,
                alpha=0.7,
                label=labels[i])

    # Configure logarithmic scale if needed
    if use_log_scale:
        ax.set_yscale('log', base=log_base)
        # Major ticks
        ax.yaxis.set_major_locator(LogLocator(base=log_base))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        # Minor ticks
        if show_minor_ticks:
            ax.yaxis.set_minor_locator(LogLocator(base=log_base, subs='all'))
            ax.yaxis.set_minor_formatter(NullFormatter())

    # Set axis limits and labels
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7, which='both')
    ax.legend()

    # Improve layout
    plt.tight_layout()

    return fig

# def visualize_data(data,
#                    title="Data Trend Analysis",
#                    xlabel="Data Index",
#                    ylabel="Value",
#                    ylim=None,
#                    figsize=(10, 6),
#                    use_log_scale=True,
#                    log_base=10,
#                    show_minor_ticks=True):
#     """
#     Visualize data trends with points and connecting lines
#
#     Args:
#         data: 1D array or list of numerical values (as strings)
#         title: Chart title
#         xlabel: X-axis label
#         ylabel: Y-axis label
#         ylim: Y-axis limits (lower, upper)
#         figsize: Figure size tuple (width, height)
#         use_log_scale: 是否使用对数刻度
#         log_base: 对数底数，默认10
#         show_minor_ticks: 是否显示次要刻度
#     """
#     # Convert data to numpy array of floats
#     data = np.asarray(data, dtype=float)
#     # Create figure and axis
#     fig, ax = plt.subplots(figsize=figsize)
#     # Plot data with markers and lines
#     ax.plot(data, 'b-', alpha=0.7, marker='o', markersize=5, linewidth=1.5, label='Data Trend')
#
#     # # Add mean line
#     # mean_val = data.mean()
#     # ax.axhline(y=mean_val, color='black', linestyle='--', alpha=0.5,
#     #            label=f'Mean: {mean_val:.4f}')
#     # Configure logarithmic scale if needed
#     if use_log_scale:
#         ax.set_yscale('log', base=log_base)
#         # Major ticks
#         ax.yaxis.set_major_locator(LogLocator(base=log_base))
#         ax.yaxis.set_major_formatter(ScalarFormatter())
#         # Minor ticks
#         if show_minor_ticks:
#             ax.yaxis.set_minor_locator(LogLocator(base=log_base, subs='all'))
#             ax.yaxis.set_minor_formatter(NullFormatter())
#     # Set axis limits and labels
#     if ylim:
#         ax.set_ylim(ylim)
#     ax.set_xlabel(xlabel, fontsize=12)
#     ax.set_ylabel(ylabel, fontsize=12)
#     ax.set_title(title, fontsize=14)
#
#     # Add grid and legend
#     # ax.grid(True, linestyle='--', alpha=0.7, which='both')
#     # ax.legend()
#
#     # Improve layout
#     plt.tight_layout()
#
#     return fig

if __name__ == '__main__':



    dateset_name = "mgsm"
    start_time = time.time()
    # task_list = list(range(3))
    layer_intervals = 1
    file_path_simple_think = f"D:\\GitHub\\Chain-of-Embedding\\OutputInfo\\en\\Sample_info\\{model_name}\\{dateset_name}\\mgsm_serious_think.pkl"
    file_path_serious_think = f"D:\\GitHub\\Chain-of-Embedding\\OutputInfo\\en\\Sample_info\\{model_name}\\{dateset_name}\\mgsm_simple_think.pkl"
    similarities_simple = get_cosine_similarity(device, layer_intervals, file_path_simple_think)
    print(similarities_simple)

    similarities_serious = get_cosine_similarity(device, layer_intervals, file_path_serious_think)
    print(similarities_serious)
    #  = f"D:\\GitHub\\Chain-of-Embedding\\{model_name}_{dateset_name}_important_layer.txt"
    # save_list_to_txt(similarities,ImportantFilePath)

    fig = visualize_data(
        data_list=[similarities_simple, similarities_serious],
        labels=["similarities_simple", "similarities_serious" ],
        title="Multiple Data Trends Comparison",
        use_log_scale=False
    )
    plt.show()






