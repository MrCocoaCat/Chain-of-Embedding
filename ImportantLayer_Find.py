import pickle
import time
import torch
import numpy as np
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_root_path = "E:/GitHub/Chain-of-Embedding/"
ImportantFilePath = "D:\\GitHub\\Chain-of-Embedding\\important_layer.txt"




def generate_unique_index(start, end, length):
    return torch.randperm(end - start)[:length].tolist()

def get_sample_info(id):
    """
    保存sample_info
    :return:
    """
    file_path = f"D:\\GitHub\\Chain-of-Embedding\\OutputInfo\\en\\Sample_info\\Qwen2.5-7B-Instruct\\commonsenseqa\\commonsenseqa_{id}.pkl"
    with open(file_path, 'rb') as file:
        sample_info = pickle.load(file)
    return  sample_info


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
    res = 0.5 + 0.5 * res
    # 将结果移回CPU以释放GPU内存
    res = res.cpu()
    # 删除中间变量以释放GPU内存
    del v1, v2, num, denom
    return res


def conunt_attention(id):
    sample_info = get_sample_info(id)
    cos_list = []
    hidden_states = sample_info["output"]["all_token_hidden_states"]
    layer_num = len(hidden_states[1])
    hidden_states_list = []
    hidden_states = [h[0].cpu() for h in hidden_states]
    hidden_states_list.append(hidden_states)
    layer_intervals = 1
    for i in range(len(hidden_states_list)):
        for j in range(layer_num - layer_intervals + 1):
            cosine = get_cos_similar_matrix(
                hidden_states_list[i][j][0],
                hidden_states_list[i][j + layer_intervals][0]
            )
            print(j, j + layer_intervals)
            similarity = torch.trace(cosine) / cosine.size(0)
            cos_list.append(similarity.item())
            del cosine

    return cos_list

def average_similarity(layer_cosine_similarity):
    return torch.tensor(layer_cosine_similarity).mean().item()

def get_cosine_similarity(device, layer_intervals, num_layer):
    # assert len(dataset) > num_data
    # model = model.to(device)
    hidden_states_list = []
    data_index = generate_unique_index(0, 800, 20)
   # data_index = [1, 15, 17, 19, 20,30,40,50,60,70,80,90]
    for i in tqdm(data_index, desc="Collecting hidden states"):
        #input_ids = torch.tensor(dataset[i]['input_ids'])
        print(f"radom id is {i}")
        #if len(input_ids.shape) != 2:
        #    input_ids = input_ids.reshape(1, -1)
        #input_ids = input_ids.to(device)
        sample_info = get_sample_info(i)
        #cos_list = []
        hidden_states = sample_info["output"]["all_token_hidden_states"][0]
        num_layer = len(sample_info["output"]["all_token_hidden_states"][1])
        # print("num_layer", num_layer)
        #hidden_states = model(input_ids, output_hidden_states=True).hidden_states
        hidden_states = [h.cpu() for h in hidden_states]
        hidden_states_list.append(hidden_states)

        #del input_ids

    cosine_similarity = [[] for _ in range(num_layer - layer_intervals + 1)]
    cosine_similarity = [[] for _ in range(num_layer - layer_intervals )]
    for i in range(len(hidden_states_list)):
        #for j in range(num_layer - layer_intervals + 1):
        for j in range(num_layer - layer_intervals ):
            print(i,j,j+1)
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

    for i, sim in enumerate(similarities):
        print(f'The cosine similarity between hidden_states {i} and hidden_states {i + layer_intervals} is {sim:.4f}')

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


if __name__ == '__main__':
    start_time = time.time() 
    task_list = list(range(3))
    layer_intervals = 1
    num_layer = 29
    similarities = get_cosine_similarity(device, layer_intervals, num_layer)
    print(similarities)
    save_list_to_txt(similarities,ImportantFilePath)


