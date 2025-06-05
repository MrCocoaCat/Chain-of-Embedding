import torch.nn.functional as F
from skimage.metrics import structural_similarity
import cv2
from skimage.feature import graycomatrix, graycoprops
import os
import pickle
import math
import torch
from multiprocessing import Pool
import time
import traceback
import numpy as np

import torch
from tqdm import tqdm
import gc
device = 'cuda'
project_root_path = "E:/GitHub/Chain-of-Embedding/"

import numpy as np


def compress_2d_array(arr, n, dim):
    """
    对二维数组指定维度进行压缩
    :param arr: 输入的二维 numpy 数组
    :param n: 压缩后的目标维度大小
    :param dim: 指定要压缩的维度，0 表示行维度，1 表示列维度
    :return: 压缩后的二维数组
    """
    if dim == 0:
        # 压缩行维度
        original_rows, cols = arr.shape
        if n > original_rows:
            raise ValueError("压缩后的行数不能大于原始行数")

        # 计算每个目标行应包含的原始行数量
        rows_per_group = original_rows / n

        # 创建结果数组
        compressed = np.zeros((n, cols))

        # 按目标维度大小进行分组压缩
        for i in range(n):
            start_idx = int(i * rows_per_group)
            end_idx = int((i + 1) * rows_per_group)

            # 处理最后一组可能需要包含剩余所有行的情况
            if i == n - 1:
                end_idx = original_rows

            compressed[i] = arr[start_idx:end_idx].mean(axis=0)

        return compressed

    elif dim == 1:
        # 压缩列维度
        rows, original_cols = arr.shape
        if n > original_cols:
            raise ValueError("压缩后的列数不能大于原始列数")

        # 计算每个目标列应包含的原始列数量
        cols_per_group = original_cols / n

        # 创建结果数组
        compressed = np.zeros((rows, n))

        # 按目标维度大小进行分组压缩
        for i in range(n):
            start_idx = int(i * cols_per_group)
            end_idx = int((i + 1) * cols_per_group)

            # 处理最后一组可能需要包含剩余所有列的情况
            if i == n - 1:
                end_idx = original_cols

            compressed[:, i] = arr[:, start_idx:end_idx].mean(axis=1)

        return compressed

    else:
        raise ValueError("维度参数 dim 必须为 0 或 1")

def compress_2d_array_old(arr, n, dim):
    """
    对二维数组指定维度进行压缩
    :param arr: 输入的二维 numpy 数组
    :param n: 分组大小
    :param dim: 指定要压缩的维度，0 表示行维度，1 表示列维度
    :return: 压缩后的二维数组
    """
    if dim == 0:
        num_rows, num_cols = arr.shape
        num_full_groups = num_rows // n
        reshaped = arr[:num_full_groups * n].reshape(-1, n, num_cols)
        means = reshaped.mean(axis=1)
        remaining = arr[num_full_groups * n:]
        if remaining.size > 0:
            remaining_mean = remaining.mean(axis=0, keepdims=True)
            means = np.vstack((means, remaining_mean))
        return means
    elif dim == 1:
        num_rows, num_cols = arr.shape
        num_full_groups = num_cols // n
        reshaped = arr[:, :num_full_groups * n].reshape(num_rows, -1, n)
        means = reshaped.mean(axis=2)
        remaining = arr[:, num_full_groups * n:]
        if remaining.size > 0:
            remaining_mean = remaining.mean(axis=1, keepdims=True)
            means = np.hstack((means, remaining_mean))
        return means
    else:
        raise ValueError("维度参数 dim 必须为 0 或 1")


def compare_images(img1_path, img2_path):
    # 读取两张图片
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # 创建 FAST 特征检测器
    fast = cv2.FastFeatureDetector_create()
    # 检测特征点
    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)

    # 创建 BRIEF 描述符提取器
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # 计算 BRIEF 描述符
    kp1, des1 = brief.compute(img1, kp1)
    kp2, des2 = brief.compute(img2, kp2)

    # 创建 BFMatcher 对象，使用汉明距离进行匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 进行特征匹配
    matches = bf.knnMatch(des1, des2, k=2)

    # 比率测试筛选可靠匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)






def contrast_image_SSIM(imageA, imageB):
    """
    结构相似性指数（SSIM）
    对比两张图片的相似度，相似度等于1 完美匹配
    :param imageA: 第一张图片的路径
    :param imageB: 第二张图片的路径
    :return: 相似度得分和差异图像
    """
    try:
        # 读取图片
        imageA = cv2.imread(imageA)
        imageB = cv2.imread(imageB)
        if imageA is None or imageB is None:
            print("无法读取图片，请检查文件路径是否正确。")
            return None, None
        # 转换为灰度图像
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        # 检查图像尺寸是否一致
        if grayA.shape != grayB.shape:
            print("两张图片的尺寸不一致，请使用相同尺寸的图片。")
            return None, None
        # 计算两个灰度图像之间的结构相似度指数,相似度等于1完美匹配
        (score, diff) = structural_similarity(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        # print("SSIM:{}".format(score))
        return score, diff
    except Exception as e:
        print(f"发生错误: {e}")
        return None, None


def contrast_image_SSIM(imageA, imageB):
    """
    结构相似性指数（SSIM）
    对比两张图片的相似度，相似度等于1 完美匹配
    :param imageA: 第一张图片的路径
    :param imageB: 第二张图片的路径
    :return: 相似度得分和差异图像
    """
    try:
        # 读取图片
        imageA = cv2.imread(imageA)
        imageB = cv2.imread(imageB)
        if imageA is None or imageB is None:
            print("无法读取图片，请检查文件路径是否正确。")
            return None, None
        # 转换为灰度图像
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        # 检查图像尺寸是否一致
        if grayA.shape != grayB.shape:
            print("两张图片的尺寸不一致，请使用相同尺寸的图片。")
            return None, None
        # 计算两个灰度图像之间的结构相似度指数,相似度等于1完美匹配
        (score, diff) = structural_similarity(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        # print("SSIM:{}".format(score))
        return score, diff
    except Exception as e:
        print(f"发生错误: {e}")
        return None, None

def tensorToImg(id):
    """
    J将tensor 转换为灰度图
    :return:
    """
    file_path = f"C:\\Users\\liyubo\\Documents\\GitHub\\Chain-of-Embedding\\OutputInfo\\en\\HiddenLayer\\Qwen2.5-7B-Instruct\\commonsenseqa\\commonsenseqa_{id}.pt"
    print(f" 读取 {file_path} Tensor")
    loaded_tensor = torch.load(file_path)
    for i in range(len(loaded_tensor)):
        # 去掉多余的维度
        tensor = loaded_tensor[i].flatten().cpu()
        print("去掉多余维度后Tensor的维度:", tensor.shape)
        height = math.isqrt(len(tensor)) +1
        width = math.isqrt(len(tensor)) +1
        new_shape = height *height
        pad_length = new_shape - len(tensor)
        padded_tensor = F.pad(tensor, (0, pad_length), 'constant', 0)
        # 重塑Tensor为三维
        tensor_2d = padded_tensor.reshape(height, width)
        numpy_array = tensor_2d.numpy()
        # 检查数组元素范围，如果不在 [0, 1] 内，进行归一化处理
        if numpy_array.min() < 0 or numpy_array.max() > 1:
            numpy_array = cv2.normalize(numpy_array, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # 如果数组是单通道，OpenCV 认为它已经是灰度图，无需转换
        # 如果需要将其转换为 OpenCV 能直接显示的 uint8 类型
        image_opencv = (numpy_array * 255).astype(np.uint8)
        #cv2.imshow('Grayscale Image', image_opencv)
        img_path = f"C:\\Users\\liyubo\\Documents\\GitHub\\Chain-of-Embedding\\OutputImg\\Qwen2.5-7B-Instruct\\commonsenseqa\\commonsenseqa_{id}\\{i}.jpg"
        base_path = f"C:\\Users\\liyubo\\Documents\\GitHub\\Chain-of-Embedding\\OutputImg\\Qwen2.5-7B-Instruct\\commonsenseqa\\commonsenseqa_{id}"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        cv2.imwrite(img_path, image_opencv)
        print(f"灰度图已保存到 {img_path}")


import numpy as np
from skimage import feature, color


def get_cos_similar_matrix(v1, v2):
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
   # res = 0.5 + 0.5 * res

    # 将结果移回CPU以释放GPU内存
    res = res.cpu()

    # 删除中间变量以释放GPU内存
    del v1, v2, num, denom

    return res


def calculate_glcm_features(img_array):
    """计算图像数组的灰度共生矩阵特征"""
    # 如果是彩色图像，转换为灰度图
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = color.rgb2gray(img_array)

    # 将图像转换为整数类型（GLCM要求）
    img_array = (img_array * 255).astype(np.uint8)

    # 计算GLCM
    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = feature.graycomatrix(img_array, distances, angles, levels=256, symmetric=True, normed=True)

    # 提取特征
    contrast = feature.graycoprops(glcm, 'contrast')
    dissimilarity = feature.graycoprops(glcm, 'dissimilarity')
    homogeneity = feature.graycoprops(glcm, 'homogeneity')
    energy = feature.graycoprops(glcm, 'energy')
    correlation = feature.graycoprops(glcm, 'correlation')
    asm = feature.graycoprops(glcm, 'ASM')

    # 将特征展平并连接
    features = np.hstack([
        contrast.flatten(),
        dissimilarity.flatten(),
        homogeneity.flatten(),
        energy.flatten(),
        correlation.flatten(),
        asm.flatten()
    ])

    return features


def compare_images_glcm(img_array1, img_array2):
    """比较两张图片的GLCM特征相似度"""
    # 计算两张图片的灰度共生矩阵特征值
    features1 = calculate_glcm_features(img_array1)
    features2 = calculate_glcm_features(img_array2)

    # 计算特征值之间的欧氏距离
    distance = np.linalg.norm(features1 - features2)
    return distance

def count_img(id):
    img_path_dir = f"E:\\GitHub\\Chain-of-Embedding\\OutputImg\\Qwen2.5-7B-Instruct\\commonsenseqa\\commonsenseqa_{id}\\"
    # 初始化文件计数器
    file_count = 0
    # 遍历文件夹中的所有文件和子文件夹
    for item in os.listdir(img_path_dir):
        item_path = os.path.join(img_path_dir, item)
        if os.path.isfile(item_path):
            file_count += 1
    print(f"文件夹 {img_path_dir} 中的文件数量为: {file_count}")
    sum_ssim =  0
    range_num = file_count
    glcm_distence_sum = 0
    for i in range(range_num-1):
        img_path1 = img_path_dir +f"{i}.jpg"
        img_path2 = img_path_dir+ f"{i+1}.jpg"
        # score 越大。则越相似
        score ,diff_matrix= contrast_image_SSIM(img_path1, img_path2)
        sum_ssim = sum_ssim + (1-score)
        distance = compare_images_glcm(img_path1, img_path2)
        # print(f"两张图片的灰度共生矩阵特征值之间的欧氏距离为: {distance}")
        glcm_distence_sum = glcm_distence_sum + distance

    average_ssim = sum_ssim / range_num
    average_glcm_distence= glcm_distence_sum / range_num
    return average_ssim,average_glcm_distence

def count_Matrix(id):
    file_path = f"E:\\GitHub\\Chain-of-Embedding\\OutputInfo\\en\\HiddenLayer\\Qwen2.5-7B-Instruct\\commonsenseqa\\commonsenseqa_{id}.pt"
    print(f" 读取 {file_path} Tensor")
    loaded_tensor = torch.load(file_path)
    matrix_list = []
    for i in range(len(loaded_tensor)):
        # 去掉多余的维度
        tensor = loaded_tensor[i].flatten().cpu()
        #print("去掉多余维度后Tensor的维度:", tensor.shape)
        height = math.isqrt(len(tensor)) + 1
        width = math.isqrt(len(tensor)) + 1
        new_shape = height * height
        pad_length = new_shape - len(tensor)
        padded_tensor = F.pad(tensor, (0, pad_length), 'constant', 0)
        # 重塑Tensor为三维
        tensor_2d = padded_tensor.reshape(height, width)
        numpy_array = tensor_2d.numpy()
        matrix_list.append(numpy_array)
        # 检查数组元素范围，如果不在 [0, 1] 内，进行归一化处理
        diff_score_sum = 0
    for i in range(len(loaded_tensor)-1):
        matrix1 = matrix_list[i]
        matrix2 = matrix_list[i+1]
        # 合并两个矩阵以找到整体的最小值和最大值
        combined_matrix = np.concatenate((matrix1.flatten(), matrix2.flatten()))
        min_val = np.min(combined_matrix)
        max_val = np.max(combined_matrix)
        # 计算 data_range
        data_range = max_val - min_val
        score= structural_similarity(matrix1, matrix2, data_range=data_range)
        diff_score = 1- score
        diff_score_sum = diff_score_sum  + diff_score
    return diff_score_sum/len(loaded_tensor)

def get_sample_info(id):
    """
    保存sample_info
    :return:
    """
    file_path = f"E:\\GitHub\\Chain-of-Embedding\\OutputInfo\\en\\Sample_info\\Qwen2.5-7B-Instruct\\commonsenseqa\\commonsenseqa_{id}.pkl"
    with open(file_path, 'rb') as file:
        sample_info = pickle.load(file)
    return  sample_info

def normalize_to_255(arr):
    arr_scaled = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return (arr_scaled * 255).astype(np.uint8)


def count_base(id,cos_dict):
    start_time = time.time()
    # random_seconds = id % 10
    # print(f"{id}即将休眠 {random_seconds} 秒")
    # 执行休眠操作
    # time.sleep(random_seconds)
    #print(f"{id}休眠结束")

    ########################################

    sample_info = get_sample_info(id)
    # output_scores = sample_info["output"]["output_scores"] # 输出层，264，每个tensor 为(1,152064)
    # ########### 传统方法，对 输出层进行分析操作 ##################
    # all_token_re = []  # 将每个layer转换为以为数组，并进行归一化处理
    # all_token_max_re = []  # 存储每一个layer 的最大值
    # for token in range(len(output_scores)):
    #     re = output_scores[token][0].tolist()  # 每一层转换为一维的数组
    #     re = F.softmax(torch.tensor(re), 0).cpu().tolist()  # softmax 函数的作用是将输入张量的每个元素转换为一个概率值，使得所有元素的概率之和为 1。第二个参数 0 表示在第 0 维上进行 softmax 操作。例如，对于一个形状为 (n,) 的一维张量，softmax 操作会将每个元素转换为一个概率值，使得这些概率值的和为 1
    #     all_token_re.append(re)
    #     all_token_max_re.append(max(re))
    # # (3) Maximum Softmax Probability
    # maxprob = np.mean(all_token_max_re) # ，最后求平均值
    # # (4) Perplexity
    # seq_ppl_list = [math.log(max_re) for max_re in all_token_max_re]  # 求每个元素的对数
    # ppl = -np.mean(seq_ppl_list)  # 再对所有的对数求平均值
    # # (5) entropy
    # from scipy.stats import entropy
    # seq_entropy_list = [entropy(re, base=2) for re in all_token_re] # 对self.all_token_re 中的每个元素，以2为底计算熵
    # entropy = np.mean(seq_entropy_list) # 求平均数
    # print(f"**********{id}: Output Info: **********\nmaxprob {maxprob}; perplexity {ppl}; entropy {entropy}\n")


    # 在使用 transformers 库中的预训练模型（如 BERT、GPT 等）进行推理时，模型会对输入的文本进行一系列的计算和转换，这些计算过程会在不同的层产生隐藏状态。
    # all_token_hidden_states 就是一个包含了模型所有层（包括嵌入层）的隐藏状态的元组。
    # 每一个元素对应着模型某一层的隐藏状态，其形状通常为 (batch_size, sequence_length, hidden_size)，其中：
    # batch_size 表示一次处理的样本数量。
    # sequence_length 表示输入序列的长度。
    # hidden_size 表示模型隐藏层的维度。
    hidden_states = sample_info["output"]["all_token_hidden_states"]
    output_len = sample_info["output"]["output_len"]
    layer_num = len(hidden_states[1])
    hs_all_layer = []
    hs_all_layer_pic = []
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
        hs_all_layer_pic.append(all_pos_hs)
        hs_all_layer.append(h)
     # compute_CoE_Mag
    # 系数a,计算二范数，即欧基里德范数，Zmag系数，系数的范围都是最有一层与第一层的差值
    norm_denominator_a = np.linalg.norm(hs_all_layer[-1] - hs_all_layer[0], ord=2)
    # 系数b 计算向量夹角（弧度）,#ZAng 系数，系数的范围都是最有一层与第一层的差值
    norm_denominator_te = np.dot(hs_all_layer[-1], hs_all_layer[0]) / (
            np.linalg.norm(hs_all_layer[-1], ord=2) * np.linalg.norm(hs_all_layer[0], ord=2))  # 计算向量夹角的余弦值
    norm_denominator_b = math.acos(norm_denominator_te)
    # 系数c
    norm_denominator_c = np.linalg.norm(hs_all_layer[-1] - hs_all_layer[0], ord=3)

    al_pic_repdiff_norm = []
    al_repdiff_norm = []
    al_semdiff = []
    al_semdiff_qz3_half_list = []
    al_semdiff_attention_list = []
    al_SSIM_diff1 = []
    al_SSIM_diff2 = []
    al_SSIM_diff3 = []
    manhattan_distance_list = []
    chebyshev_distance_list = []
    # norm_2_list = []
    norm_2_qz1_list = []
    # norm_2_qz2_list = []
    norm_2_qz3_list = []
    norm_2_qz3_half_list = []
    norm_2_list_attention_list = []
    norm_3_list = []
    score_grey_list = []
    score_grey_list_win3 = []
    score_grey_list_win7 = []
    score_pic_list = []
    mse_value_list = []
    glcm_distance_list = []
    a = hs_all_layer_pic[-1]
    b = hs_all_layer_pic[0]
    a = compress_2d_array(a, len(a), 1)
    b = compress_2d_array(b, len(a), 1)
    combined_matrix = np.concatenate((np.array(a), np.array(b)))
    min_val = np.min(combined_matrix)
    max_val = np.max(combined_matrix)
    # 计算 data_range
    data_range = max_val - min_val
    score_pic_n = structural_similarity(a, b, data_range=data_range, )
    a_grey = normalize_to_255(a)
    b_grey = normalize_to_255(b)
    diff = a - b
    score_grey_n = structural_similarity(a_grey,b_grey )
    score_grey_n_win3 = structural_similarity(a_grey, b_grey,win_size=3)
    score_grey_n_win7 = structural_similarity(a_grey, b_grey, win_size=7)
    mse_value_n = np.mean((a_grey - b_grey) ** 2)
    glcm_distance_n = compare_images_glcm(a, b)
    norm_denominator_a =   np.linalg.norm(diff,"fro")

    # # 记录变化最大的layer
    # cos_dict = {}
    # for i in range(len(hs_all_layer) - 1):
    #     a = hs_all_layer[i]
    #     b = hs_all_layer[len(hs_all_layer)-1]
    #     dot_product = np.dot(a, b)
    #     norm_a = np.linalg.norm(a)
    #     norm_b = np.linalg.norm(b)
    #     cos = dot_product / (norm_a * norm_b)
    #     # ，某一层的输入与输出之间的余弦相似度越高，就表明该层的重要性越低，反之亦然。
    #     cos_dict.setdefault(i, cos)
    #
    # print(cos_dict)

    for i  in range(len(hs_all_layer_pic)-1):
        a = hs_all_layer_pic[i + 1]
        b = hs_all_layer_pic[i]

        a = compress_2d_array(a, len(a), 1)
        b = compress_2d_array(b, len(a), 1)
        diff = a - b
        # # 计算差值的二范数
        norm_2 = np.linalg.norm(diff, "fro")
        al_pic_repdiff_norm.append(norm_2 / norm_denominator_a) # 除以系数，并添加到列表中
        glcm_distance=compare_images_glcm(a,b)
        glcm_distance_list.append(glcm_distance/glcm_distance_n)
        combined_matrix = np.concatenate((np.array(a), np.array(b)))
        min_val = np.min(combined_matrix)
        max_val = np.max(combined_matrix)
        # 计算 data_range
        data_range = max_val - min_val
        score_pic = structural_similarity(a, b, data_range=data_range, )
        score_pic_list.append(score_pic / score_pic_n)
        a_grey = normalize_to_255(a)
        b_grey = normalize_to_255(b)
        score_grey = structural_similarity(a_grey, b_grey)
        score_grey_win3 = structural_similarity(a_grey, b_grey, window=3)
        score_grey_win7 = structural_similarity(a_grey, b_grey, window=11)
        score_grey_list.append(score_grey / score_grey_n)
        score_grey_list_win3.append(score_grey_win3 / score_grey_n_win3)
        score_grey_list_win7.append(score_grey_win7 / score_grey_n_win7)
        mse_value = np.mean((a_grey - b_grey) ** 2)
        mse_value_list.append(mse_value/mse_value_n)

    for i in range(len(hs_all_layer) - 1):
        a = hs_all_layer[i]
        b = hs_all_layer[i +1 ]
        diff = b - a
        # # 计算差值的二范数
        norm_2 = np.linalg.norm(diff, ord=2)
        al_repdiff_norm.append(norm_2 / norm_denominator_a) # 除以系数，并添加到列表中
        # norm_2_list.append(norm_2)
        norm_2_qz1_list.append(i * norm_2/ norm_denominator_a)
        norm_2_qz3_list.append(i * i * norm_2 / norm_denominator_a)
        # norm_2_qz2_list.append((len(hs_all_layer) - i) * norm_2/ norm_denominator_a)
        norm_3 = np.linalg.norm(diff, ord=3)
        norm_3_list.append(norm_3)
        # # 计算夹角
        dot_product = np.dot(a, b)  # 计算两个数组的点积
        norm_a = np.linalg.norm(a, ord=2)   # 分别计算二范数
        norm_b = np.linalg.norm(b, ord=2) # 分别计算二范数
        similarity = dot_product / (norm_a * norm_b) # 点积除以二范数
        similarity = similarity if similarity <= 1 else 1
        arccos_sim = math.acos(similarity)  # 计算反余弦值
        al_semdiff.append(arccos_sim / norm_denominator_b)
        if i < (len(hs_all_layer) - 1) / 2:
            norm_2_qz3_half_list.append(i * i * norm_2 / norm_denominator_a)
            al_semdiff_qz3_half_list.append(i * i *  arccos_sim / norm_denominator_b )

        # ang = math.acos(cos_dict[i])
        # attention =  ang * ang * ang
        attention = cos_dict[i]
        # print(f"attention is {attention}")
        al_semdiff_attention_list.append(arccos_sim / norm_denominator_b * attention)
        norm_2_list_attention_list.append(norm_2 / norm_denominator_a * attention)

        # 除以系数，并添加到列表中

        # 像曼哈顿距离是计算各维度差值的绝对值之和，关注各维度差值绝对值的最大值，闵可夫斯基距离则是更通用的一种距离度量形式
        # 合并两个矩阵以找到整体的最小值和最大值
        combined_matrix = np.concatenate((np.array(a), np.array(b)))
        min_val = np.min(combined_matrix)
        max_val = np.max(combined_matrix)
        # 计算 data_range
        data_range = max_val - min_val
        score1 = structural_similarity(a, b, data_range=data_range, )
        score2 = structural_similarity(a, b, data_range=data_range, win_size=3)
        diff_score1 = score1
        al_SSIM_diff1.append(diff_score1)
        al_SSIM_diff2.append(score2 )
        al_SSIM_diff3.append((score2 + 1)** 2 )
        manhattan_distance = np.sum(np.abs(diff))
        manhattan_distance_list.append(manhattan_distance)
        chebyshev_distance = np.max(np.abs(diff))
        chebyshev_distance_list.append(chebyshev_distance)
    # 这个值也就是coe_mag
    al_repdiff_ave = np.mean(np.array(al_repdiff_norm))
    al_pic_repdiff_ave= np.mean(np.array(al_pic_repdiff_norm))
    # # 计算方差。方差用于衡量一组数据的离散程度，也就是数据相对于均值的分散情况。方差越大，说明数据越分散；方差越小，说明数据越集中在均值附近。
    # al_repdiff_var = np.var(np.array(al_repdiff_norm))
    #
    #al_semdiff_norm = np.array(al_semdiff)
    # # 求余弦角平均数 coe_ang = al_semdiff_ave
    al_semdiff_ave = np.mean(np.array(al_semdiff))
    al_semdiff_ave_qz3 = np.mean(np.array(al_semdiff_qz3_half_list))
    # # 计算方差，未使用
    # al_semdiff_var = np.var(np.array(al_semdiff_norm))

    # #coe_r ,夹角和二范数的结合
    print(f"al_repdiff_ave is {al_repdiff_ave},al_semdiff_ave is {al_semdiff_ave}")
    coe_r =  al_repdiff_ave - al_semdiff_ave

    # # coe_c
    # x_list = np.array([al_repdiff_norm[i] * math.cos(al_semdiff_norm[i]) for i in range(len(al_semdiff_norm))])
    # y_list = np.array([al_repdiff_norm[i] * math.sin(al_semdiff_norm[i]) for i in range(len(al_semdiff_norm))])
    # al_combdiff_x_ave = np.mean(x_list)
    # al_combdiff_y_ave = np.mean(y_list)
    # # al_combdiff_x_var = np.mean(x_list)
    # # al_combdiff_y_var = np.mean(y_list)
    # coe_c =  math.sqrt(al_combdiff_x_ave ** 2 + al_combdiff_y_ave ** 2)

    #############################################################
    score_pic_ave = np.mean(np.array(score_pic_list))
    score_pic_var = np.var(np.array(score_pic_list))

    score_gray_ave = np.mean(np.array(score_grey_list))
    score_gray_var = np.var(np.array(score_grey_list))

    score_gray_var_win3 = np.var(np.array(score_grey_list_win3))
    score_gray_ave_win3= np.mean(np.array(score_grey_list_win3))
    score_gray_ave_win7 = np.mean(np.array(score_grey_list_win7))
    score_gray_var_win7 = np.var(np.array(score_grey_list_win7))

    # norm_2_list_ave = np.mean(np.array(norm_2_list))
    norm_2_qz1_ave = np.mean(np.array(norm_2_qz1_list))
    #norm_2_qz2_ave = np.mean(np.array(norm_2_qz2_list))
    norm_2_qz3_ave = np.mean(np.array(norm_2_qz3_list))
    norm_2_qz3_half_ave = np.mean(np.array(norm_2_qz3_half_list))

    al_semdiff_attention = np.mean(np.array(al_semdiff_attention_list))
    norm_2_list_attention = np.mean(np.array(norm_2_list_attention_list))

    norm_3_list_ave = np.mean(np.array(norm_3_list))
    norm_3_list_ave_n = norm_3_list_ave / norm_denominator_c

    mse_value_ave = np.mean(np.array(mse_value_list))
    # 计算均值。均值也称为平均数，它是一组数据的总和除以数据的个数。均值能够反映出这组数据的中心位置或典型水平。

    combined_matrix = np.concatenate((np.array(hs_all_layer[-1]), np.array(hs_all_layer[0])))
    min_val = np.min(combined_matrix)
    max_val = np.max(combined_matrix)
    data_range = max_val - min_val

    al_SSIM_diff_ave1 = np.mean(np.array(al_SSIM_diff1))
    x = structural_similarity(hs_all_layer[-1] , hs_all_layer[0],data_range=data_range)
    al_SSIM_diff_ave1_x = al_SSIM_diff_ave1 / x

    al_SSIM_diff_ave2 = np.mean(np.array(al_SSIM_diff2))
    y = structural_similarity(hs_all_layer[-1], hs_all_layer[0],data_range=data_range )
    al_SSIM_diff_ave1_y = al_SSIM_diff_ave2 / y

    al_SSIM_diff_ave3 = np.mean(np.array(al_SSIM_diff3))
    al_SSIM_diff_ave1_z = al_SSIM_diff_ave3 / y
    manhattan_distance_ave = np.mean(np.array(manhattan_distance_list))
    chebyshev_distance_ave = np.mean(np.array(chebyshev_distance_list))
    glcm_distance_ave = np.mean(np.array(glcm_distance_list))

    end_time = time.time()
    elapsed_time = end_time - start_time
    #print(f"********** {id}:CoE Score Info: **********\nMag {al_repdiff_ave}; Ang {al_semdiff_ave}; R {coe_r}; C {coe_c}\n")
    print(f"************{id} finished cost {elapsed_time}****************************************")
    return {
       # "norm_2_list_ave":norm_2_list_ave,
        # * 序号
        "norm_2_qz1_ave":norm_2_qz1_ave,
        #"norm_2_qz2_ave":norm_2_qz2_ave,
        "norm_2_qz3_ave": norm_2_qz3_ave,
        "norm_2_qz3_half_ave":norm_2_qz3_half_ave,
        "norm_3_list_ave":norm_3_list_ave,
        "norm_3_list_ave_n": norm_3_list_ave_n,
        "Mag": al_repdiff_ave,
        "Mag_qz3":al_semdiff_ave_qz3,
        "Mag_pic": al_pic_repdiff_ave,
        "Ang": al_semdiff_ave,
        "R": coe_r,
        "al_semdiff_attention":al_semdiff_attention,
        "norm_2_list_attention":norm_2_list_attention,
        #"C": coe_c,
        #"maxprob": maxprob,
        #"ppl":ppl,
        #"entropy":entropy,
        "al_SSIM_diff_ave1":al_SSIM_diff_ave1,
        "al_SSIM_diff_ave1_x":al_SSIM_diff_ave1_x,
        "al_SSIM_diff_ave1_y": al_SSIM_diff_ave1_y,
        "al_SSIM_diff_ave1_z": al_SSIM_diff_ave1_z,

        "manhattan_distance_ave":manhattan_distance_ave,
        "chebyshev_distance_ave":chebyshev_distance_ave,
        "score_pic_ave": score_pic_ave,
        "score_pic_var": score_pic_var,

        "score_gray_ave":score_gray_ave,
        "score_gray_var": score_gray_var,

        "score_gray_ave_win3": score_gray_ave_win3,
        "score_gray_var_win3": score_gray_var_win3,
        "score_gray_ave_win7": score_gray_ave_win7,
        "score_gray_var_win7": score_gray_var_win7,
        "mse_value_ave" : mse_value_ave,

        "score_gray_list": score_grey_list,
        "mse_value_list": mse_value_list,

        "hs_all_layer_pic": hs_all_layer_pic,
        "hs_all_layer": hs_all_layer,
        "glcm_distance_ave":glcm_distance_ave

    }


def work(id,attention_dict):
    pid = os.getpid()
    str_out = f"PID {pid} ----Processing task {id}  "
    print(f"***{str_out}***")
    try:
        base_line_re = count_base(id,attention_dict)
    except:
        stack_trace = traceback.format_exc()
        print(stack_trace)
    base_dict = {"id": id, "val": base_line_re}
    return base_dict

def conunt_attention(id ):
    sample_info = get_sample_info(id)
    cos_dict = {}
    hidden_states = sample_info["output"]["all_token_hidden_states"]
    output_len = sample_info["output"]["output_len"]
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
            similarity = torch.trace(cosine) / cosine.size(0)
            cos_dict.setdefault(j,similarity.item())
            del cosine

        #cos = get_cos_similar_matrix(a,b)
        #dot_product = np.dot(a, b)
        #norm_a = np.linalg.norm(a)
        #norm_b = np.linalg.norm(b)
        #cos = dot_product / (norm_a * norm_b)
        # ，某一层的输入与输出之间的余弦相似度越高，就表明该层的重要性越低，反之亦然。
        # cos_dict.setdefault(i, cos)
    # 进行排序，越大说明越不重要，排在前面，
    sorted_items = sorted(cos_dict.items(), key=lambda item: item[1],reverse=True)
    sort_a = {}
    i = 0
    for item in sorted_items:
        i = i + 1
        k =  math.acos(item[1])
        sort_a.setdefault(item[0],i * k)
    print(sort_a)
    return sort_a

if __name__ == '__main__':
    start_time = time.time() 
    task_list = list(range(1000))
    p=Pool(16)
    res_l=[]
    attention_dict = conunt_attention(1)
    for task_id in task_list:
        res=p.apply_async(work,args=(task_id,attention_dict))
        res_l.append(res)
    p.close()
    p.join()
    # print([res.get() for res in res_l]) #该结果已经传给回调函数处理了
    ssim_list = [res.get() for res in res_l]
    base_dict_list = {}
    for base_dict in ssim_list:
        id = base_dict["id"]
        val = base_dict["val"]
        base_dict_list[id]=val
    # 打印最终结果
    glcm_list = []
    ssim_matrix_list = []
    # print("Final ssim_list:", ssim_list)
    # print("Final glcm_list:", glcm_list)
    # print("Final ssim_matrix_list:", ssim_matrix_list)
    # print("Final base_list:", base_dict_list)

    file_path = 'E:\\GitHub\\Chain-of-Embedding\\OutputImg\\Qwen2.5-7B-Instruct\\commonsenseqa\\base_list.pickle'
    # 以二进制写入模式打开文件
    with open(file_path, 'wb') as file:
        img_list = {"ssim_list":ssim_list,
                    "glcm_list":glcm_list,
                    "ssim_matrix_list":ssim_matrix_list,
                    "base_list":base_dict_list}
        # 使用 pickle.dump 方法将数组写入文件
        pickle.dump(img_list, file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(len(base_dict_list))
    print(f"数组已成功保存到 {file_path}")
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"程序运行时间: {minutes} 分 {seconds} 秒")


