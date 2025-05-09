import torch.nn.functional as F
from skimage.metrics import structural_similarity
import cv2
from skimage.feature import graycomatrix, graycoprops
import os
import sys
import time
import numpy as np
import pickle
import math
import torch
from multiprocessing import Pool
from multiprocessing import Queue
import time
import random
import time
import traceback



device = 'cpu'
project_root_path = "E:/GitHub/Chain-of-Embedding/"

import numpy as np

import numpy as np


def compress_2d_array(arr, n, dim):
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

def calculate_glcm_features(image_path):
    # 读取图片并转换为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 计算灰度共生矩阵
    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(image, distances, angles, levels=256, symmetric=True, normed=True)

    # 提取灰度共生矩阵的特征值
    contrast = graycoprops(glcm, 'contrast')
    correlation = graycoprops(glcm, 'correlation')
    energy = graycoprops(glcm, 'energy')
    homogeneity = graycoprops(glcm, 'homogeneity')

    # 将特征值展平为一维数组
    features = np.hstack([contrast.flatten(), correlation.flatten(), energy.flatten(), homogeneity.flatten()])
    return features

def compare_images_glcm(image_path1, image_path2):
    # 计算两张图片的灰度共生矩阵特征值
    features1 = calculate_glcm_features(image_path1)
    features2 = calculate_glcm_features(image_path2)

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

def count_base(id):
    start_time = time.time()
    # random_seconds = id % 10
    # print(f"{id}即将休眠 {random_seconds} 秒")
    # 执行休眠操作
    # time.sleep(random_seconds)
    #print(f"{id}休眠结束")
    sample_info = get_sample_info(id)
    output_scores = sample_info["output"]["output_scores"] # 输出层，264，每个tensor 为(1,152064)
    ########### 传统方法，对 输出层进行分析操作 ##################
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

    al_repdiff_norm = []
    al_semdiff = []
    al_SSIM_diff1 = []
    al_SSIM_diff2 = []
    manhattan_distance_list = []
    chebyshev_distance_list = []
    norm_2_list = []
    norm_3_list = []
    score_pic_list = []
    a = hs_all_layer_pic[-1]
    b = hs_all_layer_pic[0]
    combined_matrix = np.concatenate((np.array(a), np.array(b)))
    min_val = np.min(combined_matrix)
    max_val = np.max(combined_matrix)
    # 计算 data_range
    data_range = max_val - min_val
    score_pic_n = structural_similarity(a, b, data_range=data_range, )
    for i  in range(len(hs_all_layer_pic)-1):
        a = hs_all_layer_pic[i + 1]
        b = hs_all_layer_pic[i]
        a = compress_2d_array(a, 100, 1)
        b = compress_2d_array(b, 100, 1)
        combined_matrix = np.concatenate((np.array(a), np.array(b)))
        min_val = np.min(combined_matrix)
        max_val = np.max(combined_matrix)
        # 计算 data_range
        data_range = max_val - min_val
        score_pic = structural_similarity(a, b, data_range=data_range, ) + 1
        score_pic_list.append(score_pic / score_pic_n)
    for i in range(len(hs_all_layer) - 1):
        a = hs_all_layer[i + 1]
        b = hs_all_layer[i]
        diff = a - b
        # 计算差值的二范数
        norm_2 = np.linalg.norm(diff, ord=2)
        al_repdiff_norm.append(norm_2 / norm_denominator_a) # 除以系数，并添加到列表中
        norm_2_list.append(norm_2)
        norm_3 = np.linalg.norm(diff, ord=3)
        norm_3_list.append(norm_3)
        # 计算夹角
        dot_product = np.dot(a, b)  # 计算两个数组的点积
        norm_a = np.linalg.norm(a, ord=2)   # 分别计算二范数
        norm_b = np.linalg.norm(b, ord=2) # 分别计算二范数
        similarity = dot_product / (norm_a * norm_b) # 点积除以二范数
        similarity = similarity if similarity <= 1 else 1
        arccos_sim = math.acos(similarity)  # 计算反余弦值
        al_semdiff.append(arccos_sim / norm_denominator_b)  # 除以系数，并添加到列表中
        # 像曼哈顿距离是计算各维度差值的绝对值之和，关注各维度差值绝对值的最大值，闵可夫斯基距离则是更通用的一种距离度量形式
        # 合并两个矩阵以找到整体的最小值和最大值
        combined_matrix = np.concatenate((np.array(a), np.array(b)))
        min_val = np.min(combined_matrix)
        max_val = np.max(combined_matrix)
        # 计算 data_range
        data_range = max_val - min_val
        score1 = structural_similarity(a, b, data_range=data_range, )
        score2 = structural_similarity(a, b, data_range=data_range, win_size=11)
        diff_score1 = 1 + score1
        al_SSIM_diff1.append(diff_score1)
        al_SSIM_diff2.append(score2 + 1)
        manhattan_distance = np.sum(np.abs(diff))
        manhattan_distance_list.append(manhattan_distance)
        chebyshev_distance = np.max(np.abs(diff))
        chebyshev_distance_list.append(chebyshev_distance)

    score_pic_ave = np.mean(np.array(score_pic_list))
    score_pic_var = np.var(np.array(score_pic_list))
    norm_2_list_ave = np.mean(np.array(norm_2_list))
    norm_3_list_ave = np.mean(np.array(norm_3_list))
    norm_3_list_ave_n = norm_3_list_ave / norm_denominator_c
    # 计算均值。均值也称为平均数，它是一组数据的总和除以数据的个数。均值能够反映出这组数据的中心位置或典型水平。
    # 这个值也就是coe_mag
    al_repdiff_ave = np.mean(np.array(al_repdiff_norm))
    # 计算方差。方差用于衡量一组数据的离散程度，也就是数据相对于均值的分散情况。方差越大，说明数据越分散；方差越小，说明数据越集中在均值附近。
    al_repdiff_var = np.var(np.array(al_repdiff_norm))

    al_semdiff_norm = np.array(al_semdiff)
    # 求余弦角平均数 coe_ang = al_semdiff_ave
    al_semdiff_ave = np.mean(np.array(al_semdiff_norm))
    # 计算方差，未使用
    al_semdiff_var = np.var(np.array(al_semdiff_norm))

    #############################################################

    al_SSIM_diff_ave1 = np.mean(np.array(al_SSIM_diff1))
    combined_matrix = np.concatenate((np.array(hs_all_layer[-1]), np.array(hs_all_layer[0])))
    min_val = np.min(combined_matrix)
    max_val = np.max(combined_matrix)
    # 计算 data_range
    data_range = max_val - min_val
    x = structural_similarity(hs_all_layer[-1] , hs_all_layer[0],data_range=data_range,)
    al_SSIM_diff_ave1_x = al_SSIM_diff_ave1 / x

    al_SSIM_diff_ave2 = np.mean(np.array(al_SSIM_diff2))
    y = structural_similarity(hs_all_layer[-1], hs_all_layer[0], data_range=data_range,win_size= 11 )
    al_SSIM_diff_ave1_y = al_SSIM_diff_ave2 / y



    manhattan_distance_ave = np.mean(np.array(manhattan_distance_list))
    chebyshev_distance_ave = np.mean(np.array(chebyshev_distance_list))
    #coe_r ,夹角和二范数的结合
    coe_r =  al_repdiff_ave - al_semdiff_ave
    # coe_c
    x_list = np.array([al_repdiff_norm[i] * math.cos(al_semdiff_norm[i]) for i in range(len(al_semdiff_norm))])
    y_list = np.array([al_repdiff_norm[i] * math.sin(al_semdiff_norm[i]) for i in range(len(al_semdiff_norm))])
    al_combdiff_x_ave = np.mean(x_list)
    al_combdiff_y_ave = np.mean(y_list)
    # al_combdiff_x_var = np.mean(x_list)
    # al_combdiff_y_var = np.mean(y_list)
    coe_c =  math.sqrt(al_combdiff_x_ave ** 2 + al_combdiff_y_ave ** 2)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"********** {id}:CoE Score Info: **********\nMag {al_repdiff_ave}; Ang {al_semdiff_ave}; R {coe_r}; C {coe_c}\n")
    print(f"************{id} finished cost {elapsed_time}****************************************")
    return {
        "norm_2_list_ave":norm_2_list_ave,
        "norm_3_list_ave":norm_3_list_ave,
        "norm_3_list_ave_n": norm_3_list_ave_n,
        "Mag": al_repdiff_ave,
        "Ang": al_semdiff_ave,
        "R": coe_r,
        "C": coe_c,
        "maxprob": maxprob,
        "ppl":ppl,
        "entropy":entropy,
        "al_SSIM_diff_ave1":al_SSIM_diff_ave1,
        "al_SSIM_diff_ave1_x":al_SSIM_diff_ave1_x,
        "manhattan_distance_ave":manhattan_distance_ave,
        "chebyshev_distance_ave":chebyshev_distance_ave,
        "score_pic_ave": score_pic_ave,
        "score_pic_var": score_pic_var,
        "al_SSIM_diff_ave1_y":al_SSIM_diff_ave1_y
    }


def worker_fun(id, ssim_queue, glcm_queue,ssim_matrix_queue,bash_queue):
    #id = task_queue.get(timeout=1)
    # average_ssim, average_glcm_distence = count_img(id)
    #ssim_matrix = count_Matrix(id)
    #ssim_queue.put(average_ssim)
    #glcm_queue.put(average_glcm_distence)
    #ssim_matrix_queue.put(ssim_matrix)
    pid = os.getpid()
    str_out  = f"Processing task {id} with PID {pid}"
    print(f"***{str_out}***")
    base_line_re = count_base(id)
    base_dict = {"id":id,"val":base_line_re}
    bash_queue.put(base_dict)
    #print(f"{id} average_ssim is :{average_ssim} ")
    #print(f"{id} 平均灰度共生矩阵的欧式距离 :{average_glcm_distence} ")
    #task_queue.task_done()
    return str_out

def get_page(id):
    pid = os.getpid()
    str_out = f"PID {pid} ----Processing task {id}  "
    print(f"***{str_out}***")
    time.sleep(1)
    try:
        base_line_re = count_base(id)
    except:
        stack_trace = traceback.format_exc()
        print("异常堆栈信息：")
        print(stack_trace)
    base_dict = {"id": id, "val": base_line_re}
    return base_dict


if __name__ == '__main__':
    start_time = time.time()
    task_list = list(range(200))
    p=Pool(32)
    res_l=[]
    for task_id in task_list:
        res=p.apply_async(get_page,args=(task_id,))
        res_l.append(res)
    p.close()
    p.join()
    print([res.get() for res in res_l]) #该结果已经传给回调函数处理了
    ssim_list = [res.get() for res in res_l]
    base_dict_list = {}
    for base_dict in ssim_list:
        id = base_dict["id"]
        val = base_dict["val"]
        base_dict_list[id]=val

    # 打印最终结果
    glcm_list = []
    ssim_matrix_list = []
    print("Final ssim_list:", ssim_list)
    print("Final glcm_list:", glcm_list)
    print("Final ssim_matrix_list:", ssim_matrix_list)
    print("Final base_list:", base_dict_list)

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


