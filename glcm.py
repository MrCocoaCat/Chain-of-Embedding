import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import torch
import numpy as np
import math
import torch.nn.functional as F
from skimage.metrics import structural_similarity
import cv2
import os
import pickle

def calculate_glcm_features(image_path):
    # 读取图片并转换为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 计算灰度共生矩阵
    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = greycomatrix(image, distances, angles, levels=256, symmetric=True, normed=True)

    # 提取灰度共生矩阵的特征值
    contrast = greycoprops(glcm, 'contrast')
    correlation = greycoprops(glcm, 'correlation')
    energy = greycoprops(glcm, 'energy')
    homogeneity = greycoprops(glcm, 'homogeneity')

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

def count_glcm(id):
    img_path_dir = f"C:\\Users\\liyubo\\Documents\\GitHub\\Chain-of-Embedding\\OutputImg\\Qwen2.5-7B-Instruct\\commonsenseqa\\commonsenseqa_{id}\\"
    # 初始化文件计数器
    file_count = 0
    # 遍历文件夹中的所有文件和子文件夹
    for item in os.listdir(img_path_dir):
        item_path = os.path.join(img_path_dir, item)
        if os.path.isfile(item_path):
            file_count += 1
    print(f"文件夹 {img_path_dir} 中的文件数量为: {file_count}")
    sum_glcm =  0
    range_num = file_count
    for i in range(range_num-1):
        img_path1 = img_path_dir +f"{i}.jpg"
        img_path2 = img_path_dir+ f"{i+1}.jpg"
        # score 越大。则越相似
        # 示例使用
        distance = compare_images_glcm(img_path1, img_path2)
        print(f"两张图片的灰度共生矩阵特征值之间的欧氏距离为: {distance}")
        sum_glcm = sum_glcm + distance
    average_glcm = sum_glcm / range_num
    return average_glcm

if __name__ == '__main__':
    sum_si = 0
    gmcl_list = []
    for id in range(500):
        #tensorToImg(id)
        average_ssim = count_glcm(id)
        ssim_list.append(average_ssim)
        print(f"{id} average_ssim is :{average_ssim} ")
    print(ssim_list)
    file_path = 'C:\\Users\\liyubo\\Documents\\GitHub\\Chain-of-Embedding\\OutputImg\\Qwen2.5-7B-Instruct\\commonsenseqa\\glcm_list.pickle'
    # 以二进制写入模式打开文件
    with open(file_path, 'wb') as file:
        # 使用 pickle.dump 方法将数组写入文件
        pickle.dump(ssim_list, file)
    print(f"数组已成功保存到 {file_path}")


