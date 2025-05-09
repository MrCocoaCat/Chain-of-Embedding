
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.nn.functional as F
import cv2
from skimage.metrics import structural_similarity
import cv2
from transformers.image_utils import pil_torch_interpolation_mapping


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

    # 计算相似性得分
    similarity_score = len(good_matches) / max(len(kp1), len(kp2))

    # # 计算直方图并比较
    # hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    # hist2 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    # similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    # print("Histogram Similarity:", similarity)


    # 在两张图像上检测关键点和计算特征描述子
    #keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    #keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    #print(f"图像1检测到 {len(keypoints1)} 个关键点")
    #print(f"图像2检测到 {len(keypoints2)} 个关键点")


    # 创建FLANN匹配器
    #flann = cv2.FlannBasedMatcher()

    # 使用knnMatch进行特征匹配
    #matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 进行筛选，保留较好的匹配结果
    #good_matches = []
    #for m, n in matches:
    #    if m.distance < 0.7 * n.distance:
    #        good_matches.append(m)

    # 计算相似度
    #similarity = len(good_matches) / max(len(descriptors1), len(descriptors2))

    #return similarity


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
        print("SSIM:{}".format(score))
        return score, diff
    except Exception as e:
        print(f"发生错误: {e}")
        return None, None

def tensorToImg():
    """
    J将tensor 转换为灰度图
    :return:
    """
    file_path = "C:\\Users\\liyubo\\Documents\\GitHub\\Chain-of-Embedding\\OutputInfo\\en\\HiddenLayer\\Qwen2.5-7B-Instruct\\commonsenseqa\\commonsenseqa_0.pt"
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
        cv2.imshow('Grayscale Image', image_opencv)

        # 转换为NumPy数组，并且转换为uint8类型
        #image_np = (tensor_3d * 255).cpu().numpy().astype(np.uint8)
        # 显示图像
        #plt.imshow(image_np, cmap='gray')
        #plt.axis('off')
        img_path = f"C:\\Users\\liyubo\\Documents\\GitHub\\Chain-of-Embedding\\OutputImg\\{i}.jpg"
        #save_path = 'grayscale_image.png'
        cv2.imwrite(img_path, image_opencv)
        print(f"灰度图已保存到 {img_path}")


import cv2
import numpy as np


def average_hash(image_path):
    """
    计算图片的均值哈希值
    :param image_path: 图片的路径
    :return: 图片的均值哈希值
    """
    try:
        # 读取图片并转换为灰度图
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("无法读取图片，请检查文件路径是否正确。")
            return None
        # 缩小图片尺寸
        resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
        # 计算均值
        mean = np.mean(resized)
        # 生成哈希值
        hash_value = (resized > mean).flatten().astype(int)
        return ''.join(map(str, hash_value))
    except Exception as e:
        print(f"发生错误: {e}")
        return None


def hamming_distance(hash1, hash2):
    """
    计算两个哈希值的汉明距离
    :param hash1: 第一个哈希值
    :param hash2: 第二个哈希值
    :return: 汉明距离
    """
    if hash1 is None or hash2 is None or len(hash1) != len(hash2):
        return None
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def verage_hash_image_difference(image_path1, image_path2):
    """
    比较两张图片的差异性
    :param image_path1: 第一张图片的路径
    :param image_path2: 第二张图片的路径
    :return: 汉明距离
    """
    hash1 = average_hash(image_path1)
    hash2 = average_hash(image_path2)
    distance = hamming_distance(hash1, hash2)
    if distance is not None:
        print(f"两张图片的汉明距离为: {distance}")
    return distance


from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.cluster import KMeans

class ImageTextureAnalyzer:
    def extract_texture_features(self, tex):
        radius = 3
        n_point = radius * 8
        lbp = local_binary_pattern(tex, n_point, radius, method="uniform")
        # 统计图像的直方图
        max_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
        # 返回特征向量
        return hist

    def build_feature_vectors(self, all_textures_list):
        features_list = []
        for eachTex in all_textures_list:
            feature = self.extract_texture_features(eachTex)
            features_list.append(feature)
        return np.array(features_list)

    def perform_clustering(self, features_list, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(features_list)
        return kmeans.cluster_centers_

    def bhattacharyya_distance(self, p, q):
        return np.sqrt(np.sum(np.sqrt(p * q)))

    def image_to_bof(self, tex, cluster_centers):
        feature = self.extract_texture_features(tex)
        distances = np.linalg.norm(cluster_centers - feature, axis=1)
        nearest_cluster = np.argmin(distances)
        bof_vector = np.zeros(len(cluster_centers))
        bof_vector[nearest_cluster] = 1
        return bof_vector

    def build_similarity_matrix(self, bof_vectors_list):
        num_images = len(bof_vectors_list)
        similarity_matrix = np.zeros((num_images, num_images))
        for i in range(num_images):
            for j in range(i + 1, num_images):
                p = bof_vectors_list[i]
                q = bof_vectors_list[j]
                similarity = 1 - self.bhattacharyya_distance(p, q)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        return similarity_matrix

if __name__ == '__main__':
    sum_si = 0
    # 示例二维 float32 数组
    #example_array = np.random.rand(10, 10).astype(np.float32)
    # 调用函数进行着色
    #color_float32_array(example_array)
    #tensorToImg()
    for i in range(100):
        img1_path = file_path = f"C:\\Users\\liyubo\\Documents\\GitHub\\Chain-of-Embedding\\OutputImg\\{i}.jpg"
        img2_path = file_path = f"C:\\Users\\liyubo\\Documents\\GitHub\\Chain-of-Embedding\\OutputImg\\{i+1}.jpg"
        # re = compare_images(img1_path,img2_path)
        # print(re)
        re = contrast_image_SSIM(img1_path, img2_path)
        print(re)
        #sum_si = sum_si + re
        # average_hash_image_difference
        re = verage_hash_image_difference(img1_path, img2_path)
        print(re)
    print(sum_si)

