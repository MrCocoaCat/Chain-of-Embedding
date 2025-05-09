from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.cluster import KMeans
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.nn.functional as F
import cv2
from skimage.metrics import structural_similarity
import cv2
from transformers.image_utils import pil_torch_interpolation_mapping


class ImageTextureAnalyzer:
    def extract_texture_features(self, tex):
        radius = 3
        n_point = radius * 8
        lbp = local_binary_pattern(tex, n_point, radius, method="uniform")
        max_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
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


def tensorToImgList():
    """
    J将tensor 转换为灰度图
    :return:
    """
    file_path = "C:\\Users\\liyubo\\Documents\\GitHub\\Chain-of-Embedding\\OutputInfo\\en\\HiddenLayer\\Qwen2.5-7B-Instruct\\commonsenseqa\\commonsenseqa_0.pt"
    print(f" 读取 {file_path} Tensor")
    loaded_tensor = torch.load(file_path)
    ImgList = []
    for i in range(len(loaded_tensor)):
        # 去掉多余的维度
        tensor = loaded_tensor[i].flatten().cpu()
        print(f"{i}去掉多余维度后Tensor的维度:{tensor.shape}")
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
        ImgList.append(image_opencv)
    return ImgList

if __name__ == '__main__':

    # 示例使用
    analyzer = ImageTextureAnalyzer()
    # 生成一些示例纹理图像
    all_textures_list = tensorToImgList()
    # 构建特征向量
    features_list = analyzer.build_feature_vectors(all_textures_list)
    # 执行聚类
    cluster_centers = analyzer.perform_clustering(features_list, num_clusters=3)
    # 将图像转换为视觉词带向量
    bof_vectors_list = [analyzer.image_to_bof(tex, cluster_centers) for tex in all_textures_list]
    # 构建相似度矩阵
    similarity_matrix = analyzer.build_similarity_matrix(bof_vectors_list)
    print(similarity_matrix)