import matplotlib.pyplot as plt
from torchvision import transforms
import torch

file_path ="C:\\Users\liyubo\Documents\GitHub\Chain-of-Embedding\OutputInfo\en\HiddenLayer\Qwen2.5-7B-Instruct\commonsenseqa\commonsenseqa_0.pt"
#loaded_tensor = torch.load(file_path)
# plt.imshow(transforms.ToPILImage()(loaded_tensor), interpolation="bicubic")
# transforms.ToPILImage()(loaded_tensor).show()
#plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 示例一维浮点数向量
vector = np.random.rand(100)

# 将一维向量转换为二维数组（高度为 1）
image = vector.reshape(1, -1)

# 显示图像
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()