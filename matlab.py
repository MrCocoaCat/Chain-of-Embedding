import matplotlib.pyplot as plt
from torchvision import transforms
import torch

file_path ="C:\\Users\liyubo\Documents\GitHub\Chain-of-Embedding\OutputInfo\en\HiddenLayer\Qwen2.5-7B-Instruct\commonsenseqa\commonsenseqa_0.pt"
loaded_tensor = torch.load(file_path)
plt.imshow(transforms.ToPILImage()(loaded_tensor), interpolation="bicubic")
transforms.ToPILImage()(loaded_tensor).show()
#plt.show()