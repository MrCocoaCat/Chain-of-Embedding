import matplotlib.pyplot as plt
import numpy as np

# # 原始的三个数组
# commonsenseqa = [1.0406007766723633, 0.7250407934188843, 0.6725648641586304, 0.6548521518707275, 0.6962506771087646, 0.7020082473754883, 0.6963197588920593, 0.6781741380691528, 0.6245853900909424, 0.6106432676315308, 0.5724992752075195, 0.5766024589538574, 0.5886215567588806, 0.5379756093025208, 0.5522096157073975, 0.5406475067138672, 0.5007950067520142, 0.4478934705257416, 0.38496798276901245, 0.3444579839706421, 0.33581867814064026, 0.332529217004776, 0.2940826416015625, 0.26545315980911255, 0.2556241452693939, 0.24504514038562775, 0.26074421405792236, 0.26371437311172485, 0.29184702038764954, 0.33730071783065796, 0.4590054452419281, 1.0048129558563232]
# belebele = [1.0059887170791626, 0.7135728001594543, 0.6433003544807434, 0.6324571371078491, 0.6661279797554016, 0.6794449687004089, 0.6760115027427673, 0.6666869521141052, 0.6112107634544373, 0.5903513431549072, 0.5575310587882996, 0.5628000497817993, 0.5742903351783752, 0.5117640495300293, 0.5289332270622253, 0.510911762714386, 0.4825960099697113, 0.44053345918655396, 0.38256826996803284, 0.347384512424469, 0.33299198746681213, 0.3334199786186218, 0.30141738057136536, 0.2726823091506958, 0.2585926055908203, 0.24959079921245575, 0.2578006982803345, 0.26576656103134155, 0.2892928719520569, 0.32539063692092896, 0.4372800886631012, 0.9780370593070984]
# mgsm = [1.015941858291626, 0.7195775508880615, 0.6542302966117859, 0.6343052387237549, 0.6840294003486633, 0.6951364278793335, 0.6988703012466431, 0.6781207323074341, 0.625819206237793, 0.613976001739502, 0.5822576284408569, 0.5786303281784058, 0.5891360640525818, 0.5385839343070984, 0.5494654178619385, 0.5345502495765686, 0.4966372847557068, 0.4503241181373596, 0.383749395608902, 0.3518247604370117, 0.32904618978500366, 0.3292572498321533, 0.2961583137512207, 0.2659922242164612, 0.2555261254310608, 0.2471965253353119, 0.25462737679481506, 0.2598040699958801, 0.2855457663536072, 0.3174981474876404, 0.4525916576385498, 1.003174066543579]
#
#
def save_list_to_txt(data_list, file_path):
    """将列表保存为文本文件"""
    with open(file_path, 'w') as f:
        for item in data_list:
            f.write(f"{item}\n")  # 每行一个元素


#model_name = "Llama-3-8B-Instruct"
model_name =  "Qwen2.5-7B-Instruct"
dateset_names = ["commonsenseqa", "belebele", "mgsm"]

#
# def read_list_from_txt(file_path):
#     """从文本文件读取列表"""
#     with open(file_path, 'r') as f:
#         return [line.strip() for line in f.readlines()]

def read_list_from_txt(file_path):
    """从文本文件读取列表，并尝试将每行数据转换为浮点数类型返回"""
    result = []
    try:
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()  # 去除每行的空白字符（如换行符等）
                try:
                    num = float(line)  # 尝试将每行内容转换为浮点数
                    result.append(num)
                except ValueError:
                    continue  # 如果转换失败，跳过该行，不添加到结果列表
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在，请检查文件路径是否正确。")
    return result

#
commonsenseqa = read_list_from_txt("D:\\GitHub\\Chain-of-Embedding\\{model_name}_{dateset_name}_important_layer.txt".format(dateset_name = "commonsenseqa",model_name=model_name))

print(commonsenseqa)


belebele = read_list_from_txt("D:\\GitHub\\Chain-of-Embedding\\{model_name}_{dateset_name}_important_layer.txt"
                                   .format(dateset_name = "belebele",model_name=model_name))

print(belebele)
mgsm = read_list_from_txt("D:\\GitHub\\Chain-of-Embedding\\{model_name}_{dateset_name}_important_layer.txt"
                                   .format(dateset_name = "mgsm",model_name=model_name))
print(mgsm)


# 创建与数组长度匹配的x轴数据
print(len(commonsenseqa))
x = np.linspace(1, len(commonsenseqa), len(commonsenseqa))  # x轴数据长度与数组一致

# 创建图形和坐标轴
plt.figure(figsize=(12, 8))  # 增大图形尺寸以便更清晰地查看

# 使用英文标签等替换中文标签

# CommonsenseQA (Talmor et al., 2019) for the Reasoning domain,
plt.plot(x, commonsenseqa, 'r-', linewidth=2, marker='o',label='Reasoning')
# and Belebele (Bandarkar259
# # et al., 2023) for the Understanding domain.260
plt.plot(x, belebele, 'g--', linewidth=2, marker='o',label='Understanding')

#  GSM8K (Cobbe et al., 2021) was chosen for the Mathematics domain,
plt.plot(x, mgsm, 'b-.', linewidth=2, marker='o',label='Mathematics')

# 添加英文标题和标签
# plt.title(model_name, fontsize=16)
plt.xlabel('Layer Index', fontsize=14)
plt.ylabel('arccos', fontsize=14)


# 设置纵坐标为对数刻度（不等比例显示）
plt.yscale('log')  # 关键修改：使用对数坐标

# 添加网格线和图例
plt.grid(True, linestyle='--', alpha=0.7)  # 显示网格线，设置样式和透明度
plt.legend(fontsize=12)  # 显示图例，设置字体大小

# 设置坐标轴刻度和范围
plt.xticks(rotation=45)  # 旋转x轴刻度标签以便更好显示
plt.tight_layout()  # 自动调整布局

# 显示图形
plt.show()


average = np.mean([commonsenseqa, belebele, mgsm], axis=0)
print(average)
ImportantFilePath = f"D:\\GitHub\\Chain-of-Embedding\\{model_name}_important_layer.txt"
save_list_to_txt(average, ImportantFilePath)