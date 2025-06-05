import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np

ImportantFilePath = "D:\\GitHub\\Chain-of-Embedding\\important_layer.txt"

def save_list_to_txt(data_list, file_path):
    """将列表保存为文本文件"""
    with open(file_path, 'w') as f:
        for item in data_list:
            f.write(f"{item}\n")  # 每行一个元素

def read_list_from_txt(file_path):
    """从文本文件读取列表"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def visualize_data(data,
                   title="Data Trend Analysis",
                   xlabel="Data Index",
                   ylabel="Value",
                   ylim=None,
                   figsize=(10, 6),
                   use_log_scale=True,
                   log_base=10,
                   show_minor_ticks=True):
    """
    Visualize data trends with points and connecting lines

    Args:
        data: 1D array or list of numerical values (as strings)
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        ylim: Y-axis limits (lower, upper)
        figsize: Figure size tuple (width, height)
        use_log_scale: 是否使用对数刻度
        log_base: 对数底数，默认10
        show_minor_ticks: 是否显示次要刻度
    """
    # Convert data to numpy array of floats
    data = np.asarray(data, dtype=float)
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    # Plot data with markers and lines
    ax.plot(data, 'b-', alpha=0.7, marker='o', markersize=5, linewidth=1.5, label='Data Trend')

    # # Add mean line
    # mean_val = data.mean()
    # ax.axhline(y=mean_val, color='black', linestyle='--', alpha=0.5,
    #            label=f'Mean: {mean_val:.4f}')
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
    # ax.grid(True, linestyle='--', alpha=0.7, which='both')
    # ax.legend()

    # Improve layout
    plt.tight_layout()

    return fig


# 使用示例
if __name__ == "__main__":
    # 你的原始数据（字符串格式）
    # data = ['0.004328814038837498', '0.5322024822235107', '0.5379875627431002', '0.5238742123950612',
    #         '0.5529060770164836', '0.5254425785758279', '0.52705533396114', '0.5113305341113697', '0.5215642208402808',
    #         '0.5164293375882235', '0.5188235423781655', '0.5167611349712719', '0.559021998535503', '0.5315533497116']
    data = read_list_from_txt(ImportantFilePath)
    # 绘制折线图
    fig = visualize_data(
        data,
        use_log_scale = True,
        #show_minor_ticks = True
    )

    # 显示图形
    plt.show()
