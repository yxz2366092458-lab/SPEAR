import numpy as np
import matplotlib.pyplot as plt

# 1. 读取npy文件
# 替换 'your_file.npy' 为您的实际文件名
import numpy as np
import os
from pathlib import Path

# 指定根路径
base_path1 = r".\starkleberg"
base_path2 = r".\ernie_results"
i = 0
print(len(os.listdir(base_path1)))

# 遍历所有子文件夹
for folder1 in os.listdir(base_path1):
    for folder2 in os.listdir(base_path2):
        if folder1 != folder2:
            continue
        folder_path1 = os.path.join(base_path1, folder1)
        folder_path2 = os.path.join(base_path2, folder2)

        # 确保是文件夹
        if os.path.isdir(folder_path1):
            # 构建文件路径
            file_path1 = os.path.join(folder_path1, "eval_reward_True_0.1_35_4.npy")
            file_path2 = os.path.join(folder_path2, "eval_reward_True_0.1_35_4.npy")

            # 检查文件是否存在
            if os.path.exists(file_path1) and os.path.exists(file_path2):
                try:
                    # 加载npy文件
                    data = np.load(file_path1)
                    print(f"成功读取: {file_path1}")
                    print(f"成功读取: {file_path2}")
                    print(f"数据形状: {data.shape}")
                    print(f"数据内容示例:\n{data[:5] if len(data.shape) == 1 else data[:2]}\n")
                    if data.shape[0] <= 1:
                        i += 1
                    data1 = np.load(file_path1, allow_pickle=True)
                    data2 = np.load(file_path2, allow_pickle=True)

                    plt.figure(figsize=(10, 6))

                    plt.plot(data1, linewidth=2, color='red')
                    plt.plot(data2, linewidth=2, color='blue')
                    plt.xlabel('step')
                    plt.ylabel('reward')
                    plt.title('rewards')

                    plt.grid(True, alpha=0.3)

                    plt.show()
                except Exception as e:
                    print(f"读取 {file_path1} 时出错: {e}")

print(i)
