import os
import torch
import re
from tqdm import tqdm

# 参数文件夹路径
params_folder_path = r"E:\download\megadepth_test_1500\C2FViT\NCC\train\params"

# 保存tensor的目标文件夹路径
tensor_folder_path = r"E:\download\megadepth_test_1500\C2FViT\NCC\train\tensor_scale"

# 如果目标文件夹不存在则创建
os.makedirs(tensor_folder_path, exist_ok=True)

# 读取文件的函数
def read_scale_from_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Scale:"):
                scale = float(line.split(":")[1].strip())
                return scale
    return None

# 生成仿射变换矩阵的函数
def get_affine_matrix(scale):
    # 确保scale是一个张量，如果不是，则转换为张量
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, dtype=torch.float32)  # 使用1/scale

    # 创建仿射变换矩阵 (2x3) 使用 scale, 角度和剪切设为 0
    matrix = torch.tensor([
        [scale, 0, 0],
        [0, scale, 0]
    ], dtype=torch.float32)

    return matrix

# 获取所有的参数文件并排序
files = [f for f in os.listdir(params_folder_path) if f.endswith("_params.txt")]
files.sort(key=lambda f: int(re.search(r'(\d+)_params\.txt', f).group(1)))

# 用于存储前10个缩放比例的列表
first_10_scales = []

# 逐个文件处理
for i, file in enumerate(tqdm(files, desc="Processing files"), start=1):
    file_path = os.path.join(params_folder_path, file)
    scale = read_scale_from_file(file_path)
    if scale is not None:
        # 如果是前10个缩放比例，添加到列表中
        if i <= 10:
            first_10_scales.append((file, scale))

        matrix = get_affine_matrix(scale)

        # 保存为单独的PT文件
        pt_file_path = os.path.join(tensor_folder_path, f"matrix_{i}.pt")
        torch.save(matrix, pt_file_path)

# 打印前10个缩放比例
print("First 10 scales used:")
for i, (file, scale) in enumerate(first_10_scales, start=1):
    print(f"Image {i} ({file}): {scale}")

print(f"All {len(files)} matrices have been saved as individual PT files.")