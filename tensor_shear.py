import os
import torch
import re
from tqdm import tqdm

# 参数文件夹路径
params_folder_path = r"E:\download\megadepth_test_1500\C2FViT\NCC\train\params"

# 保存tensor的目标文件夹路径
tensor_folder_path = r"E:\download\megadepth_test_1500\C2FViT\NCC\train\tensor_shear"

# 如果目标文件夹不存在则创建
os.makedirs(tensor_folder_path, exist_ok=True)

# 读取文件的函数
def read_shear_from_file(file_path):
    shear_x = 0.0
    shear_y = 0.0
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Shear:"):
                shear_values = line.split(":")[1].strip().strip("()").split(",")
                shear_x = float(shear_values[0].strip())
                shear_y = float(shear_values[1].strip())
    return shear_x, shear_y

# Convert degrees to radians
def degrees_to_radians(degrees):
    return degrees * (torch.pi / 180.0)

# 生成仿射变换矩阵的函数，使用逆剪切
def get_inverse_affine_matrix_with_shear(shear_x, shear_y):
    # 固定scale为1
    scale = 1.0

    # 将剪切角度从度转换为弧度，并将其转换为张量
    shear_x_rad = torch.tensor(degrees_to_radians(shear_x), dtype=torch.float32)
    shear_y_rad = torch.tensor(degrees_to_radians(shear_y), dtype=torch.float32)

    # 计算逆剪切的仿射变换矩阵 (2x3)
    # 使用负的剪切值并转换为弧度的正切值
    matrix = torch.tensor([
        [scale, -torch.tan(shear_x_rad), 0],
        [-torch.tan(shear_y_rad), scale, 0]
    ], dtype=torch.float32)

    return matrix

# 获取所有的参数文件并排序
files = [f for f in os.listdir(params_folder_path) if f.endswith("_params.txt")]
files.sort(key=lambda f: int(re.search(r'(\d+)_params\.txt', f).group(1)))

# 用于存储前10个剪切参数的列表
first_10_shears = []

# 逐个文件处理
for i, file in enumerate(tqdm(files, desc="Processing files"), start=1):
    file_path = os.path.join(params_folder_path, file)
    shear_x, shear_y = read_shear_from_file(file_path)

    # 如果是前10个剪切参数，添加到列表中
    if i <= 10:
        first_10_shears.append((file, shear_x, shear_y))

    # 获取逆剪切的仿射变换矩阵
    matrix = get_inverse_affine_matrix_with_shear(shear_x, shear_y)

    # 保存为单独的PT文件
    pt_file_path = os.path.join(tensor_folder_path, f"matrix_{i}.pt")
    torch.save(matrix, pt_file_path)

# 打印前10个剪切参数
print("First 10 shears used (for inverse transformation):")
for i, (file, shear_x, shear_y) in enumerate(first_10_shears, start=1):
    print(f"Image {i} ({file}): Shear_x = {shear_x}, Shear_y = {shear_y}")

print(f"All {len(files)} inverse matrices have been saved as individual PT files.")