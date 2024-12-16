import os
import torch
import re
from tqdm import tqdm

# 参数文件夹路径
params_folder_path = r"E:\download\megadepth_test_1500\C2FViT\NCC\train\params"

# 保存tensor的目标文件夹路径
tensor_folder_path = r"E:\download\megadepth_test_1500\C2FViT\NCC\train\tensor_rotation"

# 如果目标文件夹不存在则创建
os.makedirs(tensor_folder_path, exist_ok=True)

# 读取文件的函数
def read_angle_from_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Angle:"):
                angle = float(line.split(":")[1].strip())
                return angle
    return None

# 生成仿射变换矩阵的函数
def get_affine_matrix(angle):
    # 确保angle是一个张量，如果不是，则转换为张量
    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(angle)

    # 将角度转换为弧度
    angle_rad = angle * (torch.pi / 180)

    # 计算cos和sin
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    # 创建仿射变换矩阵 (2x3)
    matrix = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0]
    ])

    return matrix

# 获取所有的参数文件并排序
files = [f for f in os.listdir(params_folder_path) if f.endswith("_params.txt")]
files.sort(key=lambda f: int(re.search(r'(\d+)_params\.txt', f).group(1)))

# 用于存储前10个角度的列表
first_10_angles = []

# 逐个文件处理
for i, file in enumerate(tqdm(files, desc="Processing files"), start=1):
    file_path = os.path.join(params_folder_path, file)
    angle = read_angle_from_file(file_path)
    if angle is not None:
        # 如果是前10个角度，添加到列表中
        if i <= 10:
            first_10_angles.append((file, angle))

        matrix = get_affine_matrix(angle)

        # 保存为单独的PT文件
        pt_file_path = os.path.join(tensor_folder_path, f"matrix_{i}.pt")
        torch.save(matrix, pt_file_path)

# 打印前10个角度
print("First 10 angles used:")
for i, (file, angle) in enumerate(first_10_angles, start=1):
    print(f"Image {i} ({file}): {angle} degrees")

print(f"All {len(files)} matrices have been saved as individual PT files.")