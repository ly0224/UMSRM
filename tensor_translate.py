import os
import torch
import re
from tqdm import tqdm

# 参数文件夹路径
params_folder_path = r"E:\download\megadepth_test_1500\C2FViT\affine\train_full\params"

# 保存tensor的目标文件夹路径
tensor_folder_path = r"E:\download\megadepth_test_1500\C2FViT\affine\train_full\tensor_translation"

# 如果目标文件夹不存在则创建
os.makedirs(tensor_folder_path, exist_ok=True)

# 读取文件的函数
def read_translation_from_file(file_path):
    translate_x = 0.0
    translate_y = 0.0
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Translation:"):
                translation_values = line.split(":")[1].strip().strip("()").split(",")
                translate_x = float(translation_values[0].strip())
                translate_y = float(translation_values[1].strip())
    return translate_x, translate_y

# 生成仿射变换矩阵的函数，使用逆平移并进行标准化
def get_normalized_inverse_affine_matrix(translate_x, translate_y, image_width, image_height):
    # 固定scale为1
    scale = 1.0

    # 将像素平移转换为归一化平移
    tx_norm = translate_x / (image_width/2)
    ty_norm = translate_y / (image_height/2)

    # 计算逆平移的标准化仿射变换矩阵 (2x3)
    # 使用负的归一化平移值
    matrix = torch.tensor([
        [scale, 0, tx_norm],
        [0, scale, ty_norm]
    ], dtype=torch.float32)

    return matrix

# 假设的图像宽度和高度
# 根据实际图像尺寸进行调整
image_width = 256   # Example image width in pixels
image_height = 256  # Example image height in pixels

# 获取所有的参数文件并排序
files = [f for f in os.listdir(params_folder_path) if f.endswith("_params.txt")]
files.sort(key=lambda f: int(re.search(r'(\d+)_params\.txt', f).group(1)))

# 用于存储前10个平移参数的列表
first_10_translations = []

# 逐个文件处理
for i, file in enumerate(tqdm(files, desc="Processing files"), start=1):
    file_path = os.path.join(params_folder_path, file)
    translate_x, translate_y = read_translation_from_file(file_path)

    # 如果是前10个平移参数，添加到列表中
    if i <= 10:
        first_10_translations.append((file, translate_x, translate_y))

    # 获取逆平移的标准化仿射变换矩阵
    matrix = get_normalized_inverse_affine_matrix(translate_x, translate_y, image_width, image_height)

    # 保存为单独的PT文件
    pt_file_path = os.path.join(tensor_folder_path, f"matrix_{i}.pt")
    torch.save(matrix, pt_file_path)

# 打印前10个平移参数
print("First 10 translations used (for inverse transformation):")
for i, (file, translate_x, translate_y) in enumerate(first_10_translations, start=1):
    print(f"Image {i} ({file}): Translate_x = {translate_x}, Translate_y = {translate_y}")

print(f"All {len(files)} inverse matrices have been saved as individual PT files.")