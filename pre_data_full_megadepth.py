import os
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

class ImageTransformPairDataset:
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # Apply preprocessing transformations (like Resize and ToTensor)
        if self.transform:
            original_image = self.transform(image)
        else:
            original_image = transforms.ToTensor()(image)

        # Generate random affine transformation parameters
        translate_x = random.uniform(-100, 100)  # Horizontal translation in pixels
        translate_y = random.uniform(-100, 100)  # Vertical translation in pixels
        shear_x = random.uniform(-30, 30)  # Shear in x-axis
        shear_y = random.uniform(-30, 30)  # Shear in y-axis
        angle = random.uniform(-90, 90)  # Rotation angle
        scale = random.uniform(0.5, 1.5)  # Scaling factor

        # Step 1: Apply translation
        translated_image = F.affine(
            original_image,
            angle=0,  # No rotation
            translate=(translate_x, translate_y),  # Apply translation
            scale=1,  # No scaling
            shear=(0, 0)  # No shear
        )

        # Step 2: Apply shear
        sheared_image = F.affine(
            translated_image,
            angle=0,  # No rotation
            translate=(0, 0),  # No additional translation
            scale=1,  # No scaling
            shear=(shear_x, shear_y)  # Apply shear
        )

        # Step 3: Apply rotation
        rotated_image = F.affine(
            sheared_image,
            angle=angle,  # Apply rotation
            translate=(0, 0),  # No additional translation
            scale=1,  # No scaling
            shear=(0, 0)  # No additional shear
        )

        # Step 4: Apply scaling
        transformed_image = F.affine(
            rotated_image,
            angle=0,  # No rotation
            translate=(0, 0),  # No translation
            scale=scale,  # Apply scaling
            shear=(0, 0)  # No shear
        )

        # Restore steps

        # 4.1: Restore scaling only
        restored_scale_only = F.affine(
            transformed_image,
            angle=0,  # No rotation
            translate=(0, 0),  # No translation
            scale=1 / scale,  # Reverse scaling
            shear=(0, 0)  # No shear
        )

        # 4.2: Restore scaling and rotation
        restored_scale_rot = F.affine(
            restored_scale_only,
            angle=-angle,  # Reverse rotation
            translate=(0, 0),  # No translation
            scale=1,  # No scaling
            shear=(0, 0)  # No shear
        )

        # 4.3: Restore scaling, rotation, and shear
        restored_scale_rot_shear = F.affine(
            restored_scale_rot,
            angle=0,  # No rotation
            translate=(0, 0),  # No translation
            scale=1,  # No scaling
            shear=(-shear_x, -shear_y)  # Reverse shear
        )

        # 4.4: Fully restore (scaling, rotation, shear, and translation)
        restored_full = F.affine(
            restored_scale_rot_shear,
            angle=0,  # No rotation
            translate=(-translate_x, -translate_y),  # Reverse translation
            scale=1,  # No scaling
            shear=(0, 0)  # No shear
        )

        return original_image, transformed_image, restored_scale_only, restored_scale_rot, restored_scale_rot_shear, restored_full, \
               angle, scale, (translate_x, translate_y), (shear_x, shear_y)

def collect_image_paths(root_dir, num_folders=25, images_per_folder=25):
    image_paths = []

    # List all directories in the root directory
    all_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    # Filter to only include folders with numeric names
    all_folders = sorted([f for f in all_folders if f.isdigit()])

    # Randomly select 25 folders
    selected_folders = random.sample(all_folders, min(num_folders, len(all_folders)))

    for folder in selected_folders:
        imgs_dir = os.path.join(root_dir, folder, 'dense0', 'imgs')
        if os.path.isdir(imgs_dir):
            all_images = [img for img in os.listdir(imgs_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            selected_images = random.sample(all_images, min(images_per_folder, len(all_images)))
            for img in selected_images:
                image_paths.append(os.path.join(imgs_dir, img))

    return image_paths

def save_images(dataset, output_dir, num_images=625):
    # Create directories to save images
    imgs1_dir = os.path.join(output_dir, 'imgs1')
    imgs2_dir = os.path.join(output_dir, 'imgs2')
    imgs_restored_scale_dir = os.path.join(output_dir, 'imgs_restored_scale')
    imgs_restored_scale_rot_dir = os.path.join(output_dir, 'imgs_restored_scale_rot')
    imgs_restored_scale_rot_shear_dir = os.path.join(output_dir, 'imgs_restored_scale_rot_shear')  # New directory
    imgs_restored_full_dir = os.path.join(output_dir, 'imgs_restored_full')
    params_dir = os.path.join(output_dir, 'params')

    os.makedirs(imgs1_dir, exist_ok=True)
    os.makedirs(imgs2_dir, exist_ok=True)
    os.makedirs(imgs_restored_scale_dir, exist_ok=True)
    os.makedirs(imgs_restored_scale_rot_dir, exist_ok=True)
    os.makedirs(imgs_restored_scale_rot_shear_dir, exist_ok=True)  # Create new directory
    os.makedirs(imgs_restored_full_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)

    for idx in range(min(num_images, len(dataset))):
        # Get image and transformation parameters from dataset
        original_image, transformed_image, restored_scale_only, restored_scale_rot, restored_scale_rot_shear, restored_full, \
        angle, scale, translation, shear = dataset[idx]

        # Define save paths
        original_image_path = os.path.join(imgs1_dir, f'{idx + 1}.jpg')
        transformed_image_path = os.path.join(imgs2_dir, f'{idx + 1}_de.jpg')
        restored_scale_only_path = os.path.join(imgs_restored_scale_dir, f'{idx + 1}_restored_scale.jpg')
        restored_scale_rot_path = os.path.join(imgs_restored_scale_rot_dir, f'{idx + 1}_restored_scale_rot.jpg')
        restored_scale_rot_shear_path = os.path.join(imgs_restored_scale_rot_shear_dir, f'{idx + 1}_restored_scale_rot_shear.jpg')  # New path
        restored_full_path = os.path.join(imgs_restored_full_dir, f'{idx + 1}_restored_full.jpg')
        params_path = os.path.join(params_dir, f'{idx + 1}_params.txt')

        # Save images
        transforms.ToPILImage()(original_image).save(original_image_path)
        transforms.ToPILImage()(transformed_image).save(transformed_image_path)
        transforms.ToPILImage()(restored_scale_only).save(restored_scale_only_path)
        transforms.ToPILImage()(restored_scale_rot).save(restored_scale_rot_path)
        transforms.ToPILImage()(restored_scale_rot_shear).save(restored_scale_rot_shear_path)  # Save the new image
        transforms.ToPILImage()(restored_full).save(restored_full_path)

        # Save transformation parameters
        with open(params_path, 'w') as f:
            f.write(f'Angle: {angle:.2f}\n')
            f.write(f'Scale: {scale:.2f}\n')
            f.write(f'Translation: ({translation[0]:.2f}, {translation[1]:.2f})\n')
            f.write(f'Shear: ({shear[0]:.2f}, {shear[1]:.2f})\n')


def main():
    # Define root directory
    root_dir = 'E:/MegaDepth/phoenix/S6/zl548/MegaDepth_v1'

    # Collect image paths
    image_paths = collect_image_paths(root_dir)

    # Define data preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Create dataset
    dataset = ImageTransformPairDataset(image_paths, transform=transform)

    # Define output directory
    output_dir = 'E:/download/megadepth_test_1500/C2FViT/affine/eval_full/megadepth'

    # Save images
    save_images(dataset, output_dir, num_images=625)
    print("图像处理和保存完成！")

if __name__ == "__main__":
    main()