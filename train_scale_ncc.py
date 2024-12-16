import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from swin_transform_3 import CONFIGS as CONFIGS_TM
import swin_transform_3 as LNet
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

def load_single_tensor(index, tensor_folder):
    file_path = os.path.join(tensor_folder, f'matrix_{index + 1}.pt')
    return torch.load(file_path)

class ImageNCCLoss(nn.Module):
    def __init__(self, epsilon=1e-5, crop_fraction=5/8):
        """
        Args:
            epsilon (float): 防止除以零的小常数。
            crop_fraction (float): 中心裁剪的比例，默认为 5/8。
        """
        super(ImageNCCLoss, self).__init__()
        self.epsilon = epsilon
        self.crop_fraction = crop_fraction  # 中心裁剪比例

    def forward(self, theta_pred, theta_gt, images2):
        """
        Args:
            theta_pred (torch.Tensor): 预测的仿射变换参数，形状 [batch_size, 2, 3]
            theta_gt (torch.Tensor): 真实的仿射变换参数，形状 [batch_size, 2, 3]
            images2 (torch.Tensor): 待对齐图像，形状 [batch_size, channels, height, width]

        Returns:
            loss (torch.Tensor): NCC 损失
            transformed_pred (torch.Tensor): 预测仿射变换后的图像2
            transformed_GT (torch.Tensor): 真实仿射变换后的图像2
        """
        batch_size, channels, height, width = images2.size()

        # 创建仿射网格
        grid_pred = F.affine_grid(theta_pred, images2.size(), align_corners=False)
        grid_gt = F.affine_grid(theta_gt, images2.size(), align_corners=False)

        # 应用仿射变换
        transformed_pred = F.grid_sample(images2, grid_pred, align_corners=False)
        transformed_GT = F.grid_sample(images2, grid_gt, align_corners=False)

        # 计算中心裁剪区域
        crop_fraction = self.crop_fraction
        crop_height = int(height * crop_fraction)
        crop_width = int(width * crop_fraction)
        h_start = (height - crop_height) // 2
        w_start = (width - crop_width) // 2
        h_end = h_start + crop_height
        w_end = w_start + crop_width

        # 裁剪图像到中心区域
        transformed_pred_cropped = transformed_pred[:, :, h_start:h_end, w_start:w_end]
        transformed_GT_cropped = transformed_GT[:, :, h_start:h_end, w_start:w_end]

        # 确保张量在内存中是连续的
        transformed_pred_cropped = transformed_pred_cropped.contiguous()
        transformed_GT_cropped = transformed_GT_cropped.contiguous()

        # 矢量化 NCC 计算
        # [batch_size, channels, H, W] -> [batch_size, channels, H*W]
        transformed_pred_flat = transformed_pred_cropped.view(batch_size, channels, -1)
        transformed_GT_flat = transformed_GT_cropped.view(batch_size, channels, -1)

        # 计算均值
        pred_mean = transformed_pred_flat.mean(dim=2, keepdim=True)
        gt_mean = transformed_GT_flat.mean(dim=2, keepdim=True)

        # 去均值
        pred_norm = transformed_pred_flat - pred_mean
        gt_norm = transformed_GT_flat - gt_mean

        # 计算分子和分母
        numerator = (pred_norm * gt_norm).sum(dim=2)  # [batch_size, channels]
        denominator = torch.sqrt((pred_norm ** 2).sum(dim=2) * (gt_norm ** 2).sum(dim=2) + self.epsilon)  # [batch_size, channels]

        # 计算 NCC
        ncc = numerator / denominator  # [batch_size, channels]

        # 计算损失
        loss = 1 - ncc.mean()

        return loss, transformed_pred, transformed_GT

def calculate_theta_BA(scale, inverse_scale=False):
    """
    Calculate the transformation matrix theta_BA based on the given scale.
    scale: Tensor of shape (batch_size, 1) or (batch_size,)
    inverse_scale: Boolean, indicating if the input scales need to be inverted (1/scale)
    """
    if scale.ndimension() == 2:
        scale = scale.squeeze(dim=1)  # Remove second dimension

    batch_size = scale.size(0)
    theta_BA = torch.zeros(batch_size, 2, 3, device=scale.device)

    theta_BA[:, 0, 0] = scale  # Scale along x-axis
    theta_BA[:, 1, 1] = scale  # Scale along y-axis
    theta_BA[:, 0, 1] = 0  # No shear
    theta_BA[:, 1, 0] = 0  # No shear
    theta_BA[:, 0, 2] = 0  # No translation
    theta_BA[:, 1, 2] = 0  # No translation

    return theta_BA

# Dataset Class for Image Pairs
class ImageTransformPairDataset(Dataset):
    def __init__(self, img1_dir, img2_dir, restored_dir, transform_img1=None, transform_img2=None,
                 transform_restored=None):
        self.img1_paths = [os.path.join(img1_dir, f"{i}.png") for i in range(1, 9001)]
        self.img2_paths = [os.path.join(img2_dir, f"{i}_de.png") for i in range(1, 9001)]
        self.restored_paths = [os.path.join(restored_dir, f"{i}_restored_scale.png") for i in range(1, 9001)]

        self.transform_img1 = transform_img1
        self.transform_img2 = transform_img2
        self.transform_restored = transform_restored

        missing_files = [p for p in self.img1_paths if not os.path.exists(p)] + \
                        [p for p in self.img2_paths if not os.path.exists(p)] + \
                        [p for p in self.restored_paths if not os.path.exists(p)]
        if missing_files:
            print(f"Missing files: {missing_files}")
            raise FileNotFoundError(f"Missing {len(missing_files)} files. First missing file: {missing_files[0]}")

    def __len__(self):
        return len(self.img1_paths)

    def __getitem__(self, idx):
        img1 = Image.open(self.img1_paths[idx]).convert('L')  # Convert to grayscale
        img2 = Image.open(self.img2_paths[idx]).convert('L')  # Convert to grayscale
        restored_img = Image.open(self.restored_paths[idx]).convert('L')  # Convert to grayscale

        if self.transform_img1:
            img1 = self.transform_img1(img1)
        else:
            img1 = transforms.ToTensor()(img1)

        if self.transform_img2:
            img2 = self.transform_img2(img2)
        else:
            img2 = transforms.ToTensor()(img2)

        if self.transform_restored:
            restored_img = self.transform_restored(restored_img)
        else:
            restored_img = transforms.ToTensor()(restored_img)

        # Stack img1 and img2 to create a (2, 1, height, width) tensor
        img_pair = torch.stack([img1, img2], dim=0)

        # Optionally remove the channel dimension to get shape (2, height, width)
        img_pair = img_pair.squeeze(1)

        return img_pair, restored_img


# Training Function
def train_model(img1_dir, img2_dir, restored_dir, config_key, device, num_epochs=50, batch_size=4, lr=1e-4,
                model_path=None):
    # Mean and Std for img1, img2, and restored_img
    mean_img1 = [0.3639]
    std_img1 = [0.1338]
    mean_img2 = [0.1904]
    std_img2 = [0.1862]
    mean_restored = [0.1875]
    std_restored = [0.2014]

    # 创建保存结果的目录
    results_dir = 'E:/download/megadepth_test_1500/C2FViT/training_results_full/scale/ncc'
    os.makedirs(results_dir, exist_ok=True)

    # 初始化记录列表
    train_losses = []
    eval_losses = []

    # Define separate transforms for img1, img2, and restored_img
    transform_img1 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_img1, std=std_img1)
    ])

    transform_img2 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_img2, std=std_img2)
    ])

    transform_restored = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_restored, std=std_restored)
    ])

    # Initialize dataset and split into train and evaluation sets
    dataset = ImageTransformPairDataset(img1_dir, img2_dir, restored_dir,
                                        transform_img1=transform_img1,
                                        transform_img2=transform_img2,
                                        transform_restored=transform_restored)

    train_size = 7000
    eval_size = 2000
    train_dataset = Subset(dataset, range(train_size))  # First 7000 for training
    eval_dataset = Subset(dataset, range(train_size, train_size + eval_size))  # Next 2000 for evaluation

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load configuration
    config = CONFIGS_TM[config_key]

    # Initialize the generator (LNet)
    generator = LNet.LNet(config).to(device)

    try:
        if model_path and os.path.exists(model_path):
            generator.load_state_dict(torch.load(model_path))
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Model path does not exist or was not provided: {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")

    # Loss function
    criterion_scale = ImageNCCLoss()  # 使用MSE损失

    # Optimizer
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)

    # Initialize the best loss variable
    best_eval_loss = float('inf')
    best_model_path = None

    # Define the directory to save checkpoints
    save_dir = 'E:/download/megadepth_test_1500/C2FViT/checkpoints/affine/scale/aviation/ncc'
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    tensor_folder_path = 'E:/download/megadepth_test_1500/C2FViT/NCC/train/tensor_scale'

    # Training loop
    for epoch in range(num_epochs):
        generator.train()
        epoch_loss_G = 0.0
        batch_count = 0

        for batch_index, (img_pair, restored_img) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            img_pair = img_pair.to(device)
            restored_img = restored_img.to(device)

            # -----------------
            #  Train Generator
            # -----------------
            # 为批次中的每个样本加载对应的 tensor，并组合成一个批次
            theta_BA_gt_list = []
            for i in range(img_pair.size(0)):  # Use img_pair.size(0) to handle the last batch size
                sample_index = batch_index * batch_size + i
                theta_BA_gt = load_single_tensor(sample_index, tensor_folder_path)
                theta_BA_gt_list.append(theta_BA_gt)

            theta_BA_gt_batch = torch.stack(theta_BA_gt_list).to(device)

            optimizer_G.zero_grad()
            _, scale = generator(img_pair)

            # Print the predicted scales every 30 batches
            if (batch_index + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_index + 1}/{len(train_loader)}], Predicted Scales: {scale.detach().cpu().numpy()}")

            theta_BA_pred = calculate_theta_BA(scale)

            img2 = img_pair[:, 1].unsqueeze(1)  # Extract second image from pair

            if theta_BA_pred.size() == theta_BA_gt_batch.size():
                loss_G, transformed_pred, transformed_GT = criterion_scale(theta_BA_pred, theta_BA_gt_batch, img2)
                loss_G.backward()
                optimizer_G.step()

                epoch_loss_G += loss_G.item()
                batch_count += 1

                # Visualize transformed images for the first batch of the first epoch
                if epoch == 0 and batch_index == 0:
                    visualize_transformed_images(img_pair, transformed_pred, transformed_GT, batch_index, batch_size)

                if epoch == 0 and batch_index == 5:
                    visualize_transformed_images(img_pair, transformed_pred, transformed_GT, batch_index, batch_size)

        # Print epoch loss
        avg_train_loss = epoch_loss_G / batch_count if batch_count > 0 else 0
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss G (Train): {avg_train_loss:.3f}")

        # Evaluation phase
        generator.eval()
        eval_loss_G = 0.0
        eval_batch_count = 0

        with torch.no_grad():
            for batch_index, (img_pair, restored_img) in enumerate(tqdm(eval_loader, desc="Evaluating")):
                img_pair = img_pair.to(device)
                restored_img = restored_img.to(device)

                # Initialize list to store ground truth tensors
                theta_BA_gt_list = []

                # Calculate the starting index for the evaluation subset
                eval_start_index = train_size  # Since evaluation indices start right after train indices

                for i in range(img_pair.size(0)):  # Use img_pair.size(0) for handling the last batch
                    sample_index = eval_start_index + batch_index * batch_size + i
                    try:
                        theta_BA_gt = load_single_tensor(sample_index, tensor_folder_path)
                        theta_BA_gt_list.append(theta_BA_gt)
                    except FileNotFoundError:
                        print(f"File not found for sample index: {sample_index}")
                        continue

                # Ensure we have a complete batch of ground truth tensors
                if len(theta_BA_gt_list) == img_pair.size(0):
                    theta_BA_gt_batch = torch.stack(theta_BA_gt_list).to(device)
                    _, scale = generator(img_pair)

                    # Print the predicted scales every 30 batches
                    if (batch_index + 1) % 100 == 0:
                        print(
                            f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_index + 1}/{len(train_loader)}], Predicted Scales: {scale.detach().cpu().numpy()}")

                    theta_BA_pred = calculate_theta_BA(scale)

                    img2 = img_pair[:, 1].unsqueeze(1)  # Extract second image from pair

                    if theta_BA_pred.size() == theta_BA_gt_batch.size():
                        # Use the image-based loss function
                        loss_G, transformed_pred, transformed_GT = criterion_scale(theta_BA_pred, theta_BA_gt_batch,
                                                                                   img2)
                        eval_loss_G += loss_G.item()
                        eval_batch_count += 1

                        if epoch == 0 and batch_index == 0:
                            visualize_transformed_images(img_pair, transformed_pred, transformed_GT, batch_index,
                                                         batch_size)

        # Calculate average evaluation loss
        avg_eval_loss = eval_loss_G / eval_batch_count if eval_batch_count > 0 else 0
        eval_losses.append(avg_eval_loss)

        # Save the best model
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_model_path = os.path.join(save_dir, f'LNet_best_{config_key}_Image_MSE_{best_eval_loss:.4f}.pth')
            torch.save(generator.state_dict(), best_model_path)
            print(f"Saved new best model: {best_model_path}")

        # Print train and eval results for the current epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")

    # 训练结束后,绘制总体损失曲线图
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), eval_losses, label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Overall Training and Evaluation Loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'overall_loss_curve_{config_key}.png'))
    plt.close()

    # 保存总体数据到Excel
    df = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Train Loss': train_losses,
        'Eval Loss': eval_losses
    })
    df.to_excel(os.path.join(results_dir, f'overall_training_results_{config_key}.xlsx'), index=False)

    print(f"Finished Training. Best model saved as {best_model_path}")
    print(f"Overall training results saved in {results_dir}")

    return generator


# Model Saving Function
def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# Function to Visualize Images and Model Output
def visualize_images(img_pair_batch, restored_img, model_output):
    batch_size = img_pair_batch.shape[0]
    fig, axs = plt.subplots(batch_size, 4, figsize=(20, batch_size * 5))  # 修改这里的列数为4

    for i in range(batch_size):
        # 原始图像（输入图像对的第一张）
        origin_image = img_pair_batch[i, 0].squeeze(0).cpu().numpy()

        # 待变换图像（输入图像对的第二张）
        source_image = img_pair_batch[i, 1].squeeze(0).cpu().numpy()

        # 应该变换后的真实图像
        target_image = restored_img[i].squeeze(0).cpu().numpy()

        # 实际预测得到的变换图像
        output_image = model_output[i].squeeze(0).cpu().detach().numpy()

        # 显示原始图像
        axs[i, 0].imshow(origin_image, cmap='gray')
        axs[i, 0].set_title(f'Origin Image {i + 1}')
        axs[i, 0].axis('off')

        # 显示待变换图像
        axs[i, 1].imshow(source_image, cmap='gray')
        axs[i, 1].set_title(f'Source Image {i + 1}')
        axs[i, 1].axis('off')

        # 显示应该变换后的真实图像
        axs[i, 2].imshow(target_image, cmap='gray')
        axs[i, 2].set_title(f'Target Image {i + 1}')
        axs[i, 2].axis('off')

        # 显示实际预测得到的变换图像
        axs[i, 3].imshow(output_image, cmap='gray')  # 使用一个新的坐标位置来显示
        axs[i, 3].set_title(f'Model Output {i + 1}')
        axs[i, 3].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_transformed_images(input_imgs, transformed_pred, transformed_GT, batch_index, batch_size):
    # Ensure the tensors are detached and on the CPU
    input_imgs = input_imgs.detach().cpu()
    transformed_pred = transformed_pred.detach().cpu()
    transformed_GT = transformed_GT.detach().cpu()

    # Convert tensors to numpy arrays
    input_imgs_np = input_imgs.numpy()
    transformed_pred_np = transformed_pred.numpy()
    transformed_GT_np = transformed_GT.numpy()

    # Print shape for debugging
    print(f"Shape of input image pair: {input_imgs_np.shape}")

    # Set up the plot; we'll visualize all images in the batch in order
    fig, axes = plt.subplots(batch_size, 4, figsize=(20, 5 * batch_size))

    for i in range(batch_size):
        # Visualize the first image in the input pair
        ax_input1 = axes[i, 0]
        img_input1 = input_imgs_np[i, 0]
        ax_input1.imshow(np.squeeze(img_input1), cmap='gray')  # Squeeze to remove the single channel dimension
        ax_input1.set_title(f'Batch {batch_index}, Sample {i + 1}: Input Image 1')
        ax_input1.axis('off')

        # Visualize the second image in the input pair
        ax_input2 = axes[i, 1]
        img_input2 = input_imgs_np[i, 1]
        ax_input2.imshow(np.squeeze(img_input2), cmap='gray')
        ax_input2.set_title(f'Batch {batch_index}, Sample {i + 1}: Input Image 2')
        ax_input2.axis('off')

        # Plot the predicted transformation
        ax_pred = axes[i, 2]
        img_pred = transformed_pred_np[i]
        ax_pred.imshow(np.squeeze(img_pred), cmap='gray')  # Use squeeze to remove the single channel dimension
        ax_pred.set_title(f'Batch {batch_index}, Sample {i + 1}: Transformed Prediction')
        ax_pred.axis('off')

        # Plot the ground truth transformation
        ax_gt = axes[i, 3]
        img_gt = transformed_GT_np[i]
        ax_gt.imshow(np.squeeze(img_gt), cmap='gray')  # Use squeeze to remove the single channel dimension
        ax_gt.set_title(f'Batch {batch_index}, Sample {i + 1}: Transformed Ground Truth')
        ax_gt.axis('off')

    plt.tight_layout()
    plt.show()

# Main Function
if __name__ == '__main__':
    # Define data directories
    img1_dir = 'E:/download/megadepth_test_1500/C2FViT/NCC/train/imgs1'
    img2_dir = 'E:/download/megadepth_test_1500/C2FViT/NCC/train/imgs2'
    restored_dir = 'E:/download/megadepth_test_1500/C2FViT/NCC/train/imgs_restored_scale'  # Path to the directory containing restored images

    # Select configuration key
    config_key = 'TransMorph'  # Replace with the actual config key

    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to the saved model weights
    model_path = 'E:/download/megadepth_test_1500/C2FViT/checkpoints/affine/scale/aviation/ncc/LNet_best_TransMorph_Image_MSE_0.3692.pth'

    # Train the model, continue from the saved weights
    trained_model = train_model(img1_dir, img2_dir, restored_dir, config_key, device, num_epochs=10, batch_size=4,
                                lr=1e-6, model_path=model_path)
