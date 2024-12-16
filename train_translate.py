import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from swin_transform_trans import CONFIGS as CONFIGS_TM
import swin_transform_trans as LNet
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

def load_single_tensor(index, tensor_folder):
    file_path = os.path.join(tensor_folder, f'matrix_{index + 1}.pt')
    return torch.load(file_path)

class ImageMSELoss(nn.Module):
    def __init__(self, use_cuda=True):
        super(ImageMSELoss, self).__init__()
        self.use_cuda = use_cuda

    def forward(self, theta_BA, theta_GT_BA, images):
        # Apply the gt and predicted affine transformations
        grid_size = images.size(2)
        batch_size = images.size(0)

        # Create the affine grid
        grid_GT = F.affine_grid(theta_GT_BA, images.size(), align_corners=False)
        grid_pred = F.affine_grid(theta_BA, images.size(), align_corners=False)

        # Apply the transformations
        transformed_GT = F.grid_sample(images, grid_GT, align_corners=False)
        transformed_pred = F.grid_sample(images, grid_pred, align_corners=False)

        # Compute MSE loss on transformed images
        loss = F.mse_loss(transformed_pred, transformed_GT)

        return loss, transformed_pred, transformed_GT


def calculate_theta_BA(translate_x, translate_y):
    """
    Calculate the transformation matrix theta_BA based on the given translate values.

    translate_x: Tensor of shape (batch_size, 1) or (batch_size,) representing translate along the x-axis
    translate_y: Tensor of shape (batch_size, 1) or (batch_size,) representing translate along the y-axis
    """
    # Remove the second dimension if it exists
    if translate_x.ndimension() == 2:
        translate_x = translate_x.squeeze(dim=1)
    if translate_y.ndimension() == 2:
        translate_y = translate_y.squeeze(dim=1)

    batch_size = translate_x.size(0)
    theta_BA = torch.zeros(batch_size, 2, 3, device=translate_x.device)

    # Set up the affine transformation matrix for translation
    theta_BA[:, 0, 0] = 1  # No scaling along x-axis
    theta_BA[:, 1, 1] = 1  # No scaling along y-axis
    theta_BA[:, 0, 2] = translate_x  # Translation along x-axis
    theta_BA[:, 1, 2] = translate_y  # Translation along y-axis

    return theta_BA

# Dataset Class for Image Pairs
class ImageTransformPairDataset(Dataset):
    def __init__(self, img1_dir, img2_dir, restored_dir, transform_img1=None, transform_img2=None,
                 transform_restored=None):
        self.img1_paths = [os.path.join(img1_dir, f"{i}_restored_full.jpg") for i in range(1, 9001)]
        self.img2_paths = [os.path.join(img2_dir, f"{i}_restored_scale_rot_shear.jpg") for i in range(1, 9001)]
        self.restored_paths = [os.path.join(restored_dir, f"{i}_restored_full.jpg") for i in range(1, 9001)]

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
    mean_img1 = [0.1816]
    std_img1 = [0.2010]
    mean_img2 = [0.1852]
    std_img2 = [0.2012]
    mean_restored = [0.1816]
    std_restored = [0.2010]

    # 创建保存结果的目录
    results_dir = 'E:/download/megadepth_test_1500/C2FViT/training_results/translate_full'
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
    criterion_translate = ImageMSELoss()  # 使用MSE损失

    # Optimizer
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)

    # Initialize the best loss variable
    best_eval_loss = float('inf')
    best_model_path = None

    # Define the directory to save checkpoints
    save_dir = 'E:/download/megadepth_test_1500/C2FViT/checkpoints/affine/translate/aviation'
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    tensor_folder_path = 'E:/download/megadepth_test_1500/C2FViT/affine/train_full/tensor_translation'

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
            _, translate_x, translate_y = generator(img_pair)

            # Print the predicted translate values every 50 batches
            if (batch_index + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_index + 1}/{len(train_loader)}], "
                    f"Predicted translate X: {translate_x.detach().cpu().numpy()}, Predicted translate Y: {translate_y.detach().cpu().numpy()}"
                )

            theta_BA_pred = calculate_theta_BA(translate_x, translate_y)

            img2 = img_pair[:, 1].unsqueeze(1)  # Extract second image from pair

            if theta_BA_pred.size() == theta_BA_gt_batch.size():
                loss_G, transformed_pred, transformed_GT = criterion_translate(theta_BA_pred, theta_BA_gt_batch, img2)
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
                    _, translate_x, translate_y = generator(img_pair)

                    # Print the predicted translates every 30 batches
                    if (batch_index + 1) % 100 == 0:
                        print(
                            f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_index + 1}/{len(train_loader)}], "
                            f"Predicted translate X: {translate_x.detach().cpu().numpy()}, Predicted translate Y: {translate_y.detach().cpu().numpy()}"
                        )

                    theta_BA_pred = calculate_theta_BA(translate_x, translate_y)

                    img2 = img_pair[:, 1].unsqueeze(1)  # Extract second image from pair

                    if theta_BA_pred.size() == theta_BA_gt_batch.size():
                        # Use the image-based loss function
                        loss_G, transformed_pred, transformed_GT = criterion_translate(theta_BA_pred, theta_BA_gt_batch,
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
    img1_dir = 'E:/download/megadepth_test_1500/C2FViT/affine/train_full/imgs_restored_full'
    img2_dir = 'E:/download/megadepth_test_1500/C2FViT/affine/train_full/imgs_restored_scale_rot_shear'
    restored_dir = 'E:/download/megadepth_test_1500/C2FViT/affine/train_full/imgs_restored_full'  # Path to the directory containing restored images

    # Select configuration key
    config_key = 'TransMorph'  # Replace with the actual config key

    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to the saved model weights
    model_path = 'E:/download/megadepth_test_1500/C2FViT/checkpoints/affine/translate/aviation/LNet_best_TransMorph_Image_MSE_0.1490.pth'

    # Train the model, continue from the saved weights
    trained_model = train_model(img1_dir, img2_dir, restored_dir, config_key, device, num_epochs=40, batch_size=4,
                                lr=1e-6, model_path=model_path)
