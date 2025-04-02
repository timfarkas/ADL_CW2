import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_ch=32):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, base_ch)
        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)

        self.pool = nn.MaxPool2d(2)

        self.middle = conv_block(base_ch * 4, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_ch * 4 + base_ch * 2, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_ch * 2 + base_ch, base_ch)

        self.out = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        m = self.middle(self.pool(e3))

        d2 = self.up2(m)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        out = self.out(d1)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        return out

def fit_sgd_pixel(model_train,
            dataloader_new,
            number_epoch: int,
            learning_rate: float,
            loss_function: torch.nn.Module,
            model_path: str,
            device: str = None) -> None:
    """
    Train a segmentation model using Stochastic Gradient Descent.

    Args:
        model_train: The neural network model to train
        trainloader: DataLoader containing the training data
        number_epoch: Number of epochs to train for
        learning_rate: Learning rate for the optimizer
        loss_function: Loss function to optimize (e.g., nn.CrossEntropyLoss for multi-class segmentation)
        model_path: Path where the trained model will be saved
        device: Device to run the training on ('cuda', 'cpu', etc.)
    """
    torch.autograd.set_detect_anomaly(True)
    print("<Training Start>")
    model_train.to(device)
    optimizer = optim.SGD(model_train.parameters(), lr=learning_rate, momentum=0.9)


    for epoch in range(number_epoch):
        correct_pixels = 0
        total_pixels = 0
        total_loss = 0
        # batch_count = 0
        for images, masks in dataloader_new:
            # print("Batch:",batch_count)
            images = images.to(device)
            masks = masks.to(device).float()
            optimizer.zero_grad()
            outputs = model_train(images)  # [B, num_classes, H, W]

            loss = loss_function(outputs, masks)
            loss.backward()
            optimizer.step()

            # Calculate pixel-wise accuracy
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5  # threshold logits
            correct_pixels += (preds == masks.bool()).sum().item()
            total_pixels += masks.numel()
            total_loss += loss.item() * images.size(0)

            # batch_count+=1
        avg_loss = total_loss / len(dataloader_new.dataset)
        pixel_accuracy = correct_pixels / total_pixels

        print(f"Epoch {epoch + 1}/{number_epoch}, Pixel Accuracy: {pixel_accuracy:.4f}, Loss: {avg_loss:.4f}")

    torch.save(model_train.state_dict(), model_path)
    print("Model saved. Number of parameters:", sum(p.numel() for p in model_train.parameters()))

def predict_pixel_classification_dataset(model: nn.Module,
                                         dataloader: torch.utils.data.DataLoader,
                                         device: str = 'cpu',
                                         threshold: float = 0.2):
    """
    Run inference on a dataset and return a new dataset with images and filtered predicted pixel-level probabilities.

    Args:
        model: Trained segmentation model (e.g., UNet).
        dataloader: DataLoader containing input images.
        device: 'cuda', 'mps', or 'cpu'.
        threshold: Minimum confidence threshold; values below will be set to 0.

    Returns:
        TensorDataset: Dataset containing (images, filtered_probs) where probs ∈ [0,1]
    """
    model.eval()
    model.to(device)

    image_list = []
    prob_mask_list = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            logits = model(images)  # [B, 1, H, W]
            probs = torch.sigmoid(logits)  # ∈ [0,1]
            filtered_probs = probs * (probs > threshold).float()  # Zero out low-confidence pixels

            image_list.append(images.cpu())
            prob_mask_list.append(filtered_probs.cpu())

    all_images = torch.cat(image_list, dim=0)
    all_probs = torch.cat(prob_mask_list, dim=0)

    print(f"Generated filtered probability masks for {len(all_images)} samples.")
    return TensorDataset(all_images, all_probs)
def visualize_predicted_masks(dataset, num_samples=4, save_path=None):
    """
    Visualize predicted masks with 4xN layout from a single batch:
    Row 1: First N images
    Row 2: Corresponding N masks
    Row 3: Next N images
    Row 4: Corresponding N masks
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_samples * 2, shuffle=False)
    images, masks = next(iter(dataloader))  # One big batch

    images1, images2 = images[:num_samples], images[num_samples:]
    masks1, masks2 = masks[:num_samples], masks[num_samples:]

    fig, axs = plt.subplots(4, num_samples, figsize=(num_samples * 2, 4 * 2))

    for i in range(num_samples):
        img1 = images1[i].permute(1, 2, 0).cpu().numpy()
        mask1 = masks1[i][0].cpu().numpy()
        img2 = images2[i].permute(1, 2, 0).cpu().numpy()
        mask2 = masks2[i][0].cpu().numpy()

        axs[0, i].imshow(img1)
        axs[0, i].set_title(f"Image {i+1}")
        axs[1, i].imshow(mask1, cmap='gray')
        axs[1, i].set_title(f"Mask {i+1}")
        axs[2, i].imshow(img2)
        axs[2, i].set_title(f"Image {i+1 + num_samples}")
        axs[3, i].imshow(mask2, cmap='gray')
        axs[3, i].set_title(f"Mask {i+1 + num_samples}")

        for row in range(4):
            axs[row, i].axis('off')

    plt.tight_layout()

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")

    plt.close()  # Just close it silently without showing

def visualize_first_batch(dataloader_new, round_num, num_samples=8):
    """
    Visualize the first `num_samples` images and labels from dataloader_new.
    """
    first_batch = next(iter(dataloader_new))
    images, labels = first_batch

    # Limit to num_samples
    images = images[:num_samples]
    labels = labels[:num_samples]

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        mask = labels[i][0].cpu().numpy()

        axs[0, i].imshow(img)
        axs[0, i].set_title(f"Round {round_num} - Img {i + 1}")
        axs[0, i].axis("off")

        axs[1, i].imshow(mask, cmap="gray")
        axs[1, i].set_title("Label")
        axs[1, i].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()
