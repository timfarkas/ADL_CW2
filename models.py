from typing import Literal
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
)
import os
from utils import resize_images, unnormalize
from torch.utils.data import TensorDataset

### Num Inputs:
#       Breed:                  37
#       Species:                2
#       Bbox:                   256x256
#       Breed + Species??:      39
#       Breed + Bbox:           37 + 256x256
#       Species + Bbox:         2 + 256x256
#       Breed+Species+Bbox?:    39 + 256x256


class BasicCAMWrapper(nn.Module):
    def __init__(self, model):
        """
        Wrapper class for models to generate Class Activation Maps (CAM) during forward pass

        Args:
            model: The base model to wrap (CNN or ResNetBackbone)
        """
        super().__init__()
        self.model = model
        self.hooks = []

    def forward(self, x, target_class=None, return_cam=False):
        """
        Forward pass that can optionally return CAM visualization

        Args:
            x: Input tensor (images)
            target_class: Optional class index to generate CAM for. If None and return_cam=True,
                          uses predicted class
            return_cam: If True, returns CAM along with model output

        Returns:
            - If return_cam=False: model output (logits)
            - If return_cam=True: tuple of (logits, cam_image, predicted_class)
        """
        # Get model outputs and feature maps
        logits, feature_maps = self.model(x, return_features=True)

        batch_size = x.size(0)
        cams = []

        for b in range(batch_size):
            feature_map = feature_maps[b]  # (C, H, W) for this batch item

            # Determine class for CAM visualization
            if target_class is None:
                # Use predicted class if target not specified
                _, predicted_class = torch.max(logits[b], 0)
                current_target = predicted_class.item()
            else:
                current_target = target_class

            # Compute CAM for this batch item
            classifier_weights = self.model.classifier.weight.data
            cam = self._compute_cam(feature_map, classifier_weights, current_target)
            cam = resize_images(cam, 256, 256)
            cams.append(torch.tensor(cam))

        # Stack all CAMs into a batch
        batch_cams = torch.stack(cams)

        return batch_cams

    def _compute_cam(self, feature_map, classifier_weights, class_idx):
        """Compute class activation map"""
        weights = (
            classifier_weights[class_idx].unsqueeze(1).unsqueeze(2)
        )  # Shape: (C, 1, 1)
        cam = torch.sum(weights * feature_map, dim=0)  # Shape: (H, W)
        cam = cam.detach().cpu().numpy()
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)  # Normalize to [0, 1]
        return cam

    def _generate_cam_image(self, img_tensor, cam):
        """Generate CAM overlay on the original image"""
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        h, w = img_np.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        heatmap = heatmap.astype(np.float32) / 255

        # Create overlay
        overlay = 0.5 * img_np + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)

        return overlay

    def visualize_batch(self, images, labels=None, num_images=4):
        """
        Visualize CAM for a batch of images

        Args:
            images: Batch of images
            labels: Optional ground truth labels
            num_images: Number of images to visualize
        """
        # Limit to specified number of images
        images = images[:num_images]

        # Prepare figure
        fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(4 * num_images, 8))

        with torch.no_grad():
            logits, feature_maps = self.model(images, return_features=True)

            for i in range(num_images):
                img = images[i]
                fmap = feature_maps[i]

                # Determine target class
                if labels is not None:
                    target_class = labels[i].item()
                else:
                    _, target_class = torch.max(logits[i], 0)
                    target_class = target_class.item()

                # Compute CAM
                cam = self._compute_cam(
                    fmap, self.model.classifier.weight.data, target_class
                )
                img_np = img.permute(1, 2, 0).cpu().numpy()

                # Original image on first row
                axes[0][i].imshow(img_np)
                axes[0][i].set_title(f"Original - Class {target_class}")
                axes[0][i].axis("off")

                # CAM overlay on second row
                cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
                cam_normalized = np.clip(cam_resized, 0, 1)

                heatmap = cv2.applyColorMap(
                    np.uint8(255 * cam_normalized), cv2.COLORMAP_JET
                )
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                heatmap = heatmap.astype(np.float32) / 255.0

                overlay = 0.5 * img_np + 0.5 * heatmap
                overlay = np.clip(overlay, 0, 1)

                axes[1][i].imshow(overlay)
                axes[1][i].set_title(f"CAM - Class {target_class}")
                axes[1][i].axis("off")

        plt.tight_layout()
        plt.show()


class CAMManager:
    """
    Class that manages different CAM methods based on the grad-cam package.
    It allows to visualise CAMs and can prepare outputs for self-training.
    """

    dataset: TensorDataset

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        target_type: str,
        method: Literal["GradCAM", "ScoreCAM", "AblationCAM"] = "GradCAM",
        target_layer=None,
        smooth: bool = False,
    ):
        """
        Args:
            model: The model to generate CAMs for
            method: CAM method ('GradCAM', 'HiResCAM', etc. naming based on grad-cam package)
            target_layer: Target layer for CAM methods, if None will be auto-detected
        """
        self.model = model
        self.method = method

        # Auto-detect target layers if not specified
        if target_layer is None:
            if hasattr(model, "features") and hasattr(model.features, "__getitem__"):
                if isinstance(model, ResNetBackbone):
                    self.target_layers = [
                        model.features[-1][-1]
                    ]  # Last block of layer4 in ResNet
                else:  # CNN
                    self.target_layers = [
                        model.features[-2]
                    ]  # Last conv layer before GAP in CNN
            else:
                raise ValueError(
                    "Unsupported model type for CAM, could not detect target layers."
                )
        else:
            self.target_layers = [target_layer]

        # Initialise the appropriate CAM method
        match method:
            case "GradCAM":
                from pytorch_grad_cam import GradCAM

                self.cam = GradCAM(model=model, target_layers=self.target_layers)
            case "ScoreCAM":
                from pytorch_grad_cam import ScoreCAM

                self.cam = ScoreCAM(model=model, target_layers=self.target_layers)
            case "AblationCAM":
                from pytorch_grad_cam import AblationCAM

                self.cam = AblationCAM(model=model, target_layers=self.target_layers)
            case _:
                raise ValueError(f"Unsupported CAM method: {method}")

        self.cam.batch_size = dataloader.batch_size
        self.dataset = self._generate_cam_dataset(
            dataloader=dataloader, target_type=target_type, smooth=smooth
        )

    def _generate_cam_dataset(self, dataloader, target_type, smooth: bool):
        """
        Generate a dataset with CAM masks for self-training.

        Args:
            dataloader: DataLoader with images

        Returns:
            TensorDataset: Contains (images, cam_mask, segmentation_masks),
                where masks are [B, 1, H, W].
        """
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        self.model.eval()

        all_images = []
        all_cams = []
        all_masks = []

        device = next(self.model.parameters()).device

        for batch_images, batch_targets in dataloader:
            all_images.append(batch_images)
            images = batch_images.to(device)

            try:
                labels = batch_targets[target_type].to(device)
                gt_masks = batch_targets["segmentation"].to(device)
            except ValueError or KeyError:
                raise ValueError(
                    f"Expected dict with keys '{target_type}' and 'segmentation'"
                )

            targets = [ClassifierOutputTarget(label.item()) for label in labels]

            grayscale_cams = self.cam(
                input_tensor=images,
                targets=targets,
                aug_smooth=smooth,
                eigen_smooth=smooth,
            )
            tensor_cams = torch.from_numpy(grayscale_cams).float().unsqueeze(1)

            all_cams.append(tensor_cams)
            all_masks.append(gt_masks.cpu())

        # Concatenate all batches
        images_tensor = torch.cat(all_images, dim=0)
        cams_tensor = torch.cat(all_cams, dim=0)
        masks_tensor = torch.cat(all_masks, dim=0)

        return TensorDataset(images_tensor, cams_tensor, masks_tensor)


class BboxHead(nn.Module):
    def __init__(self, adapter="CNN"):
        super().__init__()
        if adapter.lower() == "cnn":
            num_inputs = 256
        elif adapter.lower() == "res18":
            num_inputs = 512
        elif adapter.lower() == "res50":
            num_inputs = 2048
        elif adapter.lower() == "res101":
            num_inputs = 2048
        else:
            num_inputs = 512
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (C,H,W) → (C,1,1)
            nn.Flatten(),  # → (C,)
            nn.Linear(num_inputs, 4),
            nn.Sigmoid(),  ### [cx, cy, w, h]
        )

        self.name = "BboxHead"

    def change_adapter(self, adapter):
        if adapter.lower() == "cnn":
            num_inputs = 256
        elif adapter.lower() == "res18":
            num_inputs = 512
        elif adapter.lower() == "res50":
            num_inputs = 2048
        elif adapter.lower() == "res101":
            num_inputs = 2048
        else:
            num_inputs = 512

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (C,H,W) → (C,1,1)
            nn.Flatten(),  # → (C,)
            nn.Linear(num_inputs, 4),
            nn.Sigmoid(),  ### [cx, cy, w, h]
        )
        self.name = "BBoxHead"

    def forward(self, z):
        return self.head(z)


class ClassifierHead(nn.Module):
    def __init__(self, num_classes=2, adapter="CNN"):
        super().__init__()
        if adapter.lower() == "cnn":
            num_inputs = 256
        elif adapter.lower() == "res18":
            num_inputs = 512
        elif adapter.lower() == "res50":
            num_inputs = 2048
        elif adapter.lower() == "res101":
            num_inputs = 2048
        else:
            num_inputs = 512
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (C,H,W) → (C,1,1)
            nn.Flatten(),  # → (C,)
            nn.Linear(num_inputs, num_classes),
            nn.Sigmoid(),
        )

        self.name = f"ClassifierHead({num_classes})"

    def change_adapter(self, adapter):
        if adapter.lower() == "cnn":
            num_inputs = 256
        elif adapter.lower() == "res18":
            num_inputs = 512
        elif adapter.lower() == "res50":
            num_inputs = 2048
        elif adapter.lower() == "res101":
            num_inputs = 2048
        else:
            num_inputs = 512

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (C,H,W) → (C,1,1)
            nn.Flatten(),  # → (C,)
            nn.Linear(num_inputs, self.num_classes),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.head(z)


class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Input: (B, 3, 256, 256)
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, padding=1
            ),  # (B, 32, 256, 256)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 32, 128, 128)
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),  # (B, 64, 128, 128)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 64, 64, 64)
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1
            ),  # (B, 128, 64, 64)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 128, 32, 32)
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1
            ),  # (B, 256, 32, 32)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 256, 16, 16)
        )

    def forward(self, img):
        return self.features(img)


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, model_type: str = "resnet18"):
        super().__init__()

        if model_type == "resnet18":
            if pretrained:
                base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                base_model = resnet18(weights=None)
        elif model_type == "resnet50":
            if pretrained:
                base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                base_model = resnet50(weights=None)
        elif model_type == "resnet101":
            if pretrained:
                base_model = resnet101(weights=ResNet101_Weights.DEFAULT)
            else:
                base_model = resnet101(weights=None)
        else:
            base_model = resnet18(weights=None)  # No pretrained weights

        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )

        # Adjust batch norm momentum for stronger normalization (deprecated)
        # Instead, cutting out ResNet101 to save on training speed and positing that WD + augmentations + smaller ResNets will address overfitting
        #for module in self.features.modules():
        #    if isinstance(module, nn.BatchNorm2d):
        #        module.momentum = 0.01 # Default is 0.1, lower value for "stronger" effect

    def forward(self, img, return_features=False):
        features = self.features(img)
        if return_features:
            return features, features
        return features


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
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        return out


class SelfTraining:
    @staticmethod
    def fit_sgd_pixel(
        model_train,
        dataloader_new,
        number_epoch: int,
        learning_rate: float,
        loss_function: torch.nn.Module,
        model_path: str,
        device: str = None,
    ) -> None:
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
            batch_count = 0
            correct_pixels = 0
            total_pixels = 0
            total_loss = 0
            # batch_count = 0
            total_batches = len(dataloader_new)
            for images, masks, masks_gt in dataloader_new:
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"Batch: {batch_count}/{total_batches}")
                images = images.to(device)
                masks = masks.to(device).float()
                # masks_gt = masks_gt.to(device).float()
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

            avg_loss = total_loss / len(dataloader_new.dataset)
            pixel_accuracy = correct_pixels / total_pixels

            print(
                f"Epoch {epoch + 1}/{number_epoch}, Pixel Accuracy: {pixel_accuracy:.4f}, Loss: {avg_loss:.4f}"
            )

        torch.save(model_train.state_dict(), model_path)
        print(
            "Model saved. Number of parameters:",
            sum(p.numel() for p in model_train.parameters()),
        )

    @staticmethod
    def predict_pixel_classification_dataset(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cpu",
        threshold: float = 0.2,
    ):
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
        gt_mask_list = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(device)
                logits = model(images)  # [B, 1, H, W]
                gt_masks = targets["segmentation"].to(device)
                probs = torch.sigmoid(logits)  # ∈ [0,1]
                filtered_probs = (
                    probs * (probs > threshold).float()
                )  # Zero out low-confidence pixels

                image_list.append(images.cpu())
                prob_mask_list.append(filtered_probs.cpu())
                gt_mask_list.append(gt_masks.cpu())

        all_images = torch.cat(image_list, dim=0)
        all_probs = torch.cat(prob_mask_list, dim=0)
        all_gts = torch.cat(gt_mask_list, dim=0)

        print(f"Generated filtered probability masks for {len(all_images)} samples.")
        return TensorDataset(all_images, all_probs, all_gts)

    @staticmethod
    def visualize_predicted_masks(dataset, num_samples=6, save_path=None):
        """
        Visualize predicted masks and ground-truth masks with 3xN layout:
        Row 1: Input images
        Row 2: Predicted masks
        Row 3: Ground truth masks
        """
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=num_samples, shuffle=False
        )
        images, masks, masks_gt = next(iter(dataloader))

        fig, axs = plt.subplots(3, num_samples, figsize=(num_samples * 3, 3 * 3))

        for i in range(num_samples):
            img = unnormalize(images[i]).permute(1, 2, 0).cpu().numpy()
            pred_mask = masks[i][0].cpu().numpy()
            gt_mask = masks_gt[i][0].cpu().numpy()

            axs[0, i].imshow(img)
            axs[0, i].set_title(f"Image {i + 1}")
            axs[1, i].imshow(pred_mask, cmap="gray")
            axs[1, i].set_title(f"Predicted")
            axs[2, i].imshow(gt_mask, cmap="gray")
            axs[2, i].set_title(f"Ground Truth")

            for row in range(3):
                axs[row, i].axis("off")

        plt.tight_layout()

        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")

        plt.close()

    @staticmethod
    def visualize_cam_samples(dataloader, num_samples=4):
        """
        Visualize image, CAM, and GT mask for a few samples from a DataLoader.

        Args:
            dataloader (DataLoader): DataLoader yielding (image, CAM, GT mask) batches
            num_samples (int): Number of samples to visualize
        """
        # Get one batch
        batch = next(iter(dataloader))
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            images, cams, masks = batch
        else:
            raise ValueError("Expected DataLoader to return (image, cam, mask) tuples.")

        # Slice to desired number of samples
        images = images[:num_samples]
        cams = cams[:num_samples]
        masks = masks[:num_samples]

        # Unnormalize images for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        unnormalized_images = images * std + mean
        unnormalized_images = unnormalized_images.clamp(0, 1)

        # Setup plot
        fig, axs = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))

        for i in range(num_samples):
            # Image
            img = unnormalized_images[i].permute(1, 2, 0).cpu().numpy()
            axs[0, i].imshow(img)
            axs[0, i].set_title(f"Image {i + 1}")
            axs[0, i].axis("off")

            # CAM (assumes shape (1, H, W))
            cam = cams[i].squeeze().cpu().numpy()
            axs[1, i].imshow(cam, cmap="jet")
            axs[1, i].set_title(f"CAM {i + 1}")
            axs[1, i].axis("off")

            # Ground Truth Mask (assumes shape (1, H, W))
            mask = masks[i].squeeze().cpu().numpy()
            axs[2, i].imshow(mask, cmap="gray")
            axs[2, i].set_title(f"GT Mask {i + 1}")
            axs[2, i].axis("off")

        plt.tight_layout()
        plt.show()

# Via binary classification, enables model to explicitly distinguish between animal and non-animal images
# This would hopefully improve CAMs by better identifying animal regions
class AnimalClassifierHead(nn.Module): 
    def __init__(self, adapter="CNN"):
        super().__init__()
        if adapter.lower() == "cnn":
            num_inputs = 256
        elif adapter.lower() == "res18":
            num_inputs = 512
        elif adapter.lower() == "res50":
            num_inputs = 2048
        elif adapter.lower() == "res101":
            num_inputs = 2048
        else:
            num_inputs = 512
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (C,H,W) → (C,1,1)
            nn.Flatten(),  # → (C,)
            nn.Linear(num_inputs, 2),  # 2 classes: 0=non-animal, 1=animal
            nn.Sigmoid(),
        )
        self.name = "AnimalClassifierHead"

    def change_adapter(self, adapter):
        if adapter.lower() == "cnn":
            num_inputs = 256
        elif adapter.lower() == "res18":
            num_inputs = 512
        elif adapter.lower() == "res50":
            num_inputs = 2048
        elif adapter.lower() == "res101":
            num_inputs = 2048
        else:
            num_inputs = 512
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (C,H,W) → (C,1,1)
            nn.Flatten(),  # → (C,)
            nn.Linear(num_inputs, 2),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.head(z)
