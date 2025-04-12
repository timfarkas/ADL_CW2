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
from evaluation import get_categories_from_normalization

import numpy as np
# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
#

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
        self.classical=False

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
                self.cam.batch_size = dataloader.batch_size
            case "ScoreCAM":
                from pytorch_grad_cam import ScoreCAM

                self.cam = ScoreCAM(model=model, target_layers=self.target_layers)
                self.cam.batch_size = dataloader.batch_size
            case "AblationCAM":
                from pytorch_grad_cam import AblationCAM

                self.cam = AblationCAM(model=model, target_layers=self.target_layers)
                self.cam.batch_size = dataloader.batch_size
            case "Classical":
                self.classical = True

            case _:
                raise ValueError(f"Unsupported CAM method: {method}")


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
                # If batch_targets is a dict
                if isinstance(batch_targets, dict):
                    print("dict")
                    labels = batch_targets[target_type].to(device)
                    gt_masks = batch_targets["segmentation"].to(device)
                # If it's a tuple of (label, mask)
                elif isinstance(batch_targets, (tuple, list)):
                    print("tuple")
                    labels = batch_targets[0].to(device)
                    gt_masks = batch_targets[1].to(device)
                # If it's just the label tensor
                else:
                    labels = batch_targets.to(device)
                    gt_masks = labels

                    '''checking by visualization'''
                    # print("labels shape:", labels.shape)
                    # print("labels unique values:", torch.unique(labels))
                    # print("gt_masks shape:", gt_masks.shape)
                    # print("gt_masks unique values:", torch.unique(gt_masks))
                    # for i in range(min(4, gt_masks.shape[0])):
                    #     plt.figure(figsize=(4, 4))
                    #     plt.imshow(gt_masks[i].squeeze().cpu().numpy(), cmap='gray')
                    #     plt.title(f"Mask {i}")
                    #     plt.axis('off')
                    #     plt.show()

            except ValueError or KeyError:
                raise ValueError(
                    f"Expected dict with keys '{target_type}' and 'segmentation'"
                )

            if self.classical:
                with torch.no_grad():
                    logits, feature_maps = self.model(batch_images,
                                                      return_features=True)  # logits: (B, C), fmap: (B, C, H, W)
                    weights = self.model.classifier.weight.data  # shape: (num_classes, C)

                    pred_classes = logits.argmax(dim=1)  # (B,)
                    batch_cams = []

                    input_size = batch_images.shape[-2:]  # get (H, W) from input

                    for i in range(batch_images.size(0)):
                        fmap = feature_maps[i]  # (C, H, W)
                        cls_idx = pred_classes[i]
                        weight_vec = weights[cls_idx].view(-1, 1, 1)  # (C, 1, 1)

                        cam = torch.sum(fmap * weight_vec, dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)
                        cam = F.relu(cam)
                        cam = cam - cam.min()
                        cam = cam / (cam.max() + 1e-8)
                        cam_resized = F.interpolate(cam, size=input_size, mode='bilinear', align_corners=False)
                        batch_cams.append(cam_resized)

                    tensor_cams = torch.cat(batch_cams, dim=0)  # (B, 1, H, W)
            else:
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
        threshold: int=0
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

            total_batches = len(dataloader_new)
            for images, masks, masks_gt in dataloader_new:
                batch_count += 1
                # if batch_count % 10 == 0:
                    # print(f"Batch: {batch_count}/{total_batches}")
                images = images.to(device)
                masks =(masks >= threshold)*masks.to(device).float()
                # masks_bin = (masks > threshold).float()
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

                '''checking batch by visualization'''
                # if batch_count == 80:  # only visualize one batch per epoch
                #     images_vis = images[:4].detach().cpu()
                #     masks_vis = masks[:4].detach().cpu()
                #     preds_vis = probs[:4].float().detach().cpu()
                #
                #     for i in range(images_vis.size(0)):
                #         fig, axs = plt.subplots(1, 3, figsize=(10, 3))
                #
                #         img = images_vis[i].permute(1, 2, 0).numpy()
                #         img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1] for visualization
                #
                #         axs[0].imshow(img)
                #         axs[0].set_title("Input Image")
                #         axs[0].axis("off")
                #
                #         axs[1].imshow(masks_vis[i].squeeze(), cmap='gray')
                #         axs[1].set_title("Ground Truth Mask")
                #         axs[1].axis("off")
                #
                #         axs[2].imshow(preds_vis[i].squeeze(), cmap='gray')
                #         axs[2].set_title("Predicted Mask")
                #         axs[2].axis("off")
                #
                #         plt.tight_layout()
                #         plt.show()

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
    def fit_sgd_seed(
        model_train,
        dataloader_new,
        number_epoch: int,
        learning_rate: float,
        loss_function: torch.nn.Module,
        model_path: str,
        device: str = None,
        threshold:int=0.1
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

        firsttime = True
        for epoch in range(number_epoch):
            batch_count = 0
            correct_pixels = 0
            total_pixels = 0
            total_loss = 0

            total_batches = len(dataloader_new)
            for images, masks, masks_gt in dataloader_new:
                batch_count += 1
                # if batch_count % 10 == 0:
                    # print(f"Batch: {batch_count}/{total_batches}")
                images = images.to(device)
                masks = masks.to(device).float()
                masks_flat = masks.view(-1)
                # Compute statsZZZ
                # threshold_bg = 0.1
                # threshold_fg = 0.7
                threshold_bg = torch.quantile(masks_flat, threshold).item()
                threshold_fg = torch.quantile(masks_flat, (1-threshold)).item()

                if firsttime==True:
                    min_val = masks_flat.min().item()
                    max_val = masks_flat.max().item()
                    print(f"masks stats → min: {min_val:.4f}, max: {max_val:.4f}, bg_thres: {threshold_bg:.4f}, fg_thres: {threshold_fg:.4f}")
                    num_fg_seeds = (masks >= threshold_fg).sum().item()/(64*64*64)
                    print(f"Foreground seeds (masks >= {threshold_fg}): {num_fg_seeds}")
                    num_bg_seeds = (masks <= threshold_bg).sum().item()/(64*64*64)
                    print(f"Background seeds (masks <= {threshold_bg}): {num_bg_seeds}")
                    firsttime=False
                seed_mask = ((masks >= threshold_fg) | (masks <= threshold_bg)).float()  # [B, H, W]
                masks_bin = torch.zeros_like(masks)
                masks_bin[masks >= threshold_fg] = 1  # Foreground
                masks_bin[masks <= threshold_bg] = 0  # Background
                # masks_gt = masks_gt.to(device).float()
                optimizer.zero_grad()
                outputs = model_train(images)  # [B, num_classes, H, W]

                loss_raw = loss_function(outputs, masks_bin)
                loss = (loss_raw * seed_mask).sum() / (seed_mask.sum() + 1e-6)


                loss.backward()
                optimizer.step()

                # Calculate pixel-wise accuracy
                probs = torch.sigmoid(outputs)
                # probs= outputs
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
    def predict_segmentation_dataset_with_basicfilter(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cpu",
        threshold: float = 0,
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
            for images, targets,gt_masks in dataloader:
                images = images.to(device)
                logits = model(images)  # [B, 1, H, W]
                probs = torch.sigmoid(logits)  # ∈ [0,1]
                filtered_probs = (
                    probs * (probs >= threshold).float()
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
    def predict_segmentation_dataset_with_grabcut(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cpu",
        threshold: float = 0.1,
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
        mask_list = []
        gt_mask_list = []

        with torch.no_grad():
            sample_count = 0
            for images, targets,gt_masks in dataloader:
                images = images.to(device)
                logits = model(images)  # [B, 1, H, W]
                probs = torch.sigmoid(logits)  # [B, 1, H, W]
                batch_size = images.size(0)
                for i in range(batch_size):
                    sample_count+=1
                    if sample_count%1000==0:
                        print(sample_count)
                    # Convert single image to NumPy (unnormalized if needed)
                    img_np = images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, C], float32
                    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)  # [H, W, C], uint8

                    # Convert prob to NumPy mask
                    prob_np = probs[i].squeeze().cpu().numpy()  # [H, W]
                    # low_thresh = 0.3
                    # high_thresh = 0.7
                    low_thresh = np.percentile(prob_np, threshold*100)
                    high_thresh = np.percentile(prob_np, (1-threshold)*100)
                    # print(low_thresh)
                    # print(high_thresh)
                    # Initialize with 2 (probable background)
                    grabcut_mask = np.full_like(prob_np, 3, dtype=np.uint8)

                    # Assign confident FG and BG based on thresholds
                    grabcut_mask[prob_np >= high_thresh] = 1  # definite foreground
                    grabcut_mask[prob_np <= low_thresh] = 0  # definite background

                    # print(grabcut_mask)

                    refined_mask = SelfTraining.grabcut_from_mask(img_np, grabcut_mask)
                    refined_mask= refined_mask.astype(np.float32)  # 0→-0.25, 1→+0.25
                    combined_map = prob_np*0.3 + refined_mask*0.7
                    # combined_map_clipped = np.clip(combined_map, 0.0, 1.0)
                    refined_mask_tensor = torch.from_numpy(combined_map).unsqueeze(0).float()
                    image_list.append(images[i].cpu())
                    mask_list.append(refined_mask_tensor.squeeze(0))  # ensure [H, W]
                    gt_mask_list.append(gt_masks[i].squeeze(0).cpu())  # ensure [H, W]

                    # SelfTraining.visualize_grabcut(img_np, grabcut_mask.squeeze(), refined_mask)
        # print(len(image_list))
        # print(len(binary_mask_list))
        # print(len(gt_mask_list))
        #
        # for i, (img, mask, gt) in enumerate(zip(image_list, binary_mask_list, gt_mask_list)):
        #     if img.shape != image_list[0].shape:
        #         print(f"❌ Image shape mismatch at index {i}: got {img.shape}, expected {image_list[0].shape}")
        #     if mask.shape != binary_mask_list[0].shape:
        #         print(
        #             f"❌ Binary mask shape mismatch at index {i}: got {mask.shape}, expected {binary_mask_list[0].shape}")
        #     if gt.shape != gt_mask_list[0].shape:
        #         print(f"❌ Ground truth shape mismatch at index {i}: got {gt.shape}, expected {gt_mask_list[0].shape}")

        all_images = torch.stack(image_list)  # [N, 3, H, W]
        all_masks = torch.stack([mask.unsqueeze(0) for mask in mask_list])  # [N, 1, H, W]
        all_gts = torch.stack([gt.unsqueeze(0) for gt in gt_mask_list])  # [N, 1, H, W]

        # print(f"Image batch size: {all_images.shape}")
        # print(f"Binary mask batch size: {all_masks.shape}")
        # print(f"GT mask batch size: {all_gts.shape}")

        print(f"Generated filtered probability masks for {len(all_images)} samples.")
        return TensorDataset(all_images, all_masks, all_gts)

    @staticmethod
    def predict_segmentation_dataset_with_mixlabel(
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

        with (torch.no_grad()):
            for images, probs,gt_masks in dataloader:
                images = images.to(device)
                logits = model(images)  # [B, 1, H, W]
                # new_probs = torch.sigmoid(logits)  # ∈ [0,1]
                new_probs =logits


                # Threshold percentage
                threshold_percentage = threshold  # Set your desired percentage
                num_pixels = new_probs.numel()  # Total number of pixels in the batch

                # Flatten the tensor and sort
                flattened_probs = new_probs.view(-1)
                sorted_probs, _ = torch.sort(flattened_probs, descending=False)
                # print("sort",sorted_probs[0],sorted_probs[-1],"len",len(sorted_probs))
                # print("num",num_pixels)

                # Get the value of the pixel at the threshold percentage
                # threshold_value_low = sorted_probs[int(num_pixels * threshold_percentage)]
                # threshold_value_high = sorted_probs[int(num_pixels * (1-threshold_percentage))]

                threshold_value_low = 0.1
                threshold_value_high = 0.5

                # Apply the threshold: set all pixels below the threshold to 0
                add_probs = (new_probs >= threshold_value_high).float() * 1.0
                filtered_probs = torch.maximum(
                    add_probs,
                    probs
                ).float()
                filtered_probs = (new_probs >= threshold_value_low).float() * filtered_probs

                image_list.append(images.cpu())
                prob_mask_list.append(filtered_probs.cpu())
                gt_mask_list.append(gt_masks.cpu())

        all_images = torch.cat(image_list, dim=0)
        all_probs = torch.cat(prob_mask_list, dim=0)
        all_gts = torch.cat(gt_mask_list, dim=0)

        print(f"Generated filtered probability masks for {len(all_images)} samples.")
        return TensorDataset(all_images, all_probs, all_gts)

    @staticmethod
    def visualize_predicted_masks(dataset, num_samples=8, save_path=None):
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

    @staticmethod
    def grabcut_from_mask(image, init_mask, iter_count=5):
        """
        Applies GrabCut using an initial seed mask.
        Returns binary mask after GrabCut, or fallback based on confident seeds.
        """
        num_fg = np.sum(init_mask == cv2.GC_FGD)
        num_bg = np.sum(init_mask == cv2.GC_BGD)

        has_fg = num_fg > 0
        has_bg = num_bg > 0

        if not has_fg or not has_bg:
            # print("⚠️ GrabCut skipped (no confident FG/BG seeds) → using fallback init_mask.")
            # # Convert: 1 (FG) and 3 (probable FG) → 1, everything else → 0
            return ((init_mask == cv2.GC_FGD) | (init_mask == cv2.GC_PR_FGD)).astype(np.uint8)

        # Proceed with GrabCut if valid seeds exist
        mask = init_mask.copy().astype('uint8')
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(image, mask, None, bgdModel, fgdModel, iter_count, mode=cv2.GC_INIT_WITH_MASK)

        binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
        return binary_mask
    @staticmethod
    def visualize_grabcut(image, init_mask, binary_mask, title_prefix="Sample"):
        """
        Visualize image, seed mask, and result from GrabCut.

        Args:
        - image: np.ndarray [H, W, 3], uint8
        - init_mask: np.ndarray [H, W], GrabCut labels {0,1,2,3}
        - binary_mask: np.ndarray [H, W], final output mask {0,1}
        """
        # Colormap for init_mask
        grabcut_vis = np.zeros_like(image)
        grabcut_vis[init_mask == 0] = [0, 0, 255]  # definite background - red
        grabcut_vis[init_mask == 1] = [0, 255, 0]  # definite foreground - green
        grabcut_vis[init_mask == 2] = [255, 0, 0]  # probable background - blue
        grabcut_vis[init_mask == 3] = [255, 255, 0]  # probable foreground - yellow

        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"{title_prefix}: Original Image")
        axs[0].axis("off")

        axs[1].imshow(grabcut_vis)
        axs[1].set_title(f"{title_prefix}: Init Mask (GrabCut Labels)")
        axs[1].axis("off")

        axs[2].imshow(binary_mask, cmap="gray")
        axs[2].set_title(f"{title_prefix}: Final Mask (Binary)")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()