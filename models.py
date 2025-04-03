from typing import Literal
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import os  
from utils import resize_images 

### Num Inputs:
#       Breed:                  37
#       Species:                2
#       Bbox:                   256x256       
#       Breed + Species??:      39
#       Breed + Bbox:           37 + 256x256
#       Species + Bbox:         2 + 256x256
#       Breed+Species+Bbox?:    39 + 256x256

# Backbone:
#   CNN
#   ResNet
# Head:
#   Bbox Head
#   Classifier Head


class BasicCAMWrapper(nn.Module):
    cam_type: Literal["CAM", "GradCAM", "ScoreCAM"] = "CAM"

    def __init__(self, model, cam_type="CAM"):
        """
        Wrapper class for models to generate Class Activation Maps (CAM) during forward pass
        
        Args:
            model: The base model to wrap (CNN or ResNetBackbone)
        """
        super().__init__()
        self.model = model
        self.hooks = []
        self.cam_type = cam_type
        
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
            match self.cam_type:
                case "CAM":
                    cam = self._compute_cam(feature_map, self.model.classifier.weight.data, current_target)
                case "ScoreCAM":
                    cam = self._compute_scorecam(self.model, feature_map, x[b], current_target)
                case _:
                    raise ValueError(f"Unsupported CAM type: {self.cam_type}")
            cam = resize_images(cam, 256, 256)
            cams.append(torch.tensor(cam))
        
        # Stack all CAMs into a batch
        batch_cams = torch.stack(cams)
        
        return batch_cams
    
    def _compute_cam(self, feature_map, classifier_weights, class_idx):
        """Compute class activation map"""
        weights = classifier_weights[class_idx].unsqueeze(1).unsqueeze(2)  # Shape: (C, 1, 1)
        cam = torch.sum(weights * feature_map, dim=0)  # Shape: (H, W)
        cam = cam.detach().cpu().numpy()
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)  # Normalize to [0, 1]
        return cam
    
    def _compute_scorecam(self, model, feature_map, input_image, target_class_idx):
        """Compute Score-CAM"""

        # Normalize activation maps to [0,1] range - maintains shape of [C, H, W]
        normalized_map = (feature_map - feature_map.min()) / (
            feature_map.max() - feature_map.min()
        )

        # Resize to the same size as the input image using bilinear interpolation
        normalized_map = F.interpolate(
            normalized_map.unsqueeze(1),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )

        # Multiply input image by each activation map (element-wise masking)
        masked_inputs = input_image.unsqueeze(
            0
        )  # Shape: [C, C, H, W]; each image has multiple masked versions

        # Forward pass all masked inputs through the model
        scores = F.softmax(model(masked_inputs), dim=1)[:, target_class_idx]  # Shape: [C]

        # Use these scores as weights for the activation maps
        weights = scores.clone().detach().reshape(1, -1, 1, 1)  # Shape: [1, C, 1, 1]
        cam = torch.sum(weights * feature_map, dim=1).squeeze(0)  # Weighted sum
        # Normalize to [0,1]
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.detach().cpu().numpy()

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
        fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(4*num_images, 8))
        
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
                cam = self._compute_cam(fmap, self.model.classifier.weight.data, target_class)
                img_np = img.permute(1, 2, 0).cpu().numpy()
                
                # Original image on first row
                axes[0][i].imshow(img_np)
                axes[0][i].set_title(f"Original - Class {target_class}")
                axes[0][i].axis('off')
                
                # CAM overlay on second row
                cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
                cam_normalized = np.clip(cam_resized, 0, 1)
                
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_normalized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                heatmap = heatmap.astype(np.float32) / 255.0
                
                overlay = 0.5 * img_np + 0.5 * heatmap
                overlay = np.clip(overlay, 0, 1)
                
                axes[1][i].imshow(overlay)
                axes[1][i].set_title(f"CAM - Class {target_class}")
                axes[1][i].axis('off')
        
        plt.tight_layout()
        plt.show()

class BboxHead(nn.Module):
    def __init__(self, adapter = "CNN"):
        super().__init__()
        num_inputs = 256 if adapter.lower() == "cnn" else 512 
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (C,H,W) → (C,1,1)
            nn.Flatten(),                  # → (C,)
            nn.Linear(num_inputs, 4),
            nn.Sigmoid() ### [cx, cy, w, h]
        )
    
    def forward(self, z):
        return self.head(z)

class ClassifierHead(nn.Module):
    def __init__(self, num_classes = 2, adapter = "CNN"):
        super().__init__()
        num_inputs = 256 if adapter.lower() == "cnn" else 512 
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (C,H,W) → (C,1,1)
            nn.Flatten(),                  # → (C,)
            nn.Linear(num_inputs, num_classes),
            nn.Sigmoid() 
        )
    
    def forward(self, z):
        return self.head(z)

class CNNBackbone(nn.Module): 
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Input: (B, 3, 256, 256)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # (B, 32, 256, 256)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 32, 128, 128)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # (B, 64, 128, 128)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 64, 64, 64)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # (B, 128, 64, 64)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 128, 32, 32)

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # (B, 256, 32, 32)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (B, 256, 16, 16)
        )

    def forward(self, img):
        return self.features(img)
        
class ResNetBackbone(nn.Module):
    def __init__(self,pretrained : bool = True):
        super().__init__()
        if pretrained:
            base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) # With pretrained weights
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
            base_model.layer4
        )

    def forward(self, img):
        return self.features(img)