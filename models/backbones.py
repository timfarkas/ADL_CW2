from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    resnet18,
    resnet50,
    resnet101,
)

from custom_types import ResNetAdapter

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
    def __init__(self, pretrained: bool = True, model_type: ResNetAdapter = "res18"):
        super().__init__()

        if model_type == "res18":
            if pretrained:
                base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                base_model = resnet18(weights=None)
        elif model_type == "res50":
            if pretrained:
                base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                base_model = resnet50(weights=None)
        elif model_type == "res101":
            if pretrained:
                base_model = resnet101(weights=ResNet101_Weights.DEFAULT)
            else:
                base_model = resnet101(weights=None)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

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
