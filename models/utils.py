import torch

from models.backbones import CNNBackbone, ResNetBackbone
from models.heads import PretrainHead


def test_models(device: torch.device):
    print("\nTesting models...\n")
    cnn_backbone = CNNBackbone().to(device)
    res_backbone = ResNetBackbone().to(device)
    cnn_bbox_head = PretrainHead(adapter="cnn", output_type="bbox").to(device)
    res_bbox_head = PretrainHead(adapter="res18", output_type="bbox").to(device)
    res_class_head = PretrainHead(adapter="res18", output_type="breed").to(device)
    res_class_head2 = PretrainHead(adapter="res50", output_type="species").to(device)
    cnn_class_head = PretrainHead(adapter="cnn", output_type="is_animal").to(device)

    # Print the number of parameters in the CNN backbone
    cnn_params = sum(p.numel() for p in cnn_backbone.parameters())
    print(f"CNN Backbone parameters: {cnn_params:,}")

    # Print the number of parameters in the ResNet backbone
    res_params = sum(p.numel() for p in res_backbone.parameters())
    print(f"ResNet Backbone parameters: {res_params:,}")