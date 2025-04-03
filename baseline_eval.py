import torch
import baseline_model
from baseline_model_loader import test_loader

model = baseline_model.UNetWrapper(
    in_channels=3,  # 3 channels for RGB images
    n_classes=3,  # 3 class for segmentation - foreground, background, unknown
    depth=3,  # 3 encoding layers, 2 decoding layers
    wf=5,  # 2^5 = 32 channels
    padding=True,  # equivalent to padding=1 in Bruce's implementation
    batch_norm=True,  # use batch normalization after layers with activation function
    up_mode="upconv",
)
model.load_state_dict(
    torch.load(
        "baseline_model.pth",
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
)
model.eval()
with torch.no_grad():
    total_correct = 0
    total_pixels = 0
    for batch in test_loader:
        images, targets = batch
        images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        targets = targets.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == targets).sum().item()
        total_pixels += targets.numel()
    accuracy = total_correct / total_pixels
    print(f"Accuracy: {accuracy:.4f}")
