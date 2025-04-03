from custom_data import OxfordPetDataset
import custom_loader
import os
from CAM.cam_model import ResNetBackbone, CNN, fit_sgd, visualize_cam
import torch
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# Hyperparameters
classification_mode = "breed"
batch_size = 32
model_type = "Res" # CNN, Res
model_dir = os.path.join("checkpoints", "CAM")
train_mode = False  # if False, will use trained local mode
num_epochs = 50

loss_function = torch.nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
minibatch_size = 32

# Data loading
try:
    dataset = OxfordPetDataset().prepare_dataset()
    train_loader, val_loader, test_loader = custom_loader.create_dataloaders(dataset, target_type=classification_mode)
    print("Data loaders created successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Model Init
model_mode = classification_mode  # species or breed
model_name = f"{model_type}_{model_mode}"
print(f"Using model {model_name}")

model_path = os.path.join(model_dir, model_name)
num_out = 37 if model_mode == "breed" else 2

# Train model
if train_mode:
    try:
        model_train = ResNetBackbone(num_classes=num_out, pretrained=True) if "Res" in model_name else CNN(out_channels=256, num_classes=num_out)
        fit_sgd(model_train, train_loader, classification_mode, num_epochs, 3e-4, batch_size, loss_function, model_path, device=device)
        print(f"Training completed for {model_name}")
    except Exception as e:
        print(f"Error during training: {e}")
        exit(1)

# Test model
print(f"TESTING MODEL {model_path}.pt")
try:
    model_test = ResNetBackbone(num_classes=num_out, pretrained=False) if "Res" in model_name else CNN(out_channels=256, num_classes=num_out)
    model_test.load_state_dict(torch.load(f"{model_path}.pt"))
    model_test.to(device)
    model_test.eval()
    print("Model loaded and set to evaluation mode.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def visualize_gradcam(model, dataloader, model_mode, device, num_images=4):
    """
    Visualise Grad-CAM for a batch of images.
    """
    try:
        # Get a batch of images
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
        images = images[:num_images].to(device)
        labels = labels[:num_images].to(device)

        # target layers for Grad-CAM
        if isinstance(model, ResNetBackbone):
            target_layers = [model.features[-1][-1]]  # last block of layer4 in ResNet
        elif isinstance(model, CNN):
            target_layers = [model.features[-2]]  # last conv layer before GAP in CNN
        else:
            raise ValueError("Unsupported model type for Grad-CAM")

        # Initialise Grad-CAM
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(label.item()) for label in labels]

        # Compute Grad-CAM
        grayscale_cams = cam(input_tensor=images, targets=targets)
        print(f"Grad-CAM computed for {num_images} images.")

        # Visualise
        _, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(4 * num_images, 8))
        for i in range(num_images):
            # Original image
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)  # Ensure range [0, 1]
            axes[0][i].imshow(img)
            axes[0][i].set_title(f"Original - Class {labels[i].item()}")
            axes[0][i].axis('off')

            # Grad-CAM overlay
            grayscale_cam = grayscale_cams[i, :]
            visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            axes[1][i].imshow(visualization)
            axes[1][i].set_title(f"Grad-CAM - Class {labels[i].item()}")
            axes[1][i].axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error in Grad-CAM visualization: {e}")
        print("Debug info:")
        print(f"Input tensor shape: {images.shape if 'images' in locals() else 'Not available'}")
        print(f"Target layers: {target_layers if 'target_layers' in locals() else 'Not defined'}")
        raise

# visualisations
print("Visualizing standard CAM...")
visualize_cam(model_test, train_loader, model_mode, device)

print("Visualizing Grad-CAM...")
visualize_gradcam(model_test, train_loader, model_mode, device)