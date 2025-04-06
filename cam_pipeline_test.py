import os
import torch

from CAM.cam_model import CNN, ResNetBackbone, fit_sgd, visualize_cam
from data import OxfordPetDataset, create_dataloaders
from models import CAMManager

# Hyperparameters
classification_mode = "breed"
batch_size = 32
model_type = "Res"  # CNN, Res
model_dir = os.path.join("checkpoints", "CAM")
train_mode = False  # if False, will use trained local mode
num_epochs = 50

loss_function = torch.nn.CrossEntropyLoss()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
minibatch_size = 32

# Data loading
try:
    dataset = OxfordPetDataset().prepare_dataset()
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=batch_size, target_type=classification_mode
    )
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
        model_train = (
            ResNetBackbone(num_classes=num_out, pretrained=True)
            if "Res" in model_name
            else CNN(out_channels=256, num_classes=num_out)
        )
        fit_sgd(
            model_train,
            train_loader,
            classification_mode,
            num_epochs,
            3e-4,
            batch_size,
            loss_function,
            model_path,
            device=device,
        )
        print(f"Training completed for {model_name}")
    except Exception as e:
        print(f"Error during training: {e}")
        exit(1)

# Test model
print(f"TESTING MODEL {model_path}.pt")
try:
    model_test = (
        ResNetBackbone(num_classes=num_out, pretrained=False)
        if "Res" in model_name
        else CNN(out_channels=256, num_classes=num_out)
    )
    model_test.load_state_dict(torch.load(f"{model_path}.pt"))
    model_test.to(device)
    model_test.eval()
    print("Model loaded and set to evaluation mode.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)


def visualize_with_cam_manager(
    model, dataloader, model_mode, device, cam_method="GradCAM", num_images=4
):
    """
    Visualise CAMs using the CAMManager class.
    """
    try:
        # Get a batch of images
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
        images = images[:num_images].to(device)
        labels = labels[:num_images].to(device)

        # Initialise CAM Manager and visualise CAM
        cam_manager = CAMManager(model, method=cam_method)
        cam_manager.visualize_batch(images, labels, num_images)

    except Exception as e:
        print(f"Error in CAM visualization: {e}")
        raise


# visualisations
print("Visualizing standard CAM...")
visualize_cam(model_test, train_loader, model_mode, device)

print("Visualizing CAM with manager...")
visualize_with_cam_manager(
    model_test, train_loader, model_mode, device, cam_method="GradCAM"
)

# generate cam dataset for self-training
print("Generating CAM dataset for self-training...")
cam_manager = CAMManager(model_test, method='GradCAM')
cam_dataset = cam_manager.generate_cam_dataset(train_loader)
print(f"Generated CAM dataset with {len(cam_dataset)} samples")
