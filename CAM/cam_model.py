
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
from torch.utils.data import TensorDataset, DataLoader
'''One additional Library OpenCV has been used in this file for processing images with CAM. However it's possible to remove this library if necessary'''

'''I only ran this code on CPU, not sure if that's compatible with GPU'''
from utils import resize_images



# to show the images and labels of a batch
def show_batch(images, species, breeds, rows=4):
    cols = len(images) // rows + int(len(images) % rows != 0)
    plt.figure(figsize=(3 * cols, 3 * rows))

    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"{species[i]} - {breeds[i]}", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# define a CNN model
class CNN(nn.Module):  # define the model
    def __init__(self, out_channels, num_classes):
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

            nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, padding=1),  # (B, out_channels, 32, 32)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Classifier head after GAP
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, img, return_features=False):
        x = self.features(img)
        if return_features:
            feature_map = x  # shape: (B, C, H, W), save the feature_map for the use of CAM
        x = F.adaptive_avg_pool2d(x, 1) # This is the Global Average Pooling (GAP) that's necessary for CAM. Which transfer a whole channel into 1 single value
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if return_features:
            return x, feature_map  # logits, conv feature map
        else:
            return x

# define a ResNet model
class ResNetBackbone(nn.Module):
    def __init__(self, num_classes,pretrained):
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

        self.in_features = base_model.fc.in_features  # 512
        self.classifier = nn.Linear(self.in_features, num_classes)

    def forward(self, img, return_features=False):
        x = self.features(img)
        if return_features:
            feature_map = x

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if return_features:
            return x, feature_map
        else:
            return x

def fit_sgd(model_train: torch.nn.Module, 
            trainloader: torch.utils.data.DataLoader, 
            label_select: str, 
            number_epoch: int, 
            learning_rate: float,
            batch_size : int,
            loss_function: torch.nn.Module, 
            model_path: str, 
            device: str = None) -> None:
    """
    Train a model using Stochastic Gradient Descent.
    
    Args:
        model_train: The neural network model to train
        trainloader: DataLoader containing the training data
        label_select: Type of label to use for training (e.g., 'breed', 'species')
        number_epoch: Number of epochs to train for
        learning_rate: Learning rate for the optimizer
        loss_function: Loss function to optimize
        model_path: Path where the trained model will be saved
        device: Device to run the training on ('cuda', 'cpu', etc.)
    """
    print("<Training Start>")
    model_train.to(device)
    optimizer = optim.SGD(model_train.parameters(), lr=learning_rate,
                          momentum=0.9)
    batch_count = 0
    for epoch in range(number_epoch):
        correct_count = 0
        sample_count = 0
        loss_sum = 0
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            batch_count += 1
            # images, breeds = images.to(device), breeds.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model_train(images) 
            loss = loss_function(outputs,labels) # I used breeds as label. Potentially can switch to species
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs[:batch_size], 1)  # Get the index of the max logit (prediction)
            
            batch_correct_count = (predicted == labels).sum().item()
            
            # Count correct predictions
            correct_count += batch_correct_count
            sample_count += batch_size
            running_loss = loss.item() * batch_size
            loss_sum += running_loss

        print(
            f"Epoch:{epoch + 1}/{number_epoch},Accuracy: {correct_count / sample_count},{correct_count}/{sample_count},Loss:{loss_sum / sample_count}")

    # save trained model
    torch.save(model_train.state_dict(), f"{model_path}")
    print("Model saved. Number of parameters:", sum(a.numel() for a in model_train.parameters()))

def compute_cam(feature_map, classifier_weights, class_idx):
    # feature_map: (C, H, W)
    # classifier_weights: (num_classes, C)
    weights = classifier_weights[class_idx].unsqueeze(1).unsqueeze(2)  # Shape: (C, 1, 1)
    cam = torch.sum(weights * feature_map, dim=0)  # Shape: (H, W)
    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)  # ReLU
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)  # Normalize to [0, 1]
    return cam

def show_cam_on_image(img_tensor, cam, title='CAM'):
    # img_tensor: (3, H, W), range [0,1]
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32) / 255
    overlay = 0.5 * img_np + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis('off')
    plt.show()

def unnormalize(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy().transpose((1, 2, 0))  # CxHxW -> HxWxC
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def visualize_cam(model, dataset, label_select, device=None):
    # Get a batch of test images
    data_iter = iter(dataset)
    images, labels = next(data_iter)
    images = images[:4].to(device)
    labels = labels[:4]  # Ensure labels match the 4 images

    # Create the figure for 2 rows x 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    model.eval()  # Just in case
    with torch.no_grad():
        logits, feature_maps = model(images, return_features=True)

        for i in range(4):
            img = images[i]
            fmap = feature_maps[i]
            target_class = labels[i].item()
            
            # Compute CAM
            cam = compute_cam(fmap, model.classifier.weight.data, target_class)
            img_np = unnormalize(img)  # Convert from tensor -> numpy [0,1] image

            # Resize CAM to match image
            cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
            cam_resized = np.clip(cam_resized, 0, 1)

            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = heatmap.astype(np.float32) / 255.0

            overlay = 0.5 * img_np + 0.5 * heatmap
            overlay = np.clip(overlay, 0, 1)

            # Plot original
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f"Original - Class {target_class}")
            axes[0, i].axis('off')

            # Plot CAM
            axes[1, i].imshow(overlay)
            axes[1, i].set_title(f"CAM - Class {target_class}")
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def get_labels_from_cam(model, image_batch, device="cpu"):
    """
    Generate pixel-level label maps using Class Activation Maps (CAM) for a batch of images.

    Args:
        model: Trained CNN or ResNetBackbone model
        image_batch: Tensor of shape (B, C, H, W)
        device: 'cpu' or 'cuda'

    Returns:
        cam_masks: List of 2D numpy arrays (one per image), resized to match image dimensions
        predicted_classes: List of predicted class indices for each image
    """
    model.eval()
    model.to(device)
    image_batch = image_batch.to(device)

    cam_masks = []
    predicted_classes = []

    with torch.no_grad():
        logits, feature_maps = model(image_batch, return_features=True)
        weights = model.classifier.weight.data  # shape: (num_classes, channels)

        for i in range(image_batch.size(0)):
            feature_map = feature_maps[i]  # shape: (C, H, W)
            _, pred_class = torch.max(logits[i], dim=0)
            predicted_classes.append(pred_class.item())

            class_weights = weights[pred_class].unsqueeze(1).unsqueeze(2)  # shape: (C, 1, 1)
            cam = torch.sum(class_weights * feature_map, dim=0)  # shape: (H, W)
            cam = cam.detach().cpu().numpy()
            cam = np.maximum(cam, 0)  # Apply ReLU
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)  # Normalize to [0, 1]

            # Resize to match image dimensions (assumed 256x256, modify if needed)
            cam_resized = cv2.resize(cam, (image_batch.shape[3], image_batch.shape[2]))
            cam_masks.append(cam_resized)

    return cam_masks, predicted_classes

def generate_cam_label_dataset(model, dataloader, device="cpu"):
    """
    Generate a new dataset using get_labels_from_cam, where input is the original image
    and label is the CAM (pixel-level pseudo label).

    Args:
        model (nn.Module): Trained CNN or ResNetBackbone model.
        dataloader (DataLoader): DataLoader yielding (images, labels) batches.
        device (str): 'cuda' or 'cpu'.

    Returns:
        TensorDataset: New dataset (image, CAM_mask) pairs.
    """
    all_inputs = []
    all_cam_masks = []

    model.eval()
    model.to(device)

    for image_batch, _ in dataloader:
        image_batch = image_batch.to(device)
        cam_masks, _ = get_labels_from_cam(model, image_batch, device)

        for i in range(len(cam_masks)):
            image_tensor = image_batch[i].cpu()
            cam_tensor = torch.tensor(cam_masks[i], dtype=torch.float32).unsqueeze(0)  # (1, H, W)

            all_inputs.append(image_tensor)
            all_cam_masks.append(cam_tensor)

    input_tensor = torch.stack(all_inputs)  # (N, 3, H, W)
    cam_tensor = torch.stack(all_cam_masks)  # (N, 1, H, W)

    print(f"Created CAM-labeled dataset with {len(input_tensor)} samples.")
    return TensorDataset(input_tensor, cam_tensor)




# if __name__ == "__main__":
#
#     from loader import H5ImageLoader
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     minibatch_size = 32
#     os.environ["CUDA_VISIBLE_DEVICES"]="0"
#     DATA_PATH = '../../../DLcourse/GroupTask/segmentation/data'
#
#     ## data loader
#     loader_train = H5ImageLoader(DATA_PATH+'/images_train.h5', minibatch_size, DATA_PATH+'/labels_train.h5')
#     loader_test = H5ImageLoader(DATA_PATH+'/images_val.h5', 20, DATA_PATH+'/labels_val.h5')
#     dataloader_train=iter(loader_train)
#     dataloader_test=iter(loader_test)
#     images, labels,sp,br = next(dataloader_train)
#     print(dataloader_train)
#     print(images[0])
#     print(labels[0])
#     print(sp[0])
#     print(br[0])
#
#     batch_size=32
#     loss_function = torch.nn.CrossEntropyLoss()
#     train_mode=False # if False, will use trained local model.
#     model_use="CNN_breed" # choose the model used
#
#
#     # try 4 types of models with CNN and ResNET and with breed and species as labels
#     if model_use=="Res_breed":
#         if train_mode == True:
#             model_train=ResNetBackbone(num_classes=37,pretrained=True)
#             fit_sgd(model_train, dataloader_train,"breed", 20, 0.01,
#                     "../../../DLcourse/GroupTask/segmentation/Res_breed.pt")
#         model_test = ResNetBackbone(num_classes=37,pretrained=False)
#         model_test.load_state_dict(torch.load(f"{model_use}.pt"))  # Load model from local
#         model_test.to(device)
#         model_test.eval()
#         visualize_cam(model_test,loader_train,"breed") # Use loaded model on training dataset to show CAM
#
#     elif model_use=="Res_species":
#         if train_mode == True:
#             model_train=ResNetBackbone(num_classes=2,pretrained=True)
#             fit_sgd(model_train, dataloader_train,"species", 20, 0.01,
#                     "../../../DLcourse/GroupTask/segmentation/Res_species.pt")
#         model_test = ResNetBackbone(num_classes=2,pretrained=False)
#         model_test.load_state_dict(torch.load(f"{model_use}.pt"))  # Load model from local
#         model_test.to(device)
#         model_test.eval()
#         visualize_cam(model_test,loader_train,"species") # Use loaded model on training dataset to show CAM
#
#     elif model_use=="CNN_breed":
#         if train_mode == True:
#             model_train=CNN(out_channels=256,num_classes=37)
#             fit_sgd(model_train, dataloader_train, "breed", 50, 0.01, "CNN_breed.pt")
#         model_test =CNN(out_channels=256,num_classes=37)
#         model_test.load_state_dict(torch.load(f"{model_use}.pt"))  # Load model from local
#         model_test.to(device)
#         model_test.eval()
#         visualize_cam(model_test,loader_train,"breed") # Use loaded model on training dataset to show CAM
#
#     elif model_use == "CNN_species":
#         if train_mode == False:
#             model_train = CNN(out_channels=256, num_classes=2)
#             fit_sgd(model_train, dataloader_train,"species", 50, 0.01, "CNN_species.pt")
#         model_test = CNN(out_channels=256, num_classes=2)
#         model_test.load_state_dict(torch.load(f"{model_use}.pt"))  # Load model from local
#         model_test.to(device)
#         model_test.eval()
#         visualize_cam(model_test,loader_train,"species") # Use loaded model on training dataset to show CAM
#
#
