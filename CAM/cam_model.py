
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


# defin a CNN model
class CNN(nn.Module):  # define the model
    def __init__(self, out_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, padding=1),
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

class CAMWrapper(nn.Module):
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
        weights = classifier_weights[class_idx].unsqueeze(1).unsqueeze(2)  # Shape: (C, 1, 1)
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


def fit_sgd(model_train, trainloader, label_select, number_epoch,learning_rate,model_path):  # training model
    print("<Training Start>")
    model_train.to(device)
    optimizer = optim.SGD(model_train.parameters(), lr=learning_rate,
                          momentum=0.9)
    batch_count = 0
    for epoch in range(number_epoch):
        correct_count = 0
        sample_count = 0
        loss_sum = 0
        for images,labels,species,breeds in trainloader:
            batch_count += 1
            # images, breeds = images.to(device), breeds.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model_train(images)
            if label_select=="breed":
                loss = loss_function(outputs,breeds) # I used breeds as label. Potentially can switch to species
            elif label_select=="species":
                loss = loss_function(outputs,species) # I used breeds as label. Potentially can switch to species
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs[:batch_size], 1)  # Get the index of the max logit (prediction)
            if label_select=="breed":
                batch_correct_count = (predicted == breeds).sum().item()
            elif label_select=="species":
                batch_correct_count = (predicted == species).sum().item()
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

def visualize_cam(model,dataset, label_select, device = None):
    # Get a batch of test images
    data_iter = iter(dataset)
    images, labels = next(data_iter)  # images: (B, 3, H, W)
    images = images[:4].to(device)  # Take first 8 images

    # Prepare figure
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))  # 2 rows, 4 columns

    with torch.no_grad():
        logits, feature_maps = model(images, return_features=True)

        for i in range(4):  # only plot 4 images
            img = images[i]  # (3, H, W)
            fmap = feature_maps[i]  # (C, H, W)
            
            target_class = labels[i].item()
            
            # Compute CAM
            cam = compute_cam(fmap, model.classifier.weight.data,target_class)
            img_np = img.permute(1, 2, 0).cpu().numpy()

            # Original image on first row
            axes[0][i].imshow(img_np)
            axes[0][i].set_title(f"Original - Class {target_class}")
            axes[0][i].axis('off')
            cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))

            cam_normalized = np.clip(cam_resized, 0, 1)

            '''This part is for making a single-channel heatmap where only the red channel has values'''
            # heatmap = np.zeros_like(img_np)
            # heatmap[..., 0] = cam_normalized

            '''This part is for drawing a heatmap with 3 channels of colors'''
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_normalized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR by default
            heatmap = heatmap.astype(np.float32) / 255.0

            overlay = 0.5 * img_np + 0.5 * heatmap
            overlay = np.clip(overlay, 0, 1)

            axes[1][i].imshow(overlay)
            axes[1][i].set_title(f"CAM - Class {target_class}")
            axes[1][i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    from loader import H5ImageLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    minibatch_size = 32
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    DATA_PATH = '../../../DLcourse/GroupTask/segmentation/data'

    ## data loader
    loader_train = H5ImageLoader(DATA_PATH+'/images_train.h5', minibatch_size, DATA_PATH+'/labels_train.h5')
    loader_test = H5ImageLoader(DATA_PATH+'/images_val.h5', 20, DATA_PATH+'/labels_val.h5')
    dataloader_train=iter(loader_train)
    dataloader_test=iter(loader_test)
    images, labels,sp,br = next(dataloader_train)
    print(dataloader_train)
    print(images[0])
    print(labels[0])
    print(sp[0])
    print(br[0])

    batch_size=32
    loss_function = torch.nn.CrossEntropyLoss()
    train_mode=False # if False, will use trained local model.
    model_use="CNN_species" # choose the model used


    # try 4 types of models with CNN and ResNET and with breed and species as labels
    if model_use=="Res_breed":
        if train_mode == True:
            model_train=ResNetBackbone(num_classes=37,pretrained=True)
            fit_sgd(model_train, dataloader_train,"breed", 20, 0.01,
                    "../../../DLcourse/GroupTask/segmentation/Res_breed.pt")
        model_test = ResNetBackbone(num_classes=37,pretrained=False)
        model_test.load_state_dict(torch.load(f"{model_use}.pt"))  # Load model from local
        model_test.to(device)
        model_test.eval()
        visualize_cam(model_test,loader_train,"breed") # Use loaded model on training dataset to show CAM

    elif model_use=="Res_species":
        if train_mode == True:
            model_train=ResNetBackbone(num_classes=2,pretrained=True)
            fit_sgd(model_train, dataloader_train,"species", 20, 0.01,
                    "../../../DLcourse/GroupTask/segmentation/Res_species.pt")
        model_test = ResNetBackbone(num_classes=2,pretrained=False)
        model_test.load_state_dict(torch.load(f"{model_use}.pt"))  # Load model from local
        model_test.to(device)
        model_test.eval()
        visualize_cam(model_test,loader_train,"species") # Use loaded model on training dataset to show CAM

    elif model_use=="CNN_breed":
        if train_mode == True:
            model_train=CNN(out_channels=256,num_classes=37)
            fit_sgd(model_train, dataloader_train, "breed", 50, 0.01, "CNN_breed.pt")
        model_test =CNN(out_channels=256,num_classes=37)
        model_test.load_state_dict(torch.load(f"{model_use}.pt"))  # Load model from local
        model_test.to(device)
        model_test.eval()
        visualize_cam(model_test,loader_train,"breed") # Use loaded model on training dataset to show CAM

    elif model_use == "CNN_species":
        if train_mode == True:
            model_train = CNN(out_channels=256, num_classes=2)
            fit_sgd(model_train, dataloader_train,"species", 50, 0.01, "CNN_species.pt")
        model_test = CNN(out_channels=256, num_classes=2)
        model_test.load_state_dict(torch.load(f"{model_use}.pt"))  # Load model from local
        model_test.to(device)
        model_test.eval()
        visualize_cam(model_test,loader_train,"species") # Use loaded model on training dataset to show CAM


