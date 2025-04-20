import os

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import (
    create_dataloaders,
)
from evaluation import (
    evaluate_model,
    evaluate_dataset,
    remap_mask
)
from models import SelfTraining, UNet

"""Fix randomization seed"""


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(10)

'''Define device'''
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

'''Create directories'''
log_dir = os.path.join("checkpoints", "Bootstrap")
os.makedirs(log_dir, exist_ok=True)
model_dir = os.path.join(log_dir, "model")
os.makedirs(model_dir, exist_ok=True)
eva_dir = os.path.join(log_dir, "evaluation")
os.makedirs(model_dir, exist_ok=True)
pre_dir = os.path.join(log_dir, "predicted_masks")
os.makedirs(model_dir, exist_ok=True)

'''Clean files in checkpoint'''  # this part of code needs to be disabled if going to use local saved models
for root, dirs, files in os.walk(log_dir):
    for file in files:
        file_path = os.path.join(root, file)
        os.remove(file_path)

'''Preparing datasets'''
batch_size = 64
classification_mode = "breed"
model_type = "CNN"
dataloader_train, dataloader_val, dataloader_test = create_dataloaders(
    batch_size,
    resize_size=64,
    target_type=[classification_mode, "segmentation"],
    lazy_loading=True,
    shuffle=False)

import matplotlib.pyplot as plt
import torch

# Get a single batch from the validation loader
batch = next(iter(dataloader_val))
images, targets = batch

# targets is expected to be a dict with classification and segmentation labels
cls_labels = targets["breed"]
seg_masks = targets["segmentation"]

# Denormalization parameters (ImageNet)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Visualize the first N samples
n_samples = 8
fig, axs = plt.subplots(3, n_samples, figsize=(n_samples * 3, 3 * 3))

for i in range(n_samples):
    # De-normalize image
    img = images[i] * std + mean
    img = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    # Segmentation mask
    mask = seg_masks[i].squeeze().cpu().numpy()

    # Classification label (just integer index)
    label = cls_labels[i].item()

    axs[0, i].imshow(img)
    axs[0, i].axis("off")
    axs[0, i].set_title(f"Image {i}")

    axs[1, i].imshow(mask, cmap="gray")
    axs[1, i].axis("off")
    axs[1, i].set_title("Seg. Mask")

    axs[2, i].text(0.5, 0.5, f"Breed: {label}", ha="center", va="center", fontsize=12)
    axs[2, i].axis("off")

plt.tight_layout()
plt.show()


print("Loading CAM dataset...")
resized_data = (torch.load("cam_data/resized_64_species_breed_cam_mask_raw.pt"))
#
'''Evaluating CAM'''
# threshold = 0
# for i in range(1, 7):
#     evaluate_dataset(resized_data, 8, f"evaluation_tred{threshold}", threshold=threshold, output_dir=eva_dir)
#     threshold = round(threshold + 0.1, 2)

'''Binarize CAM as training labels and create dataloader'''
cam_threshold = 0.2
binarized_data = []
sample_count = 0
for image, cam, mask in resized_data:
    cam_binary = (cam > cam_threshold).float()
    binarized_data.append((image, cam_binary, mask))
    sample_count += 1

binarized_dataset = torch.utils.data.TensorDataset(
    torch.stack([x[0] for x in binarized_data]),
    torch.stack([x[1] for x in binarized_data]),
    torch.stack([x[2] for x in binarized_data]),
)

backup_dataset = binarized_dataset
new_dataset = backup_dataset

SelfTraining.visualize_predicted_masks(
    new_dataset, num_samples=8, save_path=f"{pre_dir}/original_mask.png"
)
dataloader_new = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

'''Get ground truth samples for evaluation'''
gt_sample_loader = create_sample_loader_from_existing_loader(dataloader_val, 100, 64, False)

'''Get ground truth samples for add-on training'''
train_data = [dataloader_train.dataset[i] for i in range(100)]
gt_data_for_train = [
    (img, get_binary_from_normalization(mask["segmentation"]), get_binary_from_normalization(mask["segmentation"]),)
    for img, mask in train_data]
gt_data_for_train = TensorDataset(
    torch.stack([x[0] for x in gt_data_for_train]),
    torch.stack([x[1] for x in gt_data_for_train]),
    torch.stack([x[2] for x in gt_data_for_train]),
)

'''Preparing for selflearning(bootstraping)'''
Skip_first_round = True # if true, will use the model"first_round_model.pt" saved at current folder to save time
Use_Bootstrap_Models = False  # if true, will used saved models in bootstrap interations
Addon_Dataset = False  # if true, new dataset will be added on original dataset and passed to the next round altogether
Add_Groundtruth = True  # if true, new ground truth will be added in the training loop. Doesn't work together with Addon_dataset
TrainSeed = False  # if true, will use seed-loss in training process (not quite work with Add_on)
GrabCut = False # if true, will use GrabCut in the process of generating new pseudo labels
Mixlabel = False  # if true, will use mixlabel in the process generating new pseudo labels
filter = 0 # value of threshold for simple filter


epochs = 5 # number of epochs each round
BOOTSTRAP_ROUNDS = 10 # maximum number of self-training
'''Running Bootstrap'''
for round_num in range(1, BOOTSTRAP_ROUNDS + 1):
    print(f"\nBootstrapping Round {round_num}")
    model_new = UNet(3, 1).to(device)  # each round, create a new model

    if round_num == 1:  # the first round should always use the dataset get from CAM, while other rounds should use dataset from previous round
        dataloader_bootstrap = dataloader_new
    print(f"Dataset: {len(dataloader_bootstrap.dataset)} samples")

    loss_function = nn.BCEWithLogitsLoss()

    model_path = os.path.join(model_dir,
                              f"Grabct{GrabCut}_Seed{TrainSeed}_filter{filter}_epoch{epochs}_addon{Addon_Dataset}_bootstrap_round{round_num}.pt")

    if round_num == 1 and Skip_first_round:  # since firstround process will always be the same(only number of epochs may be different), can used saved first round model to save time
        model_new = UNet(3, 1).to(device)
        model_new.load_state_dict(torch.load(f"first_round_model.pt"))
        print("Model loaded successfully.")
    elif Use_Bootstrap_Models and os.path.exists(model_path):  # if going to use pre-saved boostrap models
        model_new = UNet(3, 1).to(device)
        model_new.load_state_dict(torch.load(f"{model_path}"))
        print("Model loaded successfully.")
    else:
        if TrainSeed:  # if use seed loss method in training loop
            SelfTraining.fit_sgd_seed(
                model_new, dataloader_bootstrap, epochs, 0.05, loss_function, model_path, device=device, threshold=0.1
            )
        else:  # normal training method
            SelfTraining.fit_sgd_pixel(
                model_new, dataloader_bootstrap, epochs, 0.05, loss_function, model_path, device=device
            )

    model_new.eval()

    '''checking loader by visualization'''

    threshold = 0.2  # threshold for evaluating the model against ground-truth samples
    for i in range(2):
        evaluate_model(model_new, gt_sample_loader, 8, f"bootstrap_round_{round_num}", threshold=threshold,
                       output_dir=eva_dir)
        threshold = round(threshold + 0.3, 2)

    print("Generating new dataset from prediction")

    if GrabCut: # use GrabCut as filter when generating new labels
        new_dataset_predict = SelfTraining.predict_segmentation_dataset_with_grabcut(
            model_new, dataloader_new, threshold=0.2)
    elif Mixlabel:  # The Mixlabel method needs to take probability input, however the input for the first round is binary (binaried CAM). We have to skip first round when mixlabels
        if round_num == 1:
            new_dataset_predict = SelfTraining.predict_segmentation_dataset_with_basicfilter(
                model_new, dataloader_new, threshold=0)
        else:
            new_dataset_predict = SelfTraining.predict_segmentation_dataset_with_mixlabel(
                model_new, dataloader_bootstrap, threshold=0.2)
    else: # basic way of generating new labels by simply applying a filter based on a threshold value
        new_dataset_predict = SelfTraining.predict_segmentation_dataset_with_basicfilter(
            model_new, dataloader_new, threshold=filter)

    #saved the newly-generated dataset for visualization
    SelfTraining.visualize_predicted_masks(
        new_dataset_predict,
        num_samples=8,
        save_path=f"{pre_dir}/Seed{TrainSeed}_Grabct{GrabCut}_filter{Basicfilter}_epoch{epochs}_addon{Addon_Dataset}_bootstrap_round{round_num}.png",
    )

    if Addon_Dataset: #this will add new dataset on top of oringinal dataset and double the number of samples
        all_images = torch.cat(
            [dataloader_new.dataset.tensors[0], new_dataset_predict.tensors[0]], dim=0
        )
        all_labels = torch.cat(
            [dataloader_new.dataset.tensors[1], new_dataset_predict.tensors[1]], dim=0
        )
        all_masks = torch.cat(
            [dataloader_new.dataset.tensors[2], new_dataset_predict.tensors[2]], dim=0
        )
        new_dataset = TensorDataset(all_images, all_labels, all_masks)
        dataloader_bootstrap = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

    elif Add_Groundtruth: #this will add some ground truth samples in the training loop
        replication_factor = 1

        gt_images_repeated = torch.cat([gt_data_for_train.tensors[0]] * replication_factor, dim=0)
        gt_labels_repeated = torch.cat([gt_data_for_train.tensors[1]] * replication_factor, dim=0)
        gt_masks_repeated = torch.cat([gt_data_for_train.tensors[2]] * replication_factor, dim=0)

        if gt_labels_repeated.ndim == 3:
            gt_labels_repeated = gt_labels_repeated.unsqueeze(1)
        if gt_masks_repeated.ndim == 3:
            gt_masks_repeated = gt_masks_repeated.unsqueeze(1)
        # Concatenate with pseudo-labeled dataset
        all_images = torch.cat([new_dataset_predict.tensors[0], gt_images_repeated], dim=0)
        all_labels = torch.cat([new_dataset_predict.tensors[1], gt_labels_repeated], dim=0)
        all_masks = torch.cat([new_dataset_predict.tensors[2], gt_masks_repeated], dim=0)
        new_dataset = TensorDataset(all_images, all_labels, all_masks)
        dataloader_bootstrap = DataLoader(new_dataset, batch_size=batch_size, shuffle=True)
    else:
        dataloader_bootstrap = DataLoader(new_dataset_predict, batch_size=batch_size, shuffle=False)

    print(f"Round {round_num} finished")
