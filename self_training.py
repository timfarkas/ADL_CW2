import os

import torch
import torch.nn as nn
from PIL import ImageFile
from torch.utils.data import DataLoader, TensorDataset

from CAM.cam_model import CNN, ResNetBackbone, fit_sgd
from models import CAMManager, SelfTraining, UNet

from data import OxfordPetDataset,create_dataloaders,create_sample_loader_from_existing_loader
from evaluation import evaluate_dataset, evaluate_model
from utils import save_tensor_dataset, unnormalize


import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

classification_mode = "breed"
batch_size = 64
dataloader_train, dataloader_val, dataloader_test = create_dataloaders(
    batch_size,
    resize_size=64,
    target_type=[classification_mode, "segmentation"],
    lazy_loading=True,
    shuffle=False)
#
#
# ImageFile.LOAD_TRUNCATED_IMAGES = True


#

# ## HYPERPARAMETERS
# num_epochs = 20
model_type = "CNN"  ## CNN, Res
model_dir = os.path.join("checkpoints", "CAM")
os.makedirs(model_dir, exist_ok=True)
# use_existing_model = True # if True, will use saved model from local
#
# loss_function = torch.nn.CrossEntropyLoss()

### MODEL INIT
# model_mode = classification_mode  ## species, breed
# model_name = f"{model_type}_{model_mode}"
# print(f"Using model {model_name}")
# model_path = os.path.join(model_dir, model_name)
#
# num_out = 37 if model_mode == "breed" else 2
#
# threshold=0
#
# if use_existing_model and os.path.exists(model_path):
#     print(f"Using MODEL {model_path}.pt")
#     model_test = (
#         ResNetBackbone(num_classes=num_out, pretrained=False)
#         if "Res" in model_name
#         else CNN(out_channels=256, num_classes=num_out)
#     )
#     model_test.load_state_dict(torch.load(f"{model_path}"))
#     model_test.to(device)
#
# else:
#     model_test = (
#         ResNetBackbone(num_classes=num_out, pretrained=False)
#         if "Res" in model_name
#         else CNN(out_channels=256, num_classes=num_out)
#     )
#     fit_sgd(
#         model_test,
#         dataloader_train,
#         classification_mode,
#         num_epochs,
#         0.05,
#         batch_size,
#         loss_function,
#         model_path,
#         device=device,
#     )

### USING CAM AS LABELS IN A NEW DATASET
# print("Generating datasets from CAM")
# use_existing_cam = True #if true, will use saved cam datase

print("Loading CAM dataset...")
data = torch.load("cam_data/resized_64_species_breed_cam_mask_binary.pt")

# Case 1: If it’s already a TensorDataset
if isinstance(data, TensorDataset):
    new_dataset = data

# Case 2: If it’s a list of tuples (image, cam, mask)
elif isinstance(data, list) and isinstance(data[0], tuple):
    images = torch.stack([x[0] for x in data])
    cams   = torch.stack([x[1] for x in data])
    masks  = torch.stack([x[2] for x in data])

    # # Binarize the CAMs
    # threshold = 0.8  # adjust as needed
    # binary_cams = (cams > threshold).float()

    # Use binary CAMs as pseudo labels (instead of original masks or raw cams)
    new_dataset = TensorDataset(images,cams, masks)

else:
    raise TypeError("Unexpected format of loaded CAM dataset.")

backup_dataset=new_dataset

# if use_existing_cam and os.path.exists("cam_data/new_dataset.pt"):
#     print("Loading CAM dataset...")
#     data = torch.load("cam_data/new_dataset.pt")
#     new_dataset = TensorDataset(data["images"], data["cams"], data["masks"])
# else:
#     cam_instance = CAMManager(
#         model=model_test,
#         dataloader=dataloader_train,
#         method="Classical",
#         target_type=classification_mode,
#     )
#     new_dataset = cam_instance.dataset
#     save_tensor_dataset(new_dataset, "cam_data/new_dataset.pt")

SelfTraining.visualize_predicted_masks(
    new_dataset, num_samples=8, save_path=f"visualizations/round_0.png"
)
dataloader_new = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

print("Getting ground_truth_sample")
gt_sample_loader = create_sample_loader_from_existing_loader(dataloader_val,100,64,False)
gt_train_loader = create_sample_loader_from_existing_loader(dataloader_train,100,64,False)


'''checking loader by visualization'''
# for images, masks in gt_sample_loader:
#     break  # Just the first batch
# # Loop through batch
# batch_size = images.size(0)
# for i in range(batch_size):
#     img = TF.to_pil_image(images[i].cpu())
#
#     # If mask is [1, H, W], squeeze it to [H, W]
#     mask = masks[i]
#     if mask.dim() == 3:
#         mask = mask.squeeze(0)
#     mask = mask.cpu().numpy()
#
#     # Plot image and mask
#     fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#     axs[0].imshow(img)
#     axs[0].set_title("Image")
#     axs[0].axis("off")
#
#     axs[1].imshow(mask, cmap='gray')
#     axs[1].set_title("Segmentation Mask")
#     axs[1].axis("off")
#
#     plt.tight_layout()
#     plt.show()
#
# for images, masks in gt_sample_loader:
#     print("Image batch shape:", images.shape)  # Expecting [B, C, H, W]
#     print("Mask batch shape:", masks.shape)    # Expecting [B, 1, H, W] or [B, H, W]
#     break  # Only check the first batch

# cam_sample= CAMManager(
#     model=model_test,
#     dataloader=gt_sample_loader,
#     method="Classical",
#     target_type=classification_mode,
# )
#
# new_gt_dataset = cam_sample.dataset


'''checking loader by visualization'''
# loader = DataLoader(new_gt_dataset, batch_size=4)
# for images, cams, masks in loader:
#     break  # Just first batch
#
# # Convert and display a few samples
# batch_size = images.size(0)
# for i in range(batch_size):
#     fig, axs = plt.subplots(1, 3, figsize=(12, 4))
#
#     # Unnormalize image if needed
#     img = TF.to_pil_image(images[i].cpu())
#     cam = cams[i][0].cpu().numpy()
#     mask = masks[i][0].cpu().numpy()
#
#     axs[0].imshow(img)
#     axs[0].set_title("Image")
#     axs[0].axis("off")
#
#     axs[1].imshow(cam, cmap='jet')
#     axs[1].set_title("CAM")
#     axs[1].axis("off")
#
#     axs[2].imshow(mask, cmap='gray')
#     axs[2].set_title("Ground Truth Mask")
#     axs[2].axis("off")
#
#     plt.tight_layout()
#     plt.show()

# for batch in loader:
#     print("Batch loaded:")
#     for i, tensor in enumerate(batch):
#         print(f"Tensor {i} shape: {tensor.shape}")
#     break  # only inspect first batch

# threshold=0
# for i in range(7):
#     evaluate_dataset(new_gt_dataset, 8, f"test", threshold=threshold)
#     threshold = round(threshold+0.1,2)

"""
SelfTraining.visualize_cam_samples(dataloader_new)
"""

use_bootstrap_models=False#if ture, will used saved models in bootstrap
add_on_dataset=True #if ture, new dataset will be added on original dataset and passed to the next round altogether
AddGroundTruth=True
TrainSeed=False #using Seed model (not quite work with Add_on)
Grabcut=True
epochs=5
filter=0
BOOTSTRAP_ROUNDS = 15
### RUNNING BOOSTRAP AND UPDATE DATASET EACH ROUND
for round_num in range(1, BOOTSTRAP_ROUNDS + 1):
    print(f"\nBootstrapping Round {round_num}")
    model_new = UNet(3, 1).to(device)
    if round_num==1:
        dataloader_bootstrap=dataloader_new
    print(f"Dataset: {len(dataloader_bootstrap.dataset)} samples")
    loss_function = nn.BCEWithLogitsLoss()
    # loss_function = nn.MSELoss()
    model_path = os.path.join(model_dir, f"Seed{TrainSeed}_Grabct{Grabcut}_filter{filter}_epoch{epochs}_addon{add_on_dataset}_bootstrap_round{round_num}.pt")
    if use_bootstrap_models and os.path.exists(model_path) :
        model_new = UNet(3, 1).to(device)
        model_new.load_state_dict(torch.load(f"{model_path}"))
        print("Model loaded successfully.")
    else:
        if TrainSeed:
            SelfTraining.fit_sgd_seed(
                model_new, dataloader_bootstrap, epochs, 0.05, loss_function, model_path, device=device,threshold=0.1
            )
        else:
            SelfTraining.fit_sgd_pixel(
                model_new, dataloader_bootstrap, epochs, 0.05, loss_function, model_path, device=device
            )

    model_new.eval()

    '''checking loader by visualization'''
    # with torch.no_grad():
    #     for images, masks in gt_sample_loader:
    #         images = images.to(device)
    #         masks = masks.to(device)
    #
    #         preds = torch.sigmoid(model_new(images))
    #         preds_bin = (preds > 0.5).float()
    #
    #         images = images.cpu()
    #         preds = preds.cpu()
    #         masks = masks.cpu()
    #
    #         # for i in range(min(10, images.shape[0])):
            #     fig, axs = plt.subplots(1, 3, figsize=(10, 3))
            #     img = images[i].permute(1, 2, 0).numpy()
            #     img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
            #
            #     axs[0].imshow(img)
            #     axs[0].set_title("Input Image")
            #     axs[0].axis("off")
            #
            #     axs[1].imshow(preds[i].squeeze(), cmap="gray")
            #     axs[1].set_title("Prediction")
            #     axs[1].axis("off")
            #
            #     axs[2].imshow(masks[i].squeeze(), cmap="gray")
            #     axs[2].set_title("Ground Truth")
            #     axs[2].axis("off")
            #
            #     plt.tight_layout()
            #     plt.show()
            # break  # only show first batch

    threshold = 0.2
    for i in range(2):
        evaluate_model(model_new,gt_sample_loader, 8, f"bootstrap_round_{round_num}", threshold=threshold)
        threshold = round(threshold+0.3,2)
    print("Generating new dataset from prediction")
    # new_dataset_predict = SelfTraining.predict_segmentation_dataset__basicfilter(
    #     model_new, dataloader_train, threshold=filter)
    if Grabcut:
        new_dataset_predict = SelfTraining.predict_segmentation_dataset_with_grabcut(
            model_new, dataloader_new)
    else:
        new_dataset_predict = SelfTraining.predict_segmentation_dataset_with_basicfilter(
            model_new, dataloader_new,threshold=0)

    SelfTraining.visualize_predicted_masks(
        new_dataset_predict,
        num_samples=8,
        save_path=f"visualizations/Seed{TrainSeed}_Grabct{Grabcut}_filter{filter}_epoch{epochs}_addon{add_on_dataset}_bootstrap_round{round_num}.png",
    )

    if add_on_dataset and AddGroundTruth:
        all_images = torch.cat(
            [dataloader_new.dataset.tensors[0], new_dataset_predict.tensors[0],gt_train_loader.dataset.tensors[0]], dim=0
        )
        all_labels = torch.cat(
            [dataloader_new.dataset.tensors[1], new_dataset_predict.tensors[1],gt_train_loader.dataset.tensors[1]], dim=0
        )
        all_masks = torch.cat(
            [dataloader_new.dataset.tensors[2], new_dataset_predict.tensors[2],gt_train_loader.dataset.tensors[1]], dim=0
        )
        new_dataset = TensorDataset(all_images, all_labels, all_masks)
        dataloader_bootstrap = DataLoader(new_dataset,batch_size=batch_size, shuffle=False)
    elif add_on_dataset and not AddGroundTruth:
        all_images = torch.cat(
            [new_dataset_predict.tensors[0],gt_train_loader.dataset.tensors[0]], dim=0
        )
        all_labels = torch.cat(
            [new_dataset_predict.tensors[1],gt_train_loader.dataset.tensors[1]], dim=0
        )
        all_masks = torch.cat(
            [new_dataset_predict.tensors[2],gt_train_loader.dataset.tensors[1]], dim=0
        )
        new_dataset = TensorDataset(all_images, all_labels, all_masks)
        dataloader_bootstrap = DataLoader(new_dataset,batch_size=batch_size, shuffle=False)
    else:
        dataloader_bootstrap = DataLoader(new_dataset_predict, batch_size=batch_size, shuffle=False)

    # Visualize results
    print(f"Visualizing predicted masks from Round {round_num}")



