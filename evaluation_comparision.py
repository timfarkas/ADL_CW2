import os

import torch
import torch.nn as nn
from PIL import ImageFile
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from data import OxfordPetDataset,create_dataloaders,create_sample_loader_from_existing_loader
from evaluation import evaluate_dataset, evaluate_model, get_binary_from_normalization
from utils import save_tensor_dataset, unnormalize
from models import CAMManager, SelfTraining, UNet


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model_dir = os.path.join("checkpoints", "EVA")
os.makedirs(model_dir, exist_ok=True)

classification_mode = "breed"
batch_size = 64
dataloader_train, dataloader_val, dataloader_test = create_dataloaders(
    batch_size,
    resize_size=64,
    target_type=[classification_mode, "segmentation"],
    lazy_loading=True,
    shuffle=False)
gt_sample_loader = create_sample_loader_from_existing_loader(dataloader_val,100,64,False)

dataset = dataloader_test.dataset  # get the dataset used inside the original DataLoader
test_data = [
    (img,(mask["segmentation"]).float(),)
    for img, mask in dataset
]

# Wrap it in a new DataLoader
dataloader_test_new = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,  # or True if you want
    num_workers=0)

epochs=40
# model_path = os.path.join(model_dir, f"baseline_model_epoch{epochs}.pt")
model_name=f"baseline_model_epoch{epochs}"
# model_name=f"best_model_selftrain"
model_path = os.path.join(model_dir, f"{model_name}.pt")
model_new = UNet(3, 1).to(device)
model_new.load_state_dict(torch.load(f"{model_path}"))
print("Model loaded successfully.")
model_new.eval()

print("Evaluation on small sample")
evaluate_model(model_new, gt_sample_loader, 8, f"{model_name}_smallset", threshold=0.5,output_dir="checkpoints/EVA/visualization")

print("Evaluation on testset")
evaluate_model(model_new, dataloader_test_new, 8, f"{model_name}_testset", threshold=0.5,output_dir="checkpoints/EVA/visualization")