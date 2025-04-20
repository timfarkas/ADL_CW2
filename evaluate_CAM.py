import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import CAMManager, SelfTraining, UNet

from data import OxfordPetDataset, create_dataloaders, create_sample_loader_from_existing_loader
from evaluation import evaluate_dataset, evaluate_model, get_binary_from_normalization, evaluate_model_with_grabcut
# from utils import save_tensor_dataset, unnormalize
import random
import numpy as np

'''Fix randomization seed'''


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

'''Define device'''
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

'''Create directories'''
log_dir = os.path.join("cam_data", "CAM")
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


print("Loading CAM dataset...")
resized_data = (torch.load("cam_data/patched_resized_64_species_breed_cam_mask.pt"))
#
'''Evaluating CAM'''
threshold = 0
for i in range(1, 7):
    evaluate_dataset(resized_data, 8, f"evaluation_tred{threshold}", threshold=threshold, output_dir=eva_dir)
    threshold = round(threshold + 0.1, 2)
