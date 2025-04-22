# AI usage statement: AI was used to assist in debugging

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from data import create_dataloaders
from evaluation import evaluate_model, remap_mask
from models import UNet


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
'''Get ground truth samples for evaluation'''
val_data = [dataloader_val.dataset[i] for i in range(100)]
gt_data_for_val = [
    (img,remap_mask(mask["segmentation"]))
    for img, mask in val_data]
gt_data_for_train = TensorDataset(
    torch.stack([x[0] for x in gt_data_for_val]),
    torch.stack([x[1] for x in gt_data_for_val]),
)
gt_sample_loader = DataLoader(gt_data_for_train,batch_size=batch_size, shuffle=False)

dataset = dataloader_test.dataset  # get the dataset used inside the original DataLoader
test_data = [
    (img,remap_mask(mask["segmentation"]).float(),)
    for img, mask in dataset
]

# Wrap it in a new DataLoader
dataloader_test_new = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,  # or True if you want
    num_workers=0)

epochs=40

'''Pick which model to test'''
# model_name=f"baseline_model_epoch{1}"
# model_name=f"best_model_selftrain"
model_name=f"first_round_model"


model_path = os.path.join(model_dir, f"{model_name}.pt")
model_new = UNet(3, 1).to(device)
model_new.load_state_dict(torch.load(f"{model_path}"))

print("Model loaded successfully.")
model_new.eval()

print("Evaluation on small sample")
evaluate_model(model_new, gt_sample_loader, 8, f"{model_name}_smallset", threshold=0.5,output_dir="checkpoints/EVA/visualization")

print("Evaluation on testset")
evaluate_model(model_new, dataloader_test_new, 8, f"{model_name}_testset", threshold=0.5,output_dir="checkpoints/EVA/visualization")