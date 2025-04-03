
from CAM.loader import H5ImageLoader
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import os
from CAM.cam_model import ResNetBackbone, CNN, fit_sgd, visualize_cam, generate_cam_label_dataset
import torch
from self_training_model import UNet, fit_sgd_pixel,predict_pixel_classification_dataset,visualize_predicted_masks, visualize_first_batch
from custom_data import OxfordPetDataset
import custom_loader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



classification_mode = "breed"


'''
# 
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
dataset = OxfordPetDataset().prepare_dataset()
dataloader_train, dataloader_val,dataloader_test = custom_loader.create_dataloaders(dataset, target_type=classification_mode)
'''

DATA_PATH = '../../DLcourse/GroupTask/segmentation/data'
image_list = []
label_list = []
for images, _, _, labels in H5ImageLoader(DATA_PATH + '/images_train.h5', 32, DATA_PATH + '/labels_train.h5'):
    image_list.append(images)
    label_list.append(labels)
images_tensor = torch.cat(image_list, dim=0)   # shape: (N, C, H, W)
labels_tensor = torch.cat(label_list, dim=0)
tensor_dataset_train = TensorDataset(images_tensor, labels_tensor)
dataloader_train = DataLoader(tensor_dataset_train, batch_size=32, shuffle=False)



## HYPERPARAMETERS
num_epochs = 30
batch_size = 32

model_type = "CNN" ## CNN, Res
model_dir = os.path.join("checkpoints", "CAM")
os.makedirs(model_dir, exist_ok=True)
train_mode=True # if False, will use trained local mode



loss_function = torch.nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
minibatch_size = 32
sample_count = 0
for images, labels in dataloader_train :
    sample_count += images.size(0)
print(f"Total number of training samples: {sample_count}")


### MODEL INIT
model_mode = classification_mode ## species, breed
model_name = f"{model_type}_{model_mode}"
print(f"Using model {model_name}")
model_path = os.path.join(model_dir, model_name)

num_out = 37 if model_mode == "breed" else 2


### TRAIN MODEL
if train_mode:
    model_train = ResNetBackbone(num_classes=num_out,pretrained=False) if "Res" in model_name else CNN(out_channels=256,num_classes=num_out)
    fit_sgd(model_train,dataloader_train, classification_mode, num_epochs, 0.01, batch_size,   loss_function, model_path, device=device)

### TESTING MODEL
print(f"TESTING MODEL {model_path}.pt")
model_test = ResNetBackbone(num_classes=num_out,pretrained=False) if "Res" in model_name else CNN(out_channels=256,num_classes=num_out)
model_test.load_state_dict(torch.load(f"{model_path}"))

model_test.to(device)
model_test.eval()
### USING CAM AS LABELS IN A NEW DATASET
new_dataset=generate_cam_label_dataset(model_test,dataloader_train ,device="cpu")
visualize_predicted_masks(new_dataset, num_samples=4, save_path=f"visualizations/round_0.png")
BOOTSTRAP_ROUNDS = 5
dataloader_new = DataLoader(new_dataset, batch_size=32, shuffle=False)

### RUNNING BOOSTRAP AND UPDATE DATASET EACH ROUND
for round_num in range(1, BOOTSTRAP_ROUNDS + 1):
    print(f"\nBootstrapping Round {round_num}")
    # visualize_first_batch(dataloader_new, round_num)
    model_new = UNet(3, 1).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    model_path = os.path.join(model_dir, f"{model_type}_bootstrap_round_{round_num}.pt")
    fit_sgd_pixel(model_new, dataloader_new, 5, 0.01, loss_function, model_path)
    # model_test_new = UNet(3, 1).to(device)
    # model_test_new.load_state_dict(torch.load(f"{model_path}"))
    model_new.to(device)
    model_new.eval()

    new_dataset_predict = predict_pixel_classification_dataset(model_new, dataloader_train, threshold = min(0.95, 0.25 + 0.05 * round_num))
    visualize_predicted_masks(new_dataset_predict, num_samples=4, save_path=f"visualizations/round_{round_num}.png")
    all_images = torch.cat([dataloader_new.dataset.tensors[0], new_dataset_predict.tensors[0]], dim=0)
    all_labels = torch.cat([dataloader_new.dataset.tensors[1], new_dataset_predict.tensors[1]], dim=0)
    combined_dataset = TensorDataset(all_images, all_labels)
    print(f"Dataset size after round {round_num}: {len(combined_dataset)} samples")
    dataloader_new = DataLoader(combined_dataset, batch_size=32, shuffle=False)

    # Visualize results
    print(f"Visualizing predicted masks from Round {round_num}")
