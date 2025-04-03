import torch
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np

generator = torch.Generator().manual_seed(27)

transform = transforms.Compose(
    [
        # Force resize to EXACT dimensions, not preserving aspect ratio
        transforms.Resize(
            (256, 256), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        # Optional normalization if using pretrained model
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Proper transformation for segmentation masks
# Modified target transform for segmentation masks
target_transform = transforms.Compose(
    [
        # Resize using nearest neighbor to preserve class values
        transforms.Resize(
            (256, 256), interpolation=transforms.InterpolationMode.NEAREST
        ),
        # Use the numpy path instead of ToTensor()
        lambda x: torch.from_numpy(np.array(x)).long().unsqueeze(0),
    ]
)

# target type is 'category' for image level labels as per requirements
full_dataset = []
for split in ["trainval", "test"]:
    dataset = OxfordIIITPet(
        root="./data",
        split=split,
        target_types="segmentation",
        transform=transform,
        target_transform=target_transform,
        download=True,
    )
    full_dataset.append(dataset)

OxfordPets_dataset = ConcatDataset(full_dataset)

# Splitting the dataset: 70% training, 15% validation, 15% testing
total_size = len(OxfordPets_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    OxfordPets_dataset, [train_size, val_size, test_size], generator=generator
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training sample: {len(train_dataset)}")
print(f"Validation sample: {len(val_dataset)}")
print(f"Testing sample: {len(test_dataset)}")
