import torch
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, random_split, ConcatDataset

generator = torch.Generator().manual_seed(27)

transform = transforms.Compose([
    # could set to 512×512 if increases performance
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
#
# target_transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

# target type is 'category' for image level labels as per requirements
full_dataset = []
for split in ['trainval', 'test']:
    dataset = OxfordIIITPet(
        root = './data',
        split = split,
        target_types = 'category',
        transform=transform,
        download = True)
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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training sample: {len(train_dataset)}")
print(f"Validation sample: {len(val_dataset)}")
print(f"Testing sample: {len(test_dataset)}")

images, labels = next(iter(train_loader))
print(images,labels)
print("One batch loaded successfully.")