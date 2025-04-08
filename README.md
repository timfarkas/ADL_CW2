ADL_CW2

**Downloading and Preparing the Dataset**

**Initialize and download the dataset**
dataset = OxfordPetDataset(root_dir="oxford_pet_data").prepare_dataset()

**Get all data items**
all_data = dataset.get_all_data()
print(f"Dataset contains {len(all_data)} images")

**Create dataloaders for classification**
train_loader, val_loader, test_loader = create_dataloaders(
    dataset, 
    batch_size=32, 
    target_type="class"
)

**Alternatively, for multiple targets**
train_loader, val_loader, test_loader = create_dataloaders(
    dataset, 
    batch_size=16, 
    target_type=["class", "bbox"]
)

**Available Target Types**
The dataset supports multiple target types that can be used individually or combined:
- "class" or "breed": Returns the breed class index (0-36, -1 for background images)
- "species": Returns the species class (0 for cat, 1 for dog, -1 for background images)
- "bbox": Returns the pet head bounding box coordinates. Option with normalize_bbox (default=True) to output 
relative values between 0 - 1. [xmin, ymin, xmax, ymax], tensor format: (B, 4) where B = batch size
- "segmentation": Returns the segmentation mask. Trimap segmentation: 1, 2, or 3. Shape is (B, 64, 64) where
B = batch size

**Image shape**
- (B, C, H, W)
- B = batch size, C = channels 3 for RGB, H = height 64, W = width 64