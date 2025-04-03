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
"class" or "breed": Returns the breed class index (0-36)
"species": Returns the species class (0 for cat, 1 for dog)
"bbox": Returns the pet head bounding box coordinates
"segmentation": Returns the segmentation mask