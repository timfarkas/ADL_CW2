**ADL_CW2**

**How to the Oxford-IIIT Pets DataLoader**
1) 'from custom_loader import get_pet_dataloaders'
2) 'train_loader, val_loader, test_loader = get_pet_dataloaders(
    batch_size=64, 
    supervision_type="bbox"
)'
- can set batch size to what is required
- supervision_type can be "class", "species" or "bbox"
