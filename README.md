ADL_CW2

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