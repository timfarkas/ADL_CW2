import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
# image = cv2.imread("sample_image_unnormalized.png")

def get_MCG(image, threshold=0.5):
    h, w = image.shape[:2]

    # Selective Search setup
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    # Create an empty mask (same size as image, single channel)
    mask = np.zeros((h, w), dtype=np.float32)

    # Fill top N proposals
    top_n = 10000
    for (x, y, rw, rh) in rects[:top_n]:
        mask[y:y+rh, x:x+rw] += 1.0

    # Normalize the mask to [0, 1]
    mask = np.clip(mask / mask.max(), 0, 1)

    # Optional: Threshold the mask (e.g., keep only confident regions)
    if threshold > 0:
        mask[mask < threshold] = 0.0

    colored_mask = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Convert to RGB so red stays red
    colored_mask_rgb = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)

    # Blend with RGB image (assuming image is already RGB!)
    blended = cv2.addWeighted(image, 0.7, colored_mask_rgb, 0.3, 0)
    return blended