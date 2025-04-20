import cv2
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("your_image.jpg")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()  # Or switchToSelectiveSearchQuality()

rects = ss.process()

# Visualize top 100 proposals
for (x, y, w, h) in rects[:100]:
    img = image.copy()
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()