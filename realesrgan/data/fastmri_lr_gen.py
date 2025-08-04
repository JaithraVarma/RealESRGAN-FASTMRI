import os
import cv2
from tqdm import tqdm

# Set paths
gt_dir = 'datasets/train_val_test/train/hr'  # input GT images
lq_dir = 'datasets/train_val_test/train/lr'  # output LQ images
scale = 4  # downscaling factor

os.makedirs(lq_dir, exist_ok=True)

# Loop over HR images
for fname in tqdm(os.listdir(gt_dir)):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(gt_dir, fname)
    img = cv2.imread(img_path)

    # Compute downscaled size
    h, w = img.shape[:2]
    h_down, w_down = h // scale, w // scale
    img_lq = cv2.resize(img, (w_down, h_down), interpolation=cv2.INTER_CUBIC)

    # Save LQ image
    cv2.imwrite(os.path.join(lq_dir, fname), img_lq)
