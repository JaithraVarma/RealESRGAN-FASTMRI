import os

# Define paths
hr_dir = 'datasets/train_val_test/train/hr'  # Path to HR images
meta_file = 'datasets/train_val_test/fastmri_train_meta.txt'  # Output meta file

# Create meta info file
with open(meta_file, 'w') as f:
    for img_file in os.listdir(hr_dir):
        if img_file.endswith('.png'):
            # Write relative path to the image
            f.write(f"{img_file}\n")

print(f"Meta info file created at {meta_file}")