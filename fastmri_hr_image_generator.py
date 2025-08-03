import h5py
import numpy as np
from PIL import Image
import os

# Define paths
h5_dir = 'C:/Users/srisr/Downloads/Knee-data-20250724T235546Z-1-002/Knee-data'  # Replace with your .h5 files directory
output_dir = 'C:/Users/srisr/Downloads/Real-ESRGAN-master/Real-ESRGAN-master/dataset/fastmri_hr'
os.makedirs(output_dir, exist_ok=True)

# Function to normalize and save image
def save_hr_image(hr_image, output_path):
    hr_image = (hr_image - hr_image.min()) / (hr_image.max() - hr_image.min() + 1e-8)  # Normalize to [0, 1]
    hr_image = (hr_image * 255).astype(np.uint8)  # Scale to [0, 255]
    Image.fromarray(hr_image).save(output_path)

# Function to reconstruct image from k-space with undersampling mask
def kspace_to_image(kspace, mask=None, recon_shape=(320, 320)):
    # kspace shape: [num_slices, height, width], complex-valued
    # mask shape: [width], boolean
    if mask is not None:
        # Apply undersampling mask (zero-fill unsampled lines)
        kspace = kspace * mask[np.newaxis, np.newaxis, :]  # Broadcast mask to kspace
    # Apply inverse FFT and shift
    image = np.fft.ifft2(kspace, axes=(-2, -1))  # iFFT on height, width
    image = np.fft.fftshift(image, axes=(-2, -1))  # Center the image
    # Compute magnitude
    image = np.abs(image)  # Shape: [num_slices, height, width]
    # Crop or pad to recon_shape (320x320)
    if image.shape[-2:] != recon_shape:
        image = resize_image(image, recon_shape)
    return image

# Function to resize/crop image to target shape
def resize_image(image, target_shape):
    # image shape: [num_slices, height, width]
    # Crop or pad to match target_shape (e.g., 320x320)
    slices, h, w = image.shape
    th, tw = target_shape
    output = np.zeros((slices, th, tw), dtype=image.dtype)
    for s in range(slices):
        # Center crop or pad
        h_start = (h - th) // 2 if h > th else 0
        w_start = (w - tw) // 2 if w > tw else 0
        h_end = h_start + min(th, h)
        w_end = w_start + min(tw, w)
        output[s, :min(th, h), :min(tw, w)] = image[s, h_start:h_end, w_start:w_end]
    return output

# Process each .h5 file
for h5_file in os.listdir(h5_dir):
    if h5_file.endswith('.h5'):
        try:
            with h5py.File(os.path.join(h5_dir, h5_file), 'r') as f:
                # Check for kspace dataset
                if 'kspace' in f:
                    kspace = f['kspace'][:]  # Shape: [35, 640, 368]
                    mask = f['mask'][:] if 'mask' in f else None  # Shape: [368]
                    # Reconstruct image from k-space
                    hr_image = kspace_to_image(kspace, mask=mask, recon_shape=(320, 320))
                else:
                    print(f"Skipping {h5_file}: No 'kspace' dataset found")
                    continue

                # Save slices
                if hr_image.ndim == 3:
                    for slice_idx in range(hr_image.shape[0]):
                        slice_image = hr_image[slice_idx]  # Shape: [320, 320]
                        output_path = os.path.join(output_dir, f"{h5_file.replace('.h5', '')}_slice{slice_idx:03d}.png")
                        save_hr_image(slice_image, output_path)
                elif hr_image.ndim == 2:
                    output_path = os.path.join(output_dir, f"{h5_file.replace('.h5', '')}.png")
                    save_hr_image(hr_image, output_path)
                else:
                    print(f"Skipping {h5_file}: Unexpected image dimensions {hr_image.shape}")

        except Exception as e:
            print(f"Error processing {h5_file}: {str(e)}")
            continue

print(f"HR images saved to {output_dir}")