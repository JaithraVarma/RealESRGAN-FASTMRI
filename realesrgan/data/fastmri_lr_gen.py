import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from realesrgan.data.realesrgan_dataset import RealESRGANDataset

opt = {
    'dataroot_gt': 'datasets/fastmri_hr',
    'meta_info': 'datasets/fastmri_meta.txt',
    'io_backend': {'type': 'disk'},
    'blur_kernel_size': 21,
    'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    'sinc_prob': 0.1,
    'blur_sigma': [0.2, 3],
    'betag_range': [0.5, 4],
    'betap_range': [1, 2],
    'blur_kernel_size2': 21,
    'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    'sinc_prob2': 0.1,
    'blur_sigma2': [0.2, 1.5],
    'betag_range2': [0.5, 4],
    'betap_range2': [1, 2],
    'final_sinc_prob': 0.8,
    'gt_size': 256,
    'use_hflip': True,
    'use_rot': False,
}

dataset = RealESRGANDataset(opt)
output_dir = 'datasets/fastmri_lq'
os.makedirs(output_dir, exist_ok=True)

for i in tqdm(range(len(dataset))):
    data = dataset[i]
    img_lq = data['gt']  # degraded low-res image as tensor
    img_path = os.path.basename(data['gt_path'])

    # Convert CHW Tensor to HWC BGR image
    img_np = img_lq.numpy().transpose(1, 2, 0)[:, :, ::-1] * 255
    img_np = img_np.clip(0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, img_path), img_np)
