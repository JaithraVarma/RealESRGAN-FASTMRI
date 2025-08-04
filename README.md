# 🧠 Fine-Tuning Real-ESRGAN on FASTMRI

This guide provides step-by-step instructions for fine-tuning the [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) model using the [FASTMRI](https://fastmri.org/) dataset for medical image super-resolution.

---

## 📦 Prerequisites

### 1. Install Dependencies

Install required Python packages:

```bash
pip install -r requirements_fastmri.txt
```

---

### 2. Download Pretrained Models

Download the pretrained generator models to the `experiments/pretrained_models` directory:

```bash
# RealESRGAN x4 (GAN-based)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models

# RealESRNet x4 (non-GAN baseline)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth -P experiments/pretrained_models
```

---

### 3. Download the Dataset

Download the FASTMRI dataset folder from Google Drive:

📁 [FASTMRI Dataset (Google Drive)](https://drive.google.com/drive/folders/1QsPIuOmi7NTKkBUyWGspQEVFaWufet1J)

The dataset follows an 80/10/10 split:

```
Train  : 936 images  
Val    : 117 images  
Test   : 117 images
```

---

## 🏋️‍♂️ Training

### 🔹 Train on a Single GPU

```bash
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml --auto_resume
```

### 🔹 Train on Multiple GPUs (e.g., 4 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=4321 \
  realesrgan/train.py \
  -opt options/finetune_realesrgan_x4plus.yml \
  --launcher pytorch \
  --auto_resume
```

> 🔧 Modify `options/finetune_realesrgan_x4plus.yml` to change training hyperparameters.

---

## 🧪 Validation & Evaluation

### 🔸 Inference on LQ Images

```bash
python inference_realesrgan.py \
  -n RealESRGAN_x4plus \
  -i datasets/train_val_test/test/lr \
  -o datasets/results \
  --model_path /path/to/your/fine_tuned_model.pth
```

Replace `/path/to/your/fine_tuned_model.pth` with your checkpoint file.

---

### 🔸 Compute PSNR / SSIM

Evaluate the model’s output with standard metrics:

```bash
python eval_metrics.py --gt_dir datasets/train_val_test/test/hr --sr_dir datasets/results
```

Example output:
```
Average PSNR: 31.8723 dB
Average SSIM: 0.8872
```

---

## 🛠️ Preprocessing: Generating LQ Images

To generate low-resolution (LR) images via bicubic downscaling from high-resolution (HR) images:

```bash
python realesrgan/data/fastmri_lr_gen.py
```

Ensure that LR and HR images have matching filenames and are correctly downscaled by the target scale (e.g., ×4).

---

## 📁 Directory Structure

```
datasets/
├── fastmri_hr/            # High-resolution images
├── fastmri_lq/            # Corresponding low-resolution images
├── train_val_test/        # Organized train/val/test split
│   ├── train/
│   ├── val/
│   └── test/
experiments/
├── pretrained_models/     # Downloaded weights
├── finetune_*/            # Training logs, checkpoints, validation results
options/
├── finetune_realesrgan_x4plus.yml
```

---

## 📌 Notes

- Validation is performed every `val_freq` iterations as defined in the config YAML.
- Outputs from validation can be found in `experiments/finetune_*/visualization/`.
- TensorBoard and WandB support is optionally available.

---

## 🔗 Resources

- [Real-ESRGAN GitHub](https://github.com/xinntao/Real-ESRGAN)
- [FASTMRI Dataset](https://fastmri.org/)
- [Paper: Real-ESRGAN](https://arxiv.org/abs/2107.10833)

---

Happy fine-tuning! 🧪🧠🔬
