# Fine-Tuning Real-ESRGAN on FASTMRI

This guide provides instructions for fine-tuning the [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) model using the [FASTMRI](https://fastmri.org/) dataset.

---

## 📦 Prerequisites

### 1. Install Required Packages

Install dependencies:

```bash
pip install -r requirements_fastmri.txt
```

---

### 2. Download Pretrained Models

Download the pretrained weights into the `experiments/pretrained_models` folder:

```bash
# RealESRGAN x4 Generator
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models

# RealESRNet x4
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth -P experiments/pretrained_models
```

---

### 3. Download Dataset

Download the dataset folder from Google Drive:

📁 [FASTMRI Dataset (Google Drive)](https://drive.google.com/drive/folders/1QsPIuOmi7NTKkBUyWGspQEVFaWufet1J)

---

## 🏋️ Training

### 1. Train on a Single GPU

```bash
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml --auto_resume
```

---

### 2. Train on Multiple GPUs (e.g., 4 GPUs)

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

---

## 🚀 Inference

To run inference on a folder of low-resolution images:

```bash
python inference_realesrgan.py \
  -n RealESRGAN_x4plus \
  -i datasets/train_val_test/test/lr \
  -o datasets/results \
  --model_path /path/to/your/fine_tuned_model.pth
```

Replace `/path/to/your/fine_tuned_model.pth` with your trained model checkpoint.

---

## 📈 Evaluation

To compute PSNR and SSIM between your super-resolved results and ground-truth high-resolution images:

```bash
python eval_metrics.py --gt_dir /path/to/hr --sr_dir /path/to/results
```

- `--gt_dir`: Path to high-resolution ground-truth images (e.g., `datasets/train_val_test/test/hr`)
- `--sr_dir`: Path to super-resolved outputs from inference (e.g., `datasets/results`)

---

## 📊 Output Example

```
Average PSNR: 31.8723 dB
Average SSIM: 0.8872
```

---

Happy fine-tuning! 🚀
