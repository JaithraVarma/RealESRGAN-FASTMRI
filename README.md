# Fine-tuning Real-ESRGAN on FASTMRI

This guide provides instructions for fine-tuning the Real-ESRGAN model using the FASTMRI dataset.

## Prerequisites

1. **Install Required Packages**  
   Install the necessary dependencies by running:  
   ```bash
   pip install -r requirements_fastmri.txt
   ```

2. **Download Pretrained Models**  
   Download the pretrained models using the following commands:  
   - For `RealESRGAN_x4plus.pth`:  
     ```bash
     wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
     ```  
   - For `RealESRNet_x4plus.pth`:  
     ```bash
     wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth -P experiments/pretrained_models
     ```

3. **Download Dataset**  
   Download the datasets folder from the following link:  
   [FASTMRI Dataset](https://drive.google.com/drive/folders/1QsPIuOmi7NTKkBUyWGspQEVFaWufet1J)

## Training

1. **Single GPU Training**  
   To train the model on a single GPU, run:  
   ```bash
   python realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml --auto_resume
   ```

2. **Multi-GPU Training (4 GPUs)**  
   To train the model on 4 GPUs, run:  
   ```bash
   python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml --launcher pytorch --auto_resume
   ```
