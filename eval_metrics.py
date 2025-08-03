import os
import cv2
import argparse
from basicsr.metrics import calculate_psnr, calculate_ssim

def evaluate_metrics(gt_dir, sr_dir):
    filenames = sorted(os.listdir(gt_dir))
    psnr_values = []
    ssim_values = []

    for filename in filenames:
        gt_path = os.path.join(gt_dir, filename)
        base, ext = os.path.splitext(filename)
        sr_path = os.path.join(sr_dir, base + "_out" + ext)

        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)

        if gt_img is None or sr_img is None:
            print(f"Skipping {filename}: image not found.")
            continue

        psnr = calculate_psnr(gt_img, sr_img, crop_border=4, test_y_channel=False)
        ssim = calculate_ssim(gt_img, sr_img, crop_border=4, test_y_channel=False)

        psnr_values.append(psnr)
        ssim_values.append(ssim)

    if psnr_values and ssim_values:
        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        print(f'Average PSNR: {avg_psnr:.4f} dB')
        print(f'Average SSIM: {avg_ssim:.4f}')
    else:
        print("No valid images found for evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PSNR and SSIM between ground-truth and SR images.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground-truth HR images.")
    parser.add_argument("--sr_dir", type=str, required=True, help="Directory containing SR images.")

    args = parser.parse_args()
    evaluate_metrics(args.gt_dir, args_sr_dir)
