import os
import time
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.color import rgb2lab, deltaE_ciede2000
from baseline import *
from CAP import *
from MSRCR import *



def calculate_ciede2000(img1, img2):
    img1_lab = rgb2lab(img1)
    img2_lab = rgb2lab(img2)
    return np.mean(deltaE_ciede2000(img1_lab, img2_lab))

def evaluate_image_pairs(GT_path, hazy_path):
    metrics = {
        'DCP': {'MSE': [], 'PSNR': [], 'SSIM': [], 'CIEDE2000': [], 'Time': []},
        'MSR': {'MSE': [], 'PSNR': [], 'SSIM': [], 'CIEDE2000': [], 'Time': []},
        'CAP': {'MSE': [], 'PSNR': [], 'SSIM': [], 'CIEDE2000': [], 'Time': []}
    }

    GT_files = os.listdir(GT_path)
    hazy_files = os.listdir(hazy_path)
    
    for hazy_file in tqdm(hazy_files, desc="Processing images"):

        base_name = hazy_file.split('_')[0]
        corresponding_GT_files = [f for f in GT_files if f.startswith(base_name)]
        
        if corresponding_GT_files:
            GT_file = corresponding_GT_files[0]
            GT_image = cv2.imread(os.path.join(GT_path, GT_file)).astype(np.float64)
            hazy_image = cv2.imread(os.path.join(hazy_path, hazy_file)).astype(np.float64)
            
            for dehaze_func, key in zip([DCP, MSR, CAP], ['DCP', 'MSR', 'CAP']):
                start_time = time.time()
                if dehaze_func == MSR:
                    J = dehaze_func(hazy_image / 255) * 255
                else:
                    J = dehaze_func(hazy_image)
                end_time = time.time()
                
                mse = MSE(GT_image, J)
                psnr = PSNR(GT_image, J, data_range=255.0)
                ssim = SSIM(GT_image, J, channel_axis=2, data_range=255.0)
                ciede2000 = calculate_ciede2000(GT_image, J)
                
                metrics[key]['MSE'].append(mse)
                metrics[key]['PSNR'].append(psnr)
                metrics[key]['SSIM'].append(ssim)
                metrics[key]['CIEDE2000'].append(ciede2000)
                metrics[key]['Time'].append(end_time - start_time)
    
    return metrics

def save_metrics(metrics, output_file):
    with open(output_file, 'w') as f:
        for algorithm, alg_metrics in metrics.items():
            f.write(f"{algorithm} metrics:\n")
            for metric, values in alg_metrics.items():
                f.write(f"  {metric} values: {values}\n")
                f.write(f"  Average {metric}: {np.mean(values)}\n")
            f.write("\n")

if __name__ == '__main__':
    folder_path = './dataset/HazeRD'
    GT_path = os.path.join(folder_path, 'GT')
    hazy_path = os.path.join(folder_path, 'hazy')
    output_file = 'HazeRD_output_metrics.txt'
    
    metrics = evaluate_image_pairs(GT_path, hazy_path)
    save_metrics(metrics, output_file)