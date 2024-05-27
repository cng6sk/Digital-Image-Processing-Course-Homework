import cv2
import math
import numpy as np
import os
from guideda_filter import *
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR


def calVSMap(I, r):
    I = np.uint8(I)
    hsvI = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    s = hsvI[:, :, 1] / 255.0
    v = hsvI[:, :, 2] / 255.0
    sigma = 0.041337
    sigmaMat = np.random.normal(0, sigma, I.shape[:2])

    output = 0.121779 + 0.959710 * v - 0.780245 * s + sigmaMat
    outputPixel = output
    output = cv2.erode(output, np.ones((r, r)))
    return output, outputPixel

def estA(img, Jdark):
    h, w, _ = img.shape
    img = img / 255.0 if np.max(img) > 1 else img

    n_bright = int(np.ceil(0.001 * h * w))
    Loc = np.argsort(Jdark, axis=None)[-n_bright:]
    Ics = img.reshape(-1, 3)
    
    Acand = Ics[Loc]
    Amag = np.linalg.norm(Acand, axis=1)
    
    Loc2 = np.argsort(Amag)
    if len(Loc2) > 20:
        A = Acand[Loc2[-20:]].max(axis=0)
    else:
        A = Acand[Loc2].max(axis=0)
    
    return A

def CAP(src):
    r = 20
    beta = 1.0
    gimfiltR = 180
    eps = 1e-3

    dR, dP = calVSMap(src, r)
    refineDR = fast_guided_filter_color(src.astype(np.float64) / 255, dP, r, eps, r // 4)

    tR = np.exp(-beta * refineDR)
    tP = np.exp(-beta * dP)

    # cv2.imshow('res/originalDepthMap.png', dR * 255)
    # cv2.imshow('res/refineDepthMap.png', refineDR * 255)

    a = estA(src, dR)
    t0 = 0.05
    t1 = 1
    I = src.astype(np.float64) / 255
    h, w, c = I.shape
    J = np.zeros_like(I)

    J[:, :, 0] = I[:, :, 0] - a[0]
    J[:, :, 1] = I[:, :, 1] - a[1]
    J[:, :, 2] = I[:, :, 2] - a[2]

    t = tR.copy()
    t[t < t0] = t0
    t[t > t1] = t1

    J[:, :, 0] /= t
    J[:, :, 1] /= t
    J[:, :, 2] /= t

    J[:, :, 0] += a[0]
    J[:, :, 1] += a[1]
    J[:, :, 2] += a[2]

    return (J * 255).astype(np.float64)


if __name__ == '__main__':
    
    input_folder = "./dataset/O-HAZY/hazy"
    # input_folder = "./img"
    output_folder = "./dataset/O-HAZY/result"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # src = cv2.imread("img/input7.png")
    src = cv2.imread("img/input5.jpg").astype(np.float64)
    src2 = cv2.imread("img/02_outdoor_GT.jpg").astype(np.float64)
    J = CAP(src)
    
    # print('原图和去雾图的MSE为{}'.format(MSE(src2, J)))
    # print('原图和去雾图的PSNR为{}'.format(PSNR(src2, J, data_range=255.0)))
    # print('原图和去雾图的SSIM为{}'.format(SSIM(src2, J, channel_axis=2, data_range=255.0)))

    # cv2.imshow('J', J)
    
    cv2.imwrite("J12.jpg", J)

    cv2.waitKey()