import cv2
import math
import numpy as np
import os
from guideda_filter import *
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

def dark_channel(im, sz):
    # 暗通道计算
    b, g, r = cv2.split(im)
    # 取每个像素位置RGB三通道中的最小值
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    # 进行最小值滤波
    dark = cv2.erode(dc, kernel)
    return dark


def atm_light(im, dark):
    # 计算全球大气光的值
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max((imsz // 1000), 1))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort(axis=0)
    indices = indices[imsz - numpx::]


    atmsum = np.zeros([1, 3])
    # 使用前0.1%的均值, 而不是最大值
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def transmission_estimate(im, A, sz):
    # 公式（12）
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * dark_channel(im3, sz)
    # tx = 0.1
    # transmission = cv2.max(transmission, tx)
    return transmission


def transmission_refine(im, et, radius, eps):
    im = np.uint8(im)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    t = guided_filter(gray, et, radius, eps)
    # t = fast_guided_filter(gray, et, radius, eps, 10)
    return t


def recover(im, t, A, tmax=0.1):
    # 公式(16)
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tmax)
    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]
    return res

def DCP(src):
    # 40
    dark_radius = 15
    dark_sz = 2 * dark_radius + 1
    # 80
    radius = 40
    eps = 1e-3
    tmax = 0.1

    I = src.astype('float64') / 255
    
    dark = dark_channel(I, dark_sz)
    A = atm_light(I, dark)
    te = transmission_estimate(I, A, dark_sz)
    t = transmission_refine(src, te, radius, eps)

    J = recover(I, t, A, tmax)
    return (J * 255).astype(np.float64)

if __name__ == '__main__':
    
    input_folder = "./dataset/O-HAZY/hazy"
    # input_folder = "./img"
    output_folder = "./dataset/O-HAZY/result"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # src = cv2.imread("img/234.png")
    src = cv2.imread("img/input5.jpg").astype(np.float64)
    src2 = cv2.imread("img/02_outdoor_GT.jpg").astype(np.float64)
    J = DCP(src)
    
    # print('原图和去雾图的MSE为{}'.format(MSE(src2, J)))
    # print('原图和去雾图的PSNR为{}'.format(PSNR(src2, J, data_range=255.0)))
    # print('原图和去雾图的SSIM为{}'.format(SSIM(src2, J, channel_axis=2, data_range=255.0)))

    # cv2.imshow('J', J)
    
    cv2.imwrite("J11.jpg", J)
    

    cv2.waitKey()