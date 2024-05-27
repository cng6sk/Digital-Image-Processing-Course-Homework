import numpy as np
import cv2
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_adapthist
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

def adapthisteq(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # 对每个通道分别应用CLAHE
    if len(img.shape) == 2:  # 灰度图像
        return clahe.apply(img)
    elif len(img.shape) == 3:  # 彩色图像
        channels = cv2.split(img)
        eq_channels = [clahe.apply(channel) for channel in channels]
        return cv2.merge(eq_channels)
    else:
        raise ValueError("图像格式不支持")

def MSR(input):
    I = input.astype('float64')

    # 分离输入图像的RGB通道
    B = I[:, :, 0]
    G = I[:, :, 1]
    R = I[:, :, 2]
    R0 = R.astype('float64')
    G0 = G.astype('float64')
    B0 = B.astype('float64')

    # 获取图像尺寸
    N1, M1 = R.shape

    # 对每个颜色通道进行多尺度Retinex
    def MSR_channel(channel, scales):
        channel_log = np.log(channel + 1)
        channel_fft2 = np.fft.fft2(channel)
        retinex = np.zeros_like(channel)

        for sigma in scales:
            # 高斯滤波
            F = np.zeros((N1, M1))
            F[N1//2, M1//2] = 1
            F = gaussian_filter(F, sigma)
            Efft = np.fft.fft2(F)
            DR0 = channel_fft2 * Efft
            DR = np.fft.ifft2(DR0)
            DRlog = np.log(np.abs(DR) + 1)
            retinex += channel_log - DRlog

        retinex /= len(scales)
        return retinex

    scales = [80, 128, 256]
    Rr = MSR_channel(R0, scales)
    Gr = MSR_channel(G0, scales)
    Br = MSR_channel(B0, scales)

    # 颜色恢复因子
    a = 125
    II = R0 + G0 + B0

    def color_recovery(channel, retinex, a, II):
        I_channel = channel * a
        C = I_channel / II
        C = np.log(C + 1)
        retinex = C * retinex
        EXPR_channel = np.exp(retinex)

        # 灰度拉伸
        MIN = EXPR_channel.min()
        MAX = EXPR_channel.max()
        EXPR_channel = (EXPR_channel - MIN) / (MAX - MIN)
        height, width = channel.shape[:2]
        EXPR_channel = equalize_adapthist(EXPR_channel, kernel_size=[height // 8, width // 8])
        return EXPR_channel

    EXPRr = color_recovery(R0, Rr, a, II)
    EXPGg = color_recovery(G0, Gr, a, II)
    EXPBb = color_recovery(B0, Br, a, II)

    # 增强亮度
    EXPRr *= 1.1
    EXPGg *= 1.1
    EXPBb *= 1.1

    J = np.stack([EXPBb, EXPGg, EXPRr], axis=2)
    return J


def MSRCR(input1):
    input1 = input1.astype('float64') / 255
    input2 = MSR(input1)
    # 将输入图像和MSR处理后的图像分离成RGB通道
    Ir = input1[:, :, 0]
    Ig = input1[:, :, 1]
    Ib = input1[:, :, 2]
    Rr = input2[:, :, 0]
    Gg = input2[:, :, 1]
    Bb = input2[:, :, 2]

    # 调用颜色恢复函数
    MSRCR_r, MSRCR_g, MSRCR_b = MSRCR_rgb(Ir, Ig, Ib, Rr, Gg, Bb)

    # 将处理后的RGB通道合并
    output = np.stack([MSRCR_r, MSRCR_g, MSRCR_b], axis=2)
    return output

def MSRCR_rgb(Ir, Ig, Ib, Rr, Gg, Bb):
    G = 192  # 常数G
    b = -30  # 常数b
    alpha = 125  # 常数alpha
    beta = 46  # 常数beta

    def scale_and_convert(channel):
        # 灰度拉伸并转换为8位图像
        min_val = np.min(channel)
        max_val = np.max(channel)
        output = 255 * (channel - min_val) / (max_val - min_val)
        output = np.clip(output, 0, 255)
        return output.astype(np.uint8)

    # 颜色恢复处理
    CRr = beta * (np.log(alpha * Ir + 1) - np.log(Ir + Ig + Ib + 1))
    Rr = G * (CRr * Rr + b)
    outputR = scale_and_convert(Rr)

    CGg = beta * (np.log(alpha * Ig + 1) - np.log(Ir + Ig + Ib + 1))
    Gg = G * (CGg * Gg + b)
    outputG = scale_and_convert(Gg)

    CBb = beta * (np.log(alpha * Ib + 1) - np.log(Ir + Ig + Ib + 1))
    Bb = G * (CBb * Bb + b)
    outputB = scale_and_convert(Bb)

    return outputR, outputG, outputB





if __name__ == '__main__':
    
    input_folder = "./dataset/O-HAZY/hazy"
    # input_folder = "./img"
    output_folder = "./dataset/O-HAZY/result"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # src = cv2.imread("img/234.png")
    src = cv2.imread("img/input6.jpg").astype(np.float64)
    src2 = cv2.imread("img/01_outdoor_GT.jpg").astype(np.float64)
    

    J = MSR(src / 255)
    JJ = J * 255
    
    # print('原图和去雾图的MSE为{}'.format(MSE(src2, JJ)))
    # print('原图和去雾图的PSNR为{}'.format(PSNR(src2, JJ, data_range=255.0)))
    # print('原图和去雾图的SSIM为{}'.format(SSIM(src2, JJ, channel_axis=2, data_range=255.0)))

    # cv2.imshow('J', J)
    
    # cv2.imwrite("J13.jpg", J)
    cv2.imwrite("J13.jpg", JJ)

    cv2.waitKey()