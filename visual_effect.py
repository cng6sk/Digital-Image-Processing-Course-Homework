from baseline import *
from CAP import *
from MSRCR import *


if __name__ == '__main__':
    
    input_folder = "./dataset/O-HAZY/hazy"
    # input_folder = "./img"
    output_folder = "./dataset/O-HAZY/result"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    src = cv2.imread("img/a2.jpg")
    # src = cv2.imread("dataset/real_world/10.jpg").astype(np.float64)
    # src2 = cv2.imread("img/01_outdoor_GT.jpg").astype(np.float64)

    J1 = DCP(src)

    J2 = CAP(src)

    J3 = MSR(src / 255)
    J3 = J3 * 255
    
    # print('原图和去雾图的MSE为{}'.format(MSE(src2, JJ)))
    # print('原图和去雾图的PSNR为{}'.format(PSNR(src2, JJ, data_range=255.0)))
    # print('原图和去雾图的SSIM为{}'.format(SSIM(src2, JJ, channel_axis=2, data_range=255.0)))

    # cv2.imshow('J', J)
    
    # cv2.imwrite("J13.jpg", J)
    cv2.imwrite("a1.jpg", J1)
    cv2.imwrite("a2.jpg", J2)
    cv2.imwrite("a3.jpg", J3)

    cv2.waitKey()