import cv2
import math
import numpy as np




def guided_filter(im, p, radius, eps):
    r = 2 * radius + 1
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q

def boxfilter(img, r):
    return cv2.blur(img, (r, r))

def fast_guided_filter(I, p, r, eps, s):
    I_sub = cv2.resize(I, (0, 0), fx=1/s, fy=1/s, interpolation=cv2.INTER_NEAREST)
    p_sub = cv2.resize(p, (0, 0), fx=1/s, fy=1/s, interpolation=cv2.INTER_NEAREST)
    r_sub = int(r / s)

    N = boxfilter(np.ones_like(I_sub), r_sub)

    mean_I = boxfilter(I_sub, r_sub) / N
    mean_p = boxfilter(p_sub, r_sub) / N
    mean_Ip = boxfilter(I_sub * p_sub, r_sub) / N
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = boxfilter(I_sub * I_sub, r_sub) / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = boxfilter(a, r_sub) / N
    mean_b = boxfilter(b, r_sub) / N

    mean_a = cv2.resize(mean_a, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_b, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_LINEAR)

    q = mean_a * I + mean_b
    return q


def fast_guided_filter_color(I, p, r, eps, s):
    # 下采样
    I_sub = cv2.resize(I, (0, 0), fx=1/s, fy=1/s, interpolation=cv2.INTER_NEAREST)
    p_sub = cv2.resize(p, (0, 0), fx=1/s, fy=1/s, interpolation=cv2.INTER_NEAREST)
    r_sub = int(r / s)

    hei, wid = p_sub.shape
    N = boxfilter(np.ones((hei, wid)), r_sub)

    mean_I_r = boxfilter(I_sub[:, :, 0], r_sub) / N
    mean_I_g = boxfilter(I_sub[:, :, 1], r_sub) / N
    mean_I_b = boxfilter(I_sub[:, :, 2], r_sub) / N

    mean_p = boxfilter(p_sub, r_sub) / N

    mean_Ip_r = boxfilter(I_sub[:, :, 0] * p_sub, r_sub) / N
    mean_Ip_g = boxfilter(I_sub[:, :, 1] * p_sub, r_sub) / N
    mean_Ip_b = boxfilter(I_sub[:, :, 2] * p_sub, r_sub) / N

    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p

    var_I_rr = boxfilter(I_sub[:, :, 0] * I_sub[:, :, 0], r_sub) / N - mean_I_r * mean_I_r
    var_I_rg = boxfilter(I_sub[:, :, 0] * I_sub[:, :, 1], r_sub) / N - mean_I_r * mean_I_g
    var_I_rb = boxfilter(I_sub[:, :, 0] * I_sub[:, :, 2], r_sub) / N - mean_I_r * mean_I_b
    var_I_gg = boxfilter(I_sub[:, :, 1] * I_sub[:, :, 1], r_sub) / N - mean_I_g * mean_I_g
    var_I_gb = boxfilter(I_sub[:, :, 1] * I_sub[:, :, 2], r_sub) / N - mean_I_g * mean_I_b
    var_I_bb = boxfilter(I_sub[:, :, 2] * I_sub[:, :, 2], r_sub) / N - mean_I_b * mean_I_b

    a = np.zeros((hei, wid, 3))
    for y in range(hei):
        for x in range(wid):
            Sigma = np.array([[var_I_rr[y, x], var_I_rg[y, x], var_I_rb[y, x]],
                              [var_I_rg[y, x], var_I_gg[y, x], var_I_gb[y, x]],
                              [var_I_rb[y, x], var_I_gb[y, x], var_I_bb[y, x]]])
            
            cov_Ip = np.array([cov_Ip_r[y, x], cov_Ip_g[y, x], cov_Ip_b[y, x]])
            a[y, x, :] = np.linalg.inv(Sigma + eps * np.eye(3)).dot(cov_Ip)

    b = mean_p - a[:, :, 0] * mean_I_r - a[:, :, 1] * mean_I_g - a[:, :, 2] * mean_I_b

    mean_a_r = boxfilter(a[:, :, 0], r_sub) / N
    mean_a_g = boxfilter(a[:, :, 1], r_sub) / N
    mean_a_b = boxfilter(a[:, :, 2], r_sub) / N
    mean_b = boxfilter(b, r_sub) / N

    mean_a_r = cv2.resize(mean_a_r, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_LINEAR)
    mean_a_g = cv2.resize(mean_a_g, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_LINEAR)
    mean_a_b = cv2.resize(mean_a_b, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_b, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_LINEAR)

    q = mean_a_r * I[:, :, 0] + mean_a_g * I[:, :, 1] + mean_a_b * I[:, :, 2] + mean_b
    return q