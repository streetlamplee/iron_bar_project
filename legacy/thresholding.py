import cv2
import numpy as np
import torch

import extension


def thresholding(img:np.ndarray, d = 3):
    value = np.sum(np.where(img > 255 / d, img, 0)) / np.count_nonzero(img > 255 / d)
    res = np.where(img > value, img, 0)
    res = res.astype(np.uint8)

    return res

def refine(img:np.ndarray, iteration = 1):
    k_size = 7

    kernel = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0]], dtype = np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    img = cv2.morphologyEx(src=img, op=cv2.MORPH_OPEN, kernel = kernel, iterations=3)
    extension.image_show(img, 'open')
    # img = cv2.morphologyEx(src=img, op=cv2.MORPH_CLOSE, kernel = kernel, iterations=1)
    # extension.image_show(img, 'close')
    # img = cv2.erode(img, (k_size, k_size), iterations= iteration)
    # extension.image_show(img, f'erode')


    # img = cv2.dilate(img, (k_size, k_size), iterations = iteration // 2)
    # extension.image_show(img, f'dilate')
    # img = cv2.morphologyEx(src=img, op=cv2.MORPH_OPEN, kernel= kernel, iterations=iteration)
    # extension.image_show(img, 'open')

    return img

def refine_(img:np.ndarray):
    one_side_size = 10
    kernel = np.zeros(shape=(one_side_size*2+1,one_side_size*2+1), dtype = np.float32)
    for i in range(0, one_side_size*2+1):
        if i % 2 == 0:
            kernel[one_side_size, i] = 1.
            kernel[i, one_side_size] = 1.

    H, W = img.shape
    kH, kW = kernel.shape
    cH, cW = kH // 2, kW // 2

    res = np.zeros_like(img, dtype = np.float32)

    for i in range(H):
        for j in range(W):
            i_start = max(i - cH, 0)
            i_end = min(i + cH, H)
            j_start = max(j - cW, 0)
            j_end = min(j + cW, W)

            k_i_start = cH - (i - i_start)
            k_i_end = cH + (i_end - i)
            k_j_start = cW - (j - j_start)
            k_j_end = cW + (j_end - j)

            img_patch = img[i_start:i_end, j_start:j_end]
            kernel_patch = kernel[k_i_start:k_i_end, k_j_start:k_j_end]

            res[i, j] = np.sum(img_patch * kernel_patch)

    res.astype(np.uint8)
    return res




if __name__ == '__main__':
    img = cv2.imread('res_iron_pointwise.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # extension.image_show(img)
    img = thresholding(img)
    extension.image_show(img)
    img = refine(img, 5)
    # img = thresholding(img, 10)

    extension.image_show(img)