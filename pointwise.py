import numpy as np
import cv2
import torch
from setuptools.command.build_ext import if_dl
from torch.onnx.symbolic_opset9 import unsqueeze


def find_cross_point(img:np.ndarray, dilate:int = 1):
    if not torch.cuda.is_available():
        raise "cuda is not available"
    kernel = np.zeros(shape=(2 * dilate + 1, 2 * dilate + 1), dtype = np.float32)
    kernel[dilate-1:dilate+2, :] = 1.
    kernel[:, dilate-1:dilate+2] = 1.

    denominator = np.count_nonzero(kernel)

    kernel = torch.from_numpy(kernel)

    device = 'cuda'

    img_tensor = list()

    height, width = img.shape
    img_pad = np.ones(shape=(height + dilate * 2, width + dilate * 2), dtype = np.float32)
    img_pad = img_pad * 255
    img_pad[dilate:dilate+height, dilate:dilate+width] = img

    for h in range(dilate, height + dilate):
        img_tensor_sub = []
        for w in range(dilate, width + dilate):
            img_tensor_sub.append(img_pad[h-dilate:h+dilate+1, w-dilate:w+dilate+1])
        img_tensor.append(img_tensor_sub)
    img_tensor = np.array(img_tensor, dtype = np.float32)
    img_tensor = torch.from_numpy(img_tensor)

    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(img_tensor.shape[0], img_tensor.shape[1], 1, 1)

    img_tensor.to(device)
    kernel.to(device)

    output = img_tensor * kernel

    output = output.detach().cpu().numpy()

    numerator = np.sum(output, axis=(-2, -1))

    res = numerator / denominator

    res = np.where(res > 250, 255, 0)

    res = res.astype(np.uint8)

    return res

def make_point_match(origin_img_arr:np.ndarray, img_arr:np.ndarray):
    '''
    param : img_arr : (N x H x W) shape array
    '''

    basis_image = img_arr[0]
    basis_image = torch.from_numpy(basis_image)
    basis_image = basis_image / 255

    basis_image_origin = origin_img_arr[0]

    for i in range(1, img_arr.shape[0]):
        img_iter = img_arr[i]
        img_iter = torch.from_numpy(img_iter)
        img_iter = img_iter / 255
        h, w = img_iter.shape

        origin_img_iter = origin_img_arr[i]

        candidate_list = []
        origin_candidate_list = []

        for x in range(-10, 11):
            for y in range(-10, 11):
                candidate = np.zeros_like(img_iter)
                origin_candidate = np.zeros_like(img_iter)
                x_start = max(0, x)
                x_end = min(h, h + x)
                y_start = max(0, y)
                y_end = min(w, w + y)

                src_x_start = max(0, -x)
                src_x_end = src_x_start + (x_end - x_start)
                src_y_start = max(0, -y)
                src_y_end = src_y_start + (y_end - y_start)

                candidate[x_start:x_end, y_start:y_end] = img_iter[src_x_start:src_x_end, src_y_start:src_y_end]
                origin_candidate[x_start:x_end, y_start:y_end] = origin_img_iter[src_x_start:src_x_end, src_y_start:src_y_end]

                candidate_list.append(candidate)
                origin_candidate_list.append(origin_candidate)

        candidate_list = np.array(candidate_list, dtype = np.uint8)
        candidate_tensor = torch.from_numpy(candidate_list)
        basis_image_iter = basis_image
        basis_image_iter = basis_image_iter.unsqueeze(0).repeat(len(candidate_list), 1, 1)

        output = candidate_tensor * basis_image_iter
        output_idx_count_nonzero = torch.argmax(torch.count_nonzero(output, dim= (-2, -1)))
        basis_image = candidate_tensor[output_idx_count_nonzero] * 1 / (i+1) + basis_image * i / (i+1)
        basis_image_origin = origin_candidate_list[output_idx_count_nonzero] * 1 / (i+1) + basis_image_origin * i / (i+1)

        # cv2.imshow('basis', (basis_image * 255).numpy().astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyWindow('basis')
        #
        # cv2.imshow('origin', basis_image_origin.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyWindow('origin')

    return (basis_image * 255).numpy().astype(np.uint8), basis_image_origin.astype(np.uint8)


if __name__ == "__main__":
    '''
    Do not run the code below except debugging
    '''
    img1 = cv2.imread('/home/user/PycharmProjects/iron_bar_sample_project/warp_image/1.png')
    img2 = cv2.imread('/home/user/PycharmProjects/iron_bar_sample_project/warp_image/2.png')
    img3 = cv2.imread('/home/user/PycharmProjects/iron_bar_sample_project/warp_image/3.png')
    img4 = cv2.imread('/home/user/PycharmProjects/iron_bar_sample_project/warp_image/4.png')
    img5 = cv2.imread('/home/user/PycharmProjects/iron_bar_sample_project/warp_image/5.png')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)

    input = [find_cross_point(img1, 15),
             find_cross_point(img2, 15),
             find_cross_point(img3, 15),
             find_cross_point(img4, 15),
             find_cross_point(img5, 15),]

    input = np.array(input)

    original_img = np.array([img1, img2, img3, img4, img5])

    point_matched_image, result = make_point_match(original_img, input)

    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyWindow('result')

    result_thres = np.where(result > (255. / len(input) * (len(input) - 2)), 255, 0)
    cv2.imshow('thres', result_thres.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyWindow('thres')