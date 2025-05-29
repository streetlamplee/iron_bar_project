from typing import Tuple

import cv2
import numpy as np
import os
from collections import deque
import torch
import itertools
import random
from tqdm import tqdm
import gc
from n_warp import warp_perspective
from n_blur import custom_blur
from find_cross_point_model.predict import predict_one_image
from extension import image_show

def stock_image(image_folder_path:str, warp_point):
    image_list = os.listdir(image_folder_path)
    res_warp = None
    res_point = None
    image_list.sort()
    for i, image_filename in enumerate(image_list):
        image = cv2.imread(os.path.join(image_folder_path, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1360,1020))

        if i == 0:
            '''
            첫 이미지는 BF를 하지 않고, 점과 warp만 수행하여, 이후 이미지들의 기준으로 삼는다.
            '''
            warp_image = warp_perspective(image, np.array(warp_point[str(i)], dtype = np.float32)) # warp만 한 이미지
            # image_show(image)
            # image_show(warp_image)
            warp_only_point, cross_points = predict_one_image(warp_image)
            print(cross_points[:100])
            warp_only_point = custom_blur(warp_only_point)
            # image_show(warp_only_point)
            res_warp = warp_image
            res_point = warp_only_point

            continue

        alpha_blending_image, alpha_blending_point_image, _ = stock_step(image, warp_only_point, np.array(warp_point[str(i)], dtype = np.float32), 8, -1, i)

        res_warp = res_warp * i / (i + 1) + alpha_blending_image * 1 / (i + 1)
        res_point = res_point * i / (i + 1) + alpha_blending_point_image * 1 / (i + 1)

        cv2.imwrite(f'./250529/250529_result_warp{i}.png', res_warp)
        cv2.imwrite(f'./250529/250529_result_point{i}.png', res_point)


    return res_warp, res_point

def stock_step(image, basis_image, warp_point, offset, best_score, idx):
    '''
    param:
    image : warp 하기 전, 철근만 뽑아낸 이미지
    basis_image : 이번 단계에서 비교하기 위한 이미지
    warp_point : offset = 0일 때 warp 하는 점들의 집합
    offset : 주고자 하는 offset, 2의 제곱수여야 함
    score : 현재 단계에서의 점수 값

    return :
    1. 최적의 offset을 적용한 철근이 뽑혀있는 이미지 (alpha blending)
    2. 최적의 offset을 적용한 point의 이미지 (alpha blending)
    3. 현재 score
    '''

    '''
    GPU 캐시 삭제 및 최적화
    '''
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    '''
    offset 확인
    '''
    offset_test = offset

    while offset_test > 1.:
        offset_test = offset_test / 2.
    if offset_test != 1.:
        raise "offset args is not a power of 2"

    '''
    GPU 연산에 필요한 요소 선언
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    top_k = 3
    top_k_queue = deque()

    '''
    make warp point and sampling 
    '''
    warp_point = torch.from_numpy(warp_point).float().to(device)
    # r = range(-offset, offset+1, offset)
    r = range(0, 8)
    combos = list(itertools.product(r, repeat = 8))
    sampled_combes = random.sample(combos, int(len(combos) / 64))

    for combo in tqdm(sampled_combes, desc=f'Searcing offset {offset}'):
        offset_tensor = torch.tensor(combo, dtype = torch.float32, device = device)
        perturbed_point = warp_point + offset_tensor.view(4,2)

        '''
        make warp image and pointfinder model output image
        '''
        # image_show(image)
        warp_image = warp_perspective(image, perturbed_point.cpu().numpy()) # 현재 offset으로 warp 한 이미지
        # image_show(warp_image)
        warp_point_image, cross_points = predict_one_image(warp_image)                    # 현재 offset으로 warp 한 이미지의 교차점만
        warp_point_image = custom_blur(warp_point_image)
        print(cross_points[:100])
        current_score = score(basis_image, warp_point_image, (idx / (idx + 1), 1 / (idx + 1)))

        if offset != 1 and current_score > best_score:
            top_k_queue.appendleft((image, basis_image, perturbed_point.cpu().numpy(), int(offset/2), current_score, idx))
            if len(top_k_queue) > top_k:
                top_k_queue.pop()

        if current_score > best_score:
            best_score = current_score
            best_warped = warp_image
            best_point_image = warp_point_image

    # image_show(best_point_image)
    # image_show(best_warped)

    if offset == 1:
        return best_warped, best_point_image, best_score
    else:
        while top_k_queue:
            args = top_k_queue.pop()
            next_warped, next_point_image, next_score = stock_step(*args)
            if next_score > best_score:
                best_score = next_score
                best_warped = next_warped
                best_point_image = next_point_image
        return best_warped, best_point_image, best_score

def score(image1, image2, weight:Tuple[float, float]):
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY).astype(np.float32)

    sum = weight[0] * image1 + weight[1] * image2

    res = np.count_nonzero(sum > (255. * 1 / 2))

    return res

