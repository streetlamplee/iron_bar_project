import random
from collections import deque
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import itertools
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from blur import custom_blur
from warp import warp_perspective
from find_cross_point_model.predict import predict_one_image

# --- MAIN BRUTE-FORCE WARPING ---
def brute_force_best_warp(img_np:np.ndarray, iron_img_np:np.ndarray, basis_np, _2d_point_np, offset:int=2, i=0, best_score = -1):
    '''

    '''
    offset_test = offset
    while offset_test > 1.:
        offset_test /= 2.
    if offset_test != 1.:
        raise "The 'offset' argument must be a power of 2."

    top_k = 3
    top_k_queue = deque()
    best_warp = None
    best_seg_warp = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # img_np_copy = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    # iron_img_np_copy = cv2.cvtColor(iron_img_np, cv2.COLOR_GRAY2BGR)
    #
    # for p in range(len(_2d_point_np)):
    #     cv2.circle(img_np_copy, _2d_point_np[p].astype(np.int32).tolist(), radius=5, thickness=-1, color=(0, 0, 255))
    #     cv2.circle(iron_img_np_copy, _2d_point_np[p].astype(np.int32).tolist(), radius=5, thickness=-1, color=(0, 0, 255))
    #
    # cv2.imshow('point_img', img_np_copy)
    # cv2.imshow('line_img', iron_img_np_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 준비: 이미지 및 기준점
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(device)  # [1, 1, H, W]
    iron_img_tensor = torch.from_numpy(iron_img_np).unsqueeze(0).unsqueeze(0).float().to(device)
    _2d_point = torch.from_numpy(_2d_point_np).to(device).float()
    basis_image = torch.from_numpy(basis_np).to(device).float()

    '''
    250514 blur 처리해서 해보기
    '''
    k_size = 5
    img_np = custom_blur(img_np)
    iron_img_np = custom_blur(iron_img_np)

    r = range(-offset, offset + 1, offset)
    cand_combos = list(itertools.product(r, repeat=8))
    combos = random.sample(cand_combos, int(len(cand_combos) / 4))

    best_warped = None
    best_seg_warped = None
    best_combo = None
    for combo in tqdm(combos, desc=f"{"Searching with offset " + str(offset) :^30} | {"Start with Best Score: " + str(best_score):^30}"):
        offset_tensor = torch.tensor(combo, dtype=torch.float32, device=device).view(4, 2)
        _2d_point_iter = _2d_point + offset_tensor

        # startpoint = _2d_point_iter.int().tolist()
        # endpoint = [[0,0],[1024,0],[1024,1024],[0,1024]]
        # warped = VF.perspective(img_tensor, startpoint, endpoint)
        # warped_iron = VF.perspective(iron_img_tensor, startpoint, endpoint)
        #
        #
        # #this iter warped point image
        # warped = F.interpolate(warped, size = (1024, 1024), mode = 'bilinear', align_corners=False)
        #
        # #this iter warped segment image
        # warped_iron = F.interpolate(warped_iron, size= (1024, 1024), mode = 'bilinear', align_corners=False)

        warped = warp_perspective(img_np, _2d_point_iter.float().detach().cpu().numpy())
        warped_iron = warp_perspective(iron_img_np, _2d_point_iter.float().detach().cpu().numpy())

        basis_iter = basis_np * i / (i + 1)
        warped_scaled = warped * 1. / (i + 1)
        target_iter = basis_iter + warped_scaled

        score = np.count_nonzero(target_iter >= 255. / 2 * 1)

        next_target_2d_point = _2d_point_iter.detach().cpu().numpy()
        if offset != 1 and score > best_score:
            top_k_queue.appendleft((img_np, iron_img_np, basis_np, next_target_2d_point, int(offset/2), i, score))
            if len(top_k_queue) > top_k:
                top_k_queue.pop()
        if score > best_score:
            best_warped = warped
            best_seg_warped = warped_iron
            best_combo = combo
            best_score = score


    tqdm.write(f"In this Recursive, best score is {best_score}")


    if offset == 1:
        if best_warped is not None and best_seg_warped is not None:
            return best_warped, best_seg_warped, best_score
        else:
            return None, None, -1

    else:
        while top_k_queue:
            args = top_k_queue.pop()
            cand_warp, cand_seg_warp, cand_score = brute_force_best_warp(*args)

            if cand_score > best_score:
                best_warped = cand_warp
                best_seg_warped = cand_seg_warp
                best_score = cand_score

        return best_warped, best_seg_warped, best_score


def brute_force_with_pointFinder(segment_np:np.ndarray, basis_np, _2d_point_np, offset:int=2, i=0, best_score = -1):
    '''
    segment_np >>
    '''
    offset_test = offset
    while offset_test > 1.:
        offset_test /= 2.
    if offset_test != 1.:
        raise "The 'offset' argument must be a power of 2."

    top_k = 2
    top_k_queue = deque()
    best_warp = None
    best_seg_warp = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # img_np_copy = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    # iron_img_np_copy = cv2.cvtColor(iron_img_np, cv2.COLOR_GRAY2BGR)
    #
    # for p in range(len(_2d_point_np)):
    #     cv2.circle(img_np_copy, _2d_point_np[p].astype(np.int32).tolist(), radius=5, thickness=-1, color=(0, 0, 255))
    #     cv2.circle(iron_img_np_copy, _2d_point_np[p].astype(np.int32).tolist(), radius=5, thickness=-1, color=(0, 0, 255))
    #
    # cv2.imshow('point_img', img_np_copy)
    # cv2.imshow('line_img', iron_img_np_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 준비: 이미지 및 기준점
    _2d_point = torch.from_numpy(_2d_point_np).to(device).float()

    '''
    250514 blur 처리해서 해보기
    '''

    r = range(-offset, offset + 1, offset)
    cand_combos = list(itertools.product(r, repeat=8))
    combos = random.sample(cand_combos, int(len(cand_combos) / 8))

    best_segment_warped = None
    best_combo = None
    for combo in tqdm(combos, desc=f"{"Searching with offset " + str(offset) :^30} | {"Start with Best Score: " + str(best_score):^30}"):
        offset_tensor = torch.tensor(combo, dtype=torch.float32, device=device).view(4, 2)
        _2d_point_iter = _2d_point + offset_tensor


        segment_warped = warp_perspective(segment_np, _2d_point_iter.float().detach().cpu().numpy())

        warped_point_image = predict_one_image(cv2.cvtColor(segment_warped, cv2.COLOR_GRAY2RGB))
        warped_point_image = custom_blur(warped_point_image)

        basis_iter = basis_np * i / (i + 1)
        warped_scaled = warped_point_image * 1. / (i + 1)
        target_iter = basis_iter + warped_scaled

        score = np.count_nonzero(target_iter >= 255. / 2 * 1)

        next_target_2d_point = _2d_point_iter.detach().cpu().numpy()
        if offset != 1 and score > best_score:
            top_k_queue.appendleft((segment_warped, basis_np, next_target_2d_point, int(offset / 2), i, score))
            if len(top_k_queue) > top_k:
                top_k_queue.pop()
        if score > best_score:
            best_segment_warped = segment_warped
            best_combo = combo
            best_score = score


    tqdm.write(f"In this Recursive, best score is {best_score}")


    if offset == 1:
        if best_segment_warped is not None:
            return best_segment_warped, best_score
        else:
            return None, -1

    else:
        while top_k_queue:
            args = top_k_queue.pop()
            cand_seg_warp, cand_score = brute_force_with_pointFinder(*args)

            if cand_score > best_score:
                best_segment_warped = cand_seg_warp
                best_score = cand_score

        return best_segment_warped, best_score