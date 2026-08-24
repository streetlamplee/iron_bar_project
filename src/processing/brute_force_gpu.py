"""
warp point 4점을 조금씩 흔들어 보며 가장 잘 맞는 조합을 찾는 탐색 코드 (실험).

사람이 찍은 4점이 정확하지 않다는 전제에서, 각 점을 상하좌우로 offset 만큼 이동시킨
모든 조합을 시도하고 "여러 시점의 결과가 가장 잘 겹치는" 조합을 고른다.
offset을 절반씩 줄이며 재귀적으로 좁혀가는 coarse-to-fine 방식이다.

현재 메인 파이프라인에서는 사용하지 않는다.
(참고: import 경로가 상대 모듈명이라 그대로는 import 되지 않으며,
 find_cross_point_model 은 이 저장소의 legacy 폴더에 있다.)
"""

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
    4점을 offset 만큼 흔들어 본 조합 중 기준 이미지와 가장 잘 겹치는 warp를 찾는다.

    :param img_np: 점수 계산에 쓸 이미지 (교차점/철근 이미지)
    :param iron_img_np: 같은 변환을 함께 적용할 segmentation 이미지
    :param basis_np: 지금까지 누적된 기준 이미지. 이것과 잘 겹칠수록 점수가 높다.
    :param _2d_point_np: 현재 warp point 4점
    :param offset: 각 점을 흔들어 볼 픽셀 폭. 2의 거듭제곱이어야 하며 재귀할수록 절반이 된다.
    :param i: 지금까지 누적한 이미지 장수 (가중 평균 비율 계산에 사용)
    :param best_score: 상위 호출에서 넘어온 현재 최고 점수
    :return: (best_warped, best_seg_warped, best_score)
    '''
    # offset을 계속 2로 나눠 1이 되는지 확인 = 2의 거듭제곱인지 검사
    offset_test = offset
    while offset_test > 1.:
        offset_test /= 2.
    if offset_test != 1.:
        raise "The 'offset' argument must be a power of 2."

    # 이번 단계에서 점수가 좋았던 상위 top_k 후보만 다음(더 촘촘한) 단계로 넘긴다.
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
    선을 굵게 만들어두면 좌표가 조금 어긋나도 겹치는 픽셀이 생겨 점수가 완만하게 변한다.
    '''
    k_size = 5
    img_np = custom_blur(img_np)
    iron_img_np = custom_blur(iron_img_np)

    # 각 점의 x, y를 -offset / 0 / +offset 중 하나로 움직인다.
    # 점 4개 x 좌표 2개 = 8자리이므로 조합은 3^8 = 6561가지.
    # 전부 시도하면 느려서 그중 1/4만 무작위로 뽑아 본다.
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

        # 흔든 좌표로 실제 warp를 수행
        warped = warp_perspective(img_np, _2d_point_iter.float().detach().cpu().numpy())
        warped_iron = warp_perspective(iron_img_np, _2d_point_iter.float().detach().cpu().numpy())

        # 기존 누적 결과(basis)와 이번 결과를 장수 비율에 맞게 섞는다.
        basis_iter = basis_np * i / (i + 1)
        warped_scaled = warped * 1. / (i + 1)
        target_iter = basis_iter + warped_scaled

        # 점수 = 섞은 뒤에도 충분히 밝게 남은 픽셀 수.
        # 두 결과가 같은 위치에서 겹칠수록 밝은 픽셀이 많아지므로, 잘 정렬됐다는 뜻이 된다.
        score = np.count_nonzero(target_iter >= 255. / 2 * 1)

        next_target_2d_point = _2d_point_iter.detach().cpu().numpy()
        # 아직 더 촘촘히 볼 여지가 있으면(offset > 1), 좋은 후보를 다음 단계 대기열에 넣는다.
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


    # offset이 1이면 더 이상 좁힐 수 없으므로 여기서 결과를 확정한다.
    if offset == 1:
        if best_warped is not None and best_seg_warped is not None:
            return best_warped, best_seg_warped, best_score
        else:
            return None, None, -1

    else:
        # 상위 후보들에 대해 offset을 절반으로 줄여 더 촘촘하게 다시 탐색한다.
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
    brute_force_best_warp 와 같은 탐색이지만, 점수를 매기는 기준이 다르다.

    warp한 segmentation을 교차점 검출 모델에 넣어 나온 "교차점 이미지"가
    기준 이미지와 얼마나 겹치는지로 점수를 매긴다.
    즉 철근 선이 아니라 교차점 위치가 맞는지를 본다. 모델을 매 조합마다 돌리므로 훨씬 느리다.

    :param segment_np: warp 대상 segmentation 이미지
    :param basis_np: 누적 기준 이미지
    :param _2d_point_np: 현재 warp point 4점
    :param offset: 흔들어 볼 픽셀 폭 (2의 거듭제곱)
    :param i: 지금까지 누적한 이미지 장수
    :param best_score: 현재 최고 점수
    :return: (best_segment_warped, best_score)
    '''
    offset_test = offset
    while offset_test > 1.:
        offset_test /= 2.
    if offset_test != 1.:
        raise "The 'offset' argument must be a power of 2."

    # 모델 추론이 들어가 느리므로 다음 단계로 넘길 후보 수를 더 줄인다.
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
    # 위 함수보다 조합을 더 적게(1/8) 뽑는다.
    combos = random.sample(cand_combos, int(len(cand_combos) / 8))

    best_segment_warped = None
    best_combo = None
    for combo in tqdm(combos, desc=f"{"Searching with offset " + str(offset) :^30} | {"Start with Best Score: " + str(best_score):^30}"):
        offset_tensor = torch.tensor(combo, dtype=torch.float32, device=device).view(4, 2)
        _2d_point_iter = _2d_point + offset_tensor


        segment_warped = warp_perspective(segment_np, _2d_point_iter.float().detach().cpu().numpy())

        # 펴진 segmentation에서 교차점을 예측하고, 겹침 판정이 쉽도록 점을 굵게 만든다.
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