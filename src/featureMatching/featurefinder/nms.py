"""
교차점 예측 결과를 정리하는 후처리 (Non-Maximum Suppression).

모델이 격자 칸마다 예측하다 보니 같은 점을 여러 번 잡는다.
겹치는 예측을 걸러 점 하나당 하나만 남기는 것이 이 모듈의 역할이다.
"""

import torch
import math


def nms(input:torch.Tensor, image_size, threshold = 0.5):
    """
    같은 교차점을 가리키는 중복 예측을 하나만 남긴다.

    이웃한 격자 칸들이 같은 점을 각각 예측하는 일이 흔해서,
    가까운 점끼리 비교해 확신도(objectness)가 높은 쪽만 남긴다.

    :param threshold: 이 거리보다 가까우면 같은 점으로 본다
    :return: 살아남은 점 목록 [[x, y, objectness], ...]
    """
    keypoints = export_point(input, image_size, 4)
    # 지울 점을 바로 제거하지 않고 표시만 해둔다 (반복 중 목록이 바뀌지 않도록).
    need_to_del = [False] * len(keypoints)
    for i, keypoint in enumerate(keypoints):
        if need_to_del[i]:
            continue
        for j, target in enumerate(keypoints):
            if i == j or need_to_del[j]:
                continue
            # 두 점 사이 거리를 재서 같은 점인지 판단
            dist = math.dist(keypoint[:2], target[:2])
            if dist < threshold:
                key_objectness = keypoint[2]
                target_objectness = target[2]
                if key_objectness >= target_objectness:
                    need_to_del[j] = True
                else:
                    need_to_del[i] = True
                    break

    res = [k for k, d in zip(keypoints, need_to_del) if not d]

    return res

def export_point(tensor:torch.Tensor, image_size, max_points_per_cell = 4):
    """
    모델이 낸 격자 텐서를 실제 이미지 좌표 목록으로 되돌린다.

    :return: [[x, y, objectness], ...] (아직 중복이 걸러지지 않은 상태)
    """
    B, C, H, W = tensor.shape
    assert C == 12, f'Expected tensor got 12 channels, but tensor got {C} channels'
    grid_size = H
    stride = image_size // grid_size

    keypoints = []

    # 텐서 shape: (H, W, C)
    tensor = tensor.squeeze(0).permute(1, 2, 0)

    for gy in range(grid_size):
        for gx in range(grid_size):
            cell = tensor[gy, gx]  # shape: (C,)
            for i in range(max_points_per_cell):
                offset = i * 3
                rel_x = cell[offset + 0]
                rel_y = cell[offset + 1]
                objectness = cell[offset + 2]

                # 칸 번호(gx, gy)에 칸 안 상대 위치를 더하고 칸 크기를 곱해 실제 좌표로 환산한다.
                abs_x = min((gx + rel_x.item()) * stride, image_size - 1)
                abs_y = min((gy + rel_y.item()) * stride, image_size - 1)
                keypoints.append([abs_x, abs_y, float(objectness)])

    return keypoints