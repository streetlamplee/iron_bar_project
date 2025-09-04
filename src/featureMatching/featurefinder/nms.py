import torch
import math


def nms(input:torch.Tensor, image_size, threshold = 0.5):
    keypoints = export_point(input, image_size, 4)
    need_to_del = [False] * len(keypoints)
    for i, keypoint in enumerate(keypoints):
        if need_to_del[i]:
            continue
        for j, target in enumerate(keypoints):
            if i == j or need_to_del[j]:
                continue
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

                abs_x = min((gx + rel_x.item()) * stride, image_size - 1)
                abs_y = min((gy + rel_y.item()) * stride, image_size - 1)
                keypoints.append([abs_x, abs_y, float(objectness)])

    return keypoints