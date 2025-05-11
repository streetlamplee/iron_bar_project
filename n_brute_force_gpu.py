import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import itertools
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_perspective_grid(M, out_h, out_w):
    device = M.device
    ys, xs = torch.meshgrid(
        torch.arange(out_h, device=device),
        torch.arange(out_w, device=device),
        indexing='ij'
    )
    ones = torch.ones_like(xs)
    coords = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3).T  # [3, H*W]
    coords = coords.float()
    warped = M @ coords  # [3, H*W]
    warped = warped[:2] / warped[2:]  # normalize
    warped = warped.T.reshape(out_h, out_w, 2)
    warped[..., 0] = (warped[..., 0] / (out_w - 1)) * 2 - 1
    warped[..., 1] = (warped[..., 1] / (out_h - 1)) * 2 - 1
    return warped.unsqueeze(0)  # [1, H, W, 2]


def warp_perspective_torch(img, M):
    _, _, H, W = img.shape
    grid = get_perspective_grid(M, H, W)
    return F.grid_sample(img, grid, align_corners=True)


# --- MAIN BRUTE-FORCE WARPING ---
def brute_force_best_warp(img_np, basis_np, _2d_point_np, offset=2, i=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 준비: 이미지 및 기준점
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(device)  # [1, 1, H, W]
    _2d_point = torch.from_numpy(_2d_point_np).to(device).float()
    basis_image = torch.from_numpy(basis_np).to(device).float()

    r = range(-offset, offset + 1)
    combos = list(itertools.product(r, repeat=8))
    print(f"Total combos: {len(combos)}")

    max_num_255 = -1
    best_M = None
    best_warped = None

    for combo in tqdm(combos, desc="Searching"):
        offset_arr = torch.tensor(combo, dtype=torch.float32, device=device).view(4, 2)
        dst_points = _2d_point + offset_arr

        # Perspective transform 행렬 (NumPy로 계산 후 변환)
        M_np = cv2.getPerspectiveTransform(_2d_point.cpu().numpy().astype(np.float32),
                                           dst_points.cpu().numpy().astype(np.float32))
        M = torch.from_numpy(M_np).float().to(device)

        startpoint = dst_points.int().tolist()
        endpoint = [[0,0],[1024,0],[1024,1024],[0,1024]]
        warped = VF.perspective(img_tensor, startpoint, endpoint)
        warped = F.interpolate(warped, size = (1024, 1024), mode = 'bilinear', align_corners=False)
        # cv2.imshow('',warped.squeeze(0).squeeze(0).detach().cpu().numpy())
        # cv2.waitKey(0)
        # cv2.destroyWindow('')
        basis_iter = basis_image * i / (i + 1)
        warped_scaled = warped * 1. / (i + 1)
        target_iter = basis_iter + warped_scaled

        num_255 = torch.count_nonzero(target_iter >= 255.).item()

        if num_255 > max_num_255:
            max_num_255 = num_255
            best_M = M
            best_warped = warped
            # best_warped_np = best_warped.detach().cpu().view(1080, 1440).numpy().astype(np.uint8)
            # cv2.imshow('test', best_warped_np)
            # cv2.waitKey(0)
            # cv2.destroyWindow('test')
            # print(f'')
    # 최종 블렌딩
    # basis_image = basis_image / 2. + best_warped / 2.

    # # 시각화
    # plt.figure(figsize=(20, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(basis_image.squeeze().detach().cpu().numpy().astype(np.uint8), cmap='gray')
    # plt.axis('off')
    # plt.title("Blended Image")
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(best_warped.squeeze().detach().cpu().numpy().astype(np.uint8), cmap='gray')
    # plt.axis('off')
    # plt.title("Best Warped")
    #
    # plt.show()

    return best_M, best_warped