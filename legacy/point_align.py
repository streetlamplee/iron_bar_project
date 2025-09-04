import numpy as np
import cv2
import os

import extension
from find_cross_point_model.predict import predict_one_image
from collections import deque

from n_blur import custom_blur
from n_warp import warp_perspective
from scipy.spatial import cKDTree
from scipy.interpolate import Rbf
from typing import List, Tuple

def icp_rigid_numpy(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    ICP (Iterative Closest Point) for rigid alignment (rotation + translation only)

    Args:
        source_points (np.ndarray): Nx2 array of points to be aligned (B)
        target_points (np.ndarray): Nx2 array of reference points (A)
        max_iterations (int): maximum number of ICP iterations
        tolerance (float): convergence criterion based on mean distance change

    Returns:
        aligned_points (np.ndarray): Nx2 aligned points (B after transform)
        rotation_matrix (np.ndarray): 2x2 rotation matrix
        translation_vector (np.ndarray): 1x2 translation vector
        final_mean_error (float): mean distance after final iteration
    """


    # Start with a copy of source points (B)
    src = np.copy(source_points)
    prev_error = float('inf')

    for i in range(max_iterations):
        # Nearest neighbor search from source to target
        target_points = np.array(target_points)
        tree = cKDTree(target_points)
        distances, indices = tree.query(src)
        indices = np.asarray(indices, dtype=int)

        # Get corresponding points in target
        matched_target = target_points[indices]

        # Estimate optimal rigid transform (rotation + translation)
        offsets = matched_target - src
        distances = np.linalg.norm(offsets, axis=1)
        centroid_src = np.mean(src, axis=0)
        centroid_tgt = np.mean(matched_target, axis=0)

        src_centered = src - centroid_src
        tgt_centered = matched_target - centroid_tgt

        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Reflection check
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_tgt - centroid_src @ R.T

        # Apply the transformation
        src = src @ R.T + t

        # Compute mean error
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return src, R, t, mean_error

def icp_rigid_trimmed_numpy(src_points, target_points, max_iter=50, tolerance=1e-6, trim_ratio=0.9):
    """
    Trimmed ICP (Rigid, 2D)

    Parameters:
        src_points: (N, 2) ndarray – Source points to align (will be transformed)
        target_points: (M, 2) ndarray – Target points (fixed)
        max_iter: maximum ICP iterations
        tolerance: convergence threshold
        trim_ratio: fraction (0 < trim_ratio ≤ 1) of closest matches to retain (e.g., 0.8 keeps top 80%)

    Returns:
        aligned_src: transformed source points (N, 2)
        R: rotation matrix (2x2)
        t: translation vector (2,)
        final_error: mean alignment error after convergence
    """
    src = src_points.copy()
    src = np.asarray(src)
    target_points = np.asarray(target_points)
    target_kd = cKDTree(target_points)

    prev_error = np.inf
    R_total = np.eye(2)
    t_total = np.zeros(2)

    for i in range(max_iter):
        # Find nearest neighbors in target
        dists, indices = target_kd.query(src)
        # Sort and trim by distance
        trim_num = int(len(dists) * trim_ratio)
        sorted_idx = np.argsort(dists)
        trim_idx = sorted_idx[:trim_num].astype(int)

        trimmed_src = src[trim_idx]
        trimmed_target = target_points[indices[trim_idx]]

        # Compute centroids
        centroid_src = trimmed_src.mean(axis=0)
        centroid_tgt = trimmed_target.mean(axis=0)

        # Subtract centroids
        src_centered = trimmed_src - centroid_src
        tgt_centered = trimmed_target - centroid_tgt

        # SVD for optimal rotation
        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_tgt - R @ centroid_src

        # Apply transformation to source
        src = (src @ R.T) + t

        # Accumulate total transformation
        R_total = R @ R_total
        t_total = R @ t_total + t

        mean_error = np.mean(dists[trim_idx])
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return src, R_total, t_total, prev_error, trim_idx

def icp_align(image_folder_path:str, warp_point, max_iter:int = 200, tolerance:float = 1e-6):
    image_filename_list = os.listdir(image_folder_path)
    image_filename_list.sort()
    idx = 0
    image_q = deque()
    image_q.extend(image_filename_list)

    src_image = cv2.imread(os.path.join(image_folder_path, image_q.popleft()))
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    src_image = cv2.resize(src_image, (1360, 1020))
    # extension.image_show(src_image)

    src_warped = warp_perspective(src_image, np.array(warp_point[f'{idx}'], dtype = np.float32))
    # extension.image_show(src_warped)

    src_predicted_image, src_predicted_point = predict_one_image(src_warped)
    # extension.image_show(src_predicted_image)

    warped_image_list = [src_warped]
    predicted_point_list = [src_predicted_point]
    aligned_point_list = [src_predicted_point]
    R_arr_list = []
    t_arr_list = []
    inlier_aligned_point_list = []
    inlier_predicted_points_list = []


    while image_q:
        idx += 1
        target_image = cv2.imread(os.path.join(image_folder_path, image_q.popleft()))
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        target_image = cv2.resize(target_image, (1360,1020))
        target_warped = warp_perspective(target_image, np.array(warp_point[f'{idx}'], dtype = np.float32))
        warped_image_list.append(target_warped)
        # extension.image_show(target_warped)

        target_predicted_image, target_predicted_point = predict_one_image(target_warped)
        predicted_point_list.append(target_predicted_point)

        aligned_target_point, R, t, error, trim_idx = icp_rigid_trimmed_numpy(src_predicted_point,target_predicted_point, max_iter, tolerance, 0.85)
        aligned_point_list.append(aligned_target_point)
        R_arr_list.append(R)
        t_arr_list.append(t)

        inlier_aligned_points = aligned_target_point[trim_idx]
        inlier_predicted_points = np.array(src_predicted_point)[trim_idx]

        inlier_aligned_point_list.append(inlier_aligned_points)
        inlier_predicted_points_list.append(inlier_predicted_points)

    aligned_image_list = [src_warped]
    for i, (R, t) in enumerate(zip(R_arr_list, t_arr_list)):
        M = np.hstack([R, t.reshape(2,1)])
        image = warped_image_list[i+1]
        aligned_image = cv2.warpAffine(image, M, dsize=(1024, 1024))
        aligned_image_list.append(aligned_image)

    res_image = np.zeros_like(src_warped)
    for aligned_img in aligned_image_list:
        aligned_img = custom_blur(aligned_img)
        aligned_img = aligned_img.astype(np.float32)
        res_image = res_image + (aligned_img * 1 / (len(aligned_image_list)))

    return res_image, aligned_image_list, predicted_point_list, aligned_point_list, inlier_aligned_point_list, inlier_predicted_points_list


def compute_tps_transform(src_points:np.ndarray,
                          dst_points:np.ndarray
                          ):
    src_x, src_y = src_points[:,0], src_points[:,1]
    dst_x, dst_y = dst_points[:,0], dst_points[:,1]

    fx = Rbf(src_x, src_y, dst_x, function='thin_plate')
    fy = Rbf(src_x, src_y, dst_y, function='thin_plate')

    return fx, fy

def apply_tps_to_image(image:np.ndarray, fx, fy, output_size:Tuple[int,int] = (1024, 1024)):
    h, w = output_size
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x_flat = grid_x.flatten()
    grid_y_flat = grid_y.flatten()

    map_x = fx(grid_x_flat, grid_y_flat).reshape(h,w).astype(np.float32)
    map_y = fy(grid_x_flat, grid_y_flat).reshape(h,w).astype(np.float32)

    warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped_image

def apply_tps_from_icp_points(
        aligned_images:List[np.ndarray],
        aligned_points:List[np.ndarray],
        reference_index: int = 0,
        output_size:Tuple[int,int] = (1024,1024)
) -> List[np.ndarray]:
    """
    ICP 결44과를 기반으로 TPS를 적용

    Parameters:
        aligned_images: ICP 정렬된 이미지 리스트
        aligned_points: ICP로 정렬된 점들의 리스트
        reference_index: 기준이 되는 이미지 인덱스
        output_size: TPS 결과 이미지 크기 (w, h)

    Returns:
        tps_aligned_images: TPS 적용된 이미지 리스트
    """
    reference_points = aligned_points[reference_index]
    reference_points = np.array(reference_points)
    tps_aligned_images = []
    tps_aligned_points = []

    for i, (img, src_pts) in enumerate(zip(aligned_images, aligned_points)):
        if i == reference_index:
            tps_aligned_images.append(img)  # 기준 이미지는 그대로
            tps_aligned_points.append(src_pts)
        else:
            fx, fy = compute_tps_transform(src_pts, reference_points)
            warped = apply_tps_to_image(img, fx, fy, output_size)
            tps_aligned_images.append(warped)

            transformed_pts = np.stack([fx(src_pts[:,0], src_pts[:, 1]),
                                       fy(src_pts[:,0], src_pts[:, 1])],
                                      axis = 1)
            tps_aligned_points.append(transformed_pts)

    return tps_aligned_images, tps_aligned_points


def visualize_correspondences(predicted_point_list, image_shape=(1024, 1024), max_colors=50, seed=42):
    src_points = predicted_point_list[0]
    vis_image = np.zeros((image_shape[1], image_shape[0], 3), dtype=np.uint8)
    pallete = extension.random_pallete(max_colors, seed)

    for target_points in predicted_point_list[1:]:
        tree = cKDTree(target_points)
        _, indices = tree.query(src_points)

        for i, idx in enumerate(indices):
            color = tuple(int(c) for c in pallete[i % max_colors])
            x1, y1 = src_points[i]
            x2, y2 = target_points[idx]
            # draw src point
            cv2.circle(vis_image, (int(y1), int(x1)), 3, color, -1)
            # draw matched target point
            cv2.circle(vis_image, (int(y2), int(x2)), 3, color, -1)

    return vis_image

if __name__ == '__main__':
    import json
    with open('cache/pnp.json', 'r') as f:
        _2d_coordinate = json.load(f)
    res_image_blending, res_image_list, res_predicted_point, res_aligned_point = icp_align('./data/image_seg', _2d_coordinate)
    vis_img = visualize_correspondences(res_predicted_point)
    extension.image_show(vis_img)