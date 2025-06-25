import numpy as np
from tqdm.utils import disp_trim

from n_camera import Camera
import cv2



def warp_perspective(image:np.ndarray, target_points:np.ndarray, dst = np.float32([[0,0],[1024,0],[1024,1024],[0,1024]])):
    M = cv2.getPerspectiveTransform(target_points, dst)
    size_x = int( np.max(dst[:,0]) - np.min(dst[:,0]) )
    size_y = int( np.max(dst[:,1]) - np.min(dst[:,1]) )
    warp = cv2.warpPerspective(image, M, (size_x, size_y), flags = cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)

    return warp