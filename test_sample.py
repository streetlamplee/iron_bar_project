import numpy as np
import cv2
import os

import extension
from n_pointClicker import PointClicker
from n_warp import warp_perspective

def test():
    test_sample_folder = os.listdir('data_sample')
    test_sample_folder = sorted(test_sample_folder)

    clicker = PointClicker(4)
    points_list = []
    for test_sample in test_sample_folder:
        sample_image = cv2.imread(os.path.join('data_sample', test_sample))
        points = clicker.get_points(sample_image)
        points_list.append(points)
    # 상부근 철근 좌표 list
    # points_list = [[(1716, 1086), (3786, 1353), (2643, 2173), (100, 1656)], [(1656, 933), (3603, 1330), (2583, 2453), (253, 1760)], [(1703, 806), (3450, 1303), (2536, 2476), (486, 1713)], [(1700, 813), (3363, 1323), (2500, 2513), (570, 1763)], [(1726, 913), (3266, 1406), (2436, 2500), (673, 1793)]]
    # 하부근 철근 좌표 list
    # points_list = [[(1666, 1313), (3743, 1610), (2613, 2516), (100, 1930)], [(1633, 1140), (3563, 1556), (2570, 2730), (256, 1996)], [(1680, 993), (3413, 1493), (2516, 2696), (490, 1906)], [(1683, 990), (3326, 1493), (2486, 2700), (580, 1943)], [(1710, 1073), (3233, 1563), (2420, 2676), (676, 1953)]]

    print(points_list)
    warp_image_list = []
    for test_sample, points in zip(test_sample_folder, points_list):
        sample_image = cv2.imread(os.path.join('data_sample', test_sample))
        warp_image = warp_perspective(sample_image, np.array(points, dtype = np.float32))
        warp_image_list.append(warp_image)

    result = np.zeros((1024,1024), dtype = np.float32)
    warp_image_list.pop()
    for warp_image in warp_image_list:
        result += cv2.cvtColor(warp_image, cv2.COLOR_BGR2GRAY).astype(np.float32) * (1 / len(warp_image_list))

    extension.image_show(result.astype(np.uint8))



if __name__ == '__main__':
    test()