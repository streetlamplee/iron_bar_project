import os
import cv2
import numpy as np

from extension import image_show
from n_pointClicker import PointClicker
import iron_bar_segmentation.predict as seg_predict
from n_warp import warp_perspective

FastDebug = True

def main():
    '''
    main run method
    '''

    '''
    데이터 위치 선언
    '''
    real_data_folder = 'data_real'
    real_data_filename_list = os.listdir(real_data_folder)

    '''
    데이터 불러오기
    '''
    real_image_list = []
    for real_data_filename in real_data_filename_list[:5]:
        real_data_path = os.path.join(real_data_folder, real_data_filename)
        real_image = cv2.imread(real_data_path)
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
        real_image_list.append(real_image)

    '''
    철근 찾기
    '''
    iron_seg_image_list = []

    for real_image in real_image_list:
        iron_seg_image = seg_predict.predict(real_image)
        iron_seg_image_list.append(iron_seg_image)


    '''
    목표 구역 설정
    '''
    isShow = False
    if FastDebug:
        warp_point_list_stack = [[(527, 387), (746, 370), (829, 483), (540, 503)], [(920, 352), (1022, 471), (705, 467), (684, 351)], [(857, 174), (1004, 264), (739, 307), (644, 202)], [(396, 409), (618, 398), (630, 518), (333, 534)], [(800, 468), (831, 590), (514, 572), (567, 460)]]
    else:
        warping_area_list = []
        warp_point_list_stack = []
        clicker = PointClicker(1)
        for real_image in real_image_list:
            real_image = cv2.resize(real_image, (1200, 900))
            # warping_area_list.append(clicker.get_points(real_image))
        warping_area_list = [[(653, 451)], [(805, 414)], [(795, 254)], [(499, 483)], [(673, 534)]]

        clicker = PointClicker(4)
        num = 0
        for real_image, warping_area in zip(real_image_list, warping_area_list):
            warping_center = warping_area[0]
            real_image = cv2.resize(real_image, (1200,900))
            point_check_image = real_image.copy()

            h_start = max(0, warping_center[1] - 225)
            h_end = min(real_image.shape[0], h_start + 450)
            w_start = max(0, warping_center[0] - 300)
            w_end = min(real_image.shape[1], w_start + 600)

            real_image = real_image[h_start:h_end, w_start:w_end]
            real_image = cv2.resize(real_image, (1200, 900))
            warp_point_list = clicker.get_points(real_image)
            warp_point_list = [(x // 2 + w_start, y // 2 + h_start) for x, y in warp_point_list]
            warp_point_list_stack.append(warp_point_list)

            if isShow:
                for warp_point in warp_point_list:
                    point_check_image = cv2.circle(point_check_image, warp_point, thickness=-1, radius = 10, color = (0, 0, 255))
                cv2.imwrite(f'{num}.png', point_check_image)
                num += 1
        print(warp_point_list_stack)

    '''
    warp_perspective
    '''
    warp_seg_image_list = []
    for iron_seg_image, warping_points in zip(iron_seg_image_list, warp_point_list_stack):
        iron_seg_image = cv2.resize(iron_seg_image, (1200, 900))
        warp_seg_image = warp_perspective(iron_seg_image, np.array(warping_points, dtype = np.float32))
        warp_seg_image_list.append(warp_seg_image)
        if FastDebug:
            image_show(warp_seg_image)

    '''
    철근 교차점 찾기
    '''
    from image_2_ironbar_model.predict import predict as cross_predict

    _, result, _ = cross_predict(warp_seg_image_list)

    image_show(result)












if __name__ == '__main__':
    main()