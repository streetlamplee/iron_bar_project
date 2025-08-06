import os
import cv2
import numpy as np

from extension import image_show
from n_pointClicker import PointClicker
import iron_bar_segmentation.predict as seg_predict
from n_warp import warp_perspective
from get_picture_from_raspi import get_picture_from_raspi
from video_to_imageset import video_to_frame, target_frame

FastDebug = True
isTop = True

def main():
    '''
    main run method
    '''
    isShow = True

    '''
    데이터 위치 선언
    '''
    # data_list = get_picture_from_raspi(False)
    data_list = target_frame('video_frame', [0, 5, 11, 19])
    '''
    철근 찾기
    '''
    iron_seg_image_list = []

    for real_image in data_list:
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
        iron_seg_image = seg_predict.predict(real_image)
        if isShow:
            image_show(iron_seg_image)

        iron_seg_image_list.append(iron_seg_image)

    '''
    목표 구역 설정
    '''

    if FastDebug:
        # warp_point_list_stack = [[(1760, 1286), (2483, 1236), (2756, 1600), (1796, 1673)], [(1310, 1360), (2056, 1323), (2096, 1720), (1100, 1773)], [(2853, 576), (3343, 869), (2456, 1010), (2133, 663)], [(3066, 1166), (3400, 1560), (2346, 1543), (2276, 1166)], [(2660, 1553), (2763, 1960), (1713, 1896), (1886, 1530)], [(2720, 1400), (2556, 1820), (1613, 1640), (1960, 1310)], [(2023, 1346), (2683, 1320), (3080, 1653), (2236, 1700)], [(1756, 1283), (2486, 1233), (2756, 1600), (1796, 1670)], [(1310, 1360), (2056, 1320), (2093, 1720), (1100, 1776)], [(1960, 1810), (2306, 2150), (1306, 2360), (1196, 1926)], [(3573, 2033), (3916, 2490), (2780, 2410), (2696, 1963)]]
        # data_real 데이터 warp point
        # warp_point_list_stack = [[(2541, 386), (4154, 1399), (899, 2135), (942, 531)], [(2867, 932), (4067, 2377), (341, 2301), (1348, 924)], [(2609, 1323), (2974, 2866), (-228, 2413), (1166, 1287)], [(2883, 1020), (2559, 1824), (1113, 1541), (1920, 950)], [(1488, 790), (3143, 1054), (2320, 2238), (100, 1318)]]
        if isTop:
            # 상부근 철근 좌표 list
            warp_point_list_stack = [[(751, 240), (1149, 382), (903, 674), (466, 481)], [(789, 144), (1180, 277), (944, 556), (511, 379)], [(621, 36), (996, 158), (755, 426), (341, 266)], [(350, 135), (718, 267), (438, 546), (36, 373)]]

        else:
            # 하부근 철근 좌표 list
            warp_point_list_stack = [[(1666, 1313), (3743, 1610), (2613, 2516), (100, 1930)], [(1633, 1140), (3563, 1556), (2570, 2730), (256, 1996)], [(1680, 993), (3413, 1493), (2516, 2696), (490, 1906)], [(1683, 990), (3326, 1493), (2486, 2700), (580, 1943)], [(1710, 1073), (3233, 1563), (2420, 2676), (676, 1953)]]

        print(warp_point_list_stack)

        num = 0
        for real_image, warp_point_list in zip(data_list, warp_point_list_stack):

            tmp = real_image.copy()
            tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
            # tmp = cv2.resize(tmp, (1200, 900))
            for warp_point in warp_point_list:
                tmp = cv2.circle(tmp, warp_point, thickness=-1, radius = 3, color = (0, 0, 255))
            cv2.imwrite(f'{num}.png', tmp)
            num += 1
            image_show(tmp)
    else:
        warping_area_list = []
        warp_point_list_stack = []

        clicker = PointClicker(4)
        num = 0
        for real_image in data_list:
            original_h, original_w = real_image.shape[:2]
            # canvas = np.ones(shape = (original_h+1000, original_w+1000, 3), dtype = np.uint8) * 255
            # real_image = cv2.resize(real_image, (1200,900))
            # canvas[500:500+original_h, 500:500+original_w] = real_image
            canvas = real_image.copy()
            point_check_image = real_image.copy()
            point_check_image = cv2.cvtColor(point_check_image, cv2.COLOR_RGB2BGR)
            warp_point_list = clicker.get_points(canvas)
            # warp_point_list = [(x - 500, y - 500) for x, y in warp_point_list]
            # print(f"warp_point_list : {warp_point_list}")
            # warp_point_list = [(int(x / 1200 * original_w), int(y / 900 * original_h)) for x, y in warp_point_list]

            warp_point_list_stack.append(warp_point_list)

            if isShow:
                for warp_point in warp_point_list:
                    point_check_image = cv2.circle(cv2.resize(point_check_image, (original_w,original_h)), warp_point, thickness=-1, radius = 3, color = (0, 0, 255))
                cv2.imwrite(f'{num}.png', point_check_image)
                num += 1
        print(warp_point_list_stack)

    '''
    warp_perspective
    '''
    warp_seg_image_list = []
    num = 0
    for iron_seg_image, warping_points, r in zip(iron_seg_image_list, warp_point_list_stack, data_list):
        warp_seg_image = warp_perspective(iron_seg_image,
                                          np.array(warping_points, dtype = np.float32),
                                          dst = np.float32([[0,0],[1024,0],[1024,1024],[0,1024]]))
        # test
        warp_real_image = warp_perspective(r,
                                           np.array(warping_points, dtype = np.float32),
                                           dst = np.float32([[0,0], [1024,0], [1024,1024], [0, 1024]]))
        warp_seg_image_list.append(warp_seg_image)
        if FastDebug:

            # image_show(warp_seg_image)
            cv2.imwrite(f'{num}_seg.png', warp_seg_image)
            cv2.imwrite(f'{num}_warp.png', warp_real_image)
            num += 1

    '''
    erode dilate로 확인하기
    '''
    target = [0,1,2,3]
    n = 0
    from n_blur import custom_blur
    result = np.zeros_like(warp_seg_image_list[0]).astype(np.float32)
    for i, warp_seg_image in enumerate(warp_seg_image_list):
        if i not in target:
            continue
        warp_seg_image = custom_blur(warp_seg_image)
        result += warp_seg_image * (1 / len(target))
        n += 14
    print(n)
    result_seg = result.astype(np.uint8)
    cv2.imwrite(f"result_top_before.png", result_seg)
    result_t = np.where(result_seg > int(255 * (len(target)-1) / len(target)), 255, 0)
    if isShow:
        image_show(result_seg)
        image_show(result_t.astype(np.uint8))
    top_btm = 'top' if isTop else 'btm'
    cv2.imwrite(f"result_{top_btm}.png", result_t)

    '''
    점 찾기 이후 선 그리기
    '''

    from find_cross_point_model.predict import predict_one_image as point_predict
    from n_draw_line import draw_line
    point_image, point_list = point_predict(result_seg, '/home/user/PycharmProjects/iron_bar_sample_project/find_cross_point_model/models/20250624_170217/epoch02015.pth')

    line_drawing_result = draw_line(point_list)

    cv2.imwrite(f"result_{top_btm}_line.png", line_drawing_result)


    return

    '''
    철근 교차점 찾기
    '''
    from image_2_ironbar_model.predict import predict as cross_predict
    input_list = []
    for warp_seg_image, image_filename in zip(warp_seg_image_list, real_data_filename_list):
        target_filename = [f'{num}.jpg' for num in target_list]
        if not any(image_filename.endswith(target) for target in target_filename):
            continue
        input_list.append(cv2.resize(warp_seg_image, (1024,1024)))
    _, result, result_point_list = cross_predict(input_list)
    if isShow:
        image_show(result)
    result_seg = cv2.cvtColor(result_seg, cv2.COLOR_GRAY2BGR)
    for result_point in result_point_list:
        result_seg = cv2.circle(cv2.resize(result_seg, (1024,1024)), result_point[1:], radius = 5, thickness = -1, color = (0,0,255))

    cv2.imwrite('result.png', result_seg.astype(np.uint8))

if __name__ == '__main__':
    main()