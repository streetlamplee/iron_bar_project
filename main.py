import os
import cv2
import numpy as np
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_path not in sys.path:
    sys.path.append(src_path)


from etc.extension import image_show
from processing.pointClicker import PointClicker
import iron_bar_segmentation.predict as seg_predict
from processing.warp import warp_perspective
from get.get_picture_from_raspi import get_picture_from_raspi
from processing.video_to_imageset import video_to_frame, target_frame

FastDebug = False
isTop = True

def prev_main():
    '''
    main run method
    '''
    isShow = True
    if not os.path.exists("./output"):
        os.makedirs("./output", exist_ok=True)

    '''
    데이터 위치 선언
    '''
    # data_list = get_picture_from_raspi(False)
    # data_list = target_frame('video_frame', [0, 5, 11, 19])
    # fname_list = ["./data_real/0818/20250818_145135.jpg",
    #               "./data_real/0818/20250818_145131.jpg",
    #               "./data_real/0818/20250818_145136.jpg",
    #               "./data_real/0818/20250818_145140.jpg",]
    folder_name = "./data/input_data/"
    fname_list = os.listdir(folder_name)
    data_list = [cv2.imread(os.path.join(folder_name,f)) for f in fname_list]
    '''
    철근 찾기
    '''
    iron_seg_image_list = []

    for i,real_image in enumerate(data_list):
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
        iron_seg_image = seg_predict.predict(real_image)
        if isShow:
            image_show(iron_seg_image)
        cv2.imwrite(f"output/2.{i}_seg.png", iron_seg_image)
        iron_seg_image_list.append(iron_seg_image)

    '''
    목표 구역 설정
    '''
    points5 = np.array([[2158,754], [3312, 1168], [2588, 2125], [1301, 1304]])
    points1 = np.array([[1579, 716], [2542, 1109], [1523, 2020], [556, 1250]])
    points6 = np.array([[2164, 380], [3551, 685], [2714, 1646], [1143, 866]])
    points10 = np.array([[1374, 289], [2462, 630], [1113, 1524], [75, 762]])
    if FastDebug:
        warp_point_list_stack = np.array([[(1651, 783), (2579, 913), (2014, 1468), (793, 1141)], [(1746, 825), (2709, 923), (2584, 1643), (1226, 1371)], [(1631, 880), (2479, 1088), (1946, 1754), (865, 1301)], [(2106, 480), (3262, 635), (3070, 1281), (1443, 870)]])

        num = 0
        for real_image, warp_point_list in zip(data_list, warp_point_list_stack):

            tmp = real_image.copy()
            # tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
            # tmp = cv2.resize(tmp, (1200, 900))
            for warp_point in warp_point_list:
                tmp = cv2.circle(tmp, warp_point, thickness=-1, radius = 7, color = (0, 0, 255))
            cv2.imwrite(f'output/1.{num}.png', tmp)
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
                cv2.imwrite(f'output/1.{num}.png', point_check_image)
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
            cv2.imwrite(f'output/4.{num}_seg_warp.png', warp_seg_image)
            cv2.imwrite(f'output/3.{num}_warp.png', warp_real_image)
            num += 1

    '''
    erode dilate로 확인하기
    '''
    target = [0,1,2,3]
    n = 0
    from processing.blur import custom_blur
    result = np.zeros_like(warp_seg_image_list[0]).astype(np.float32)
    for i, warp_seg_image in enumerate(warp_seg_image_list):
        if i not in target:
            continue
        warp_seg_image = custom_blur(warp_seg_image)
        result += warp_seg_image * (1 / len(target))
        n += 14
    print(n)
    result_seg = result.astype(np.uint8)
    cv2.imwrite(f"output/5.result_top_before.png", result_seg)
    result_t = np.where(result_seg > int(255 * (len(target)-1) / len(target)), 255, 0)
    if isShow:
        image_show(result_seg)
        image_show(result_t.astype(np.uint8))
    top_btm = 'top' if isTop else 'btm'
    cv2.imwrite(f"output/6.result_{top_btm}.png", result_t)

    '''
    점 찾기 이후 선 그리기
    '''

    # from find_cross_point_model.predict import predict_one_image as point_predict
    # from n_draw_line import draw_line
    # point_image, point_list = point_predict(result_seg, '/home/user/PycharmProjects/iron_bar_sample_project/find_cross_point_model/models/20250624_170217/epoch02015.pth')
    #
    # line_drawing_result = draw_line(point_list)
    #
    # cv2.imwrite(f"result_{top_btm}_line.png", line_drawing_result)


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

import os
import cv2
import numpy as np
# from get_picture_from_raspi import get_picture_from_raspi
# import warp_point_finder.detect_marker
# import warp_point_finder.prjection
import iron_bar_segmentation.predict as seg_predict
from processing.warp import warp_perspective
from processing.blur import custom_blur
from etc.extension import image_show
from featureMatching.featureMatching import perspectiveTransfrom, getHomographySift


def main():
    '''
    프로젝트의 메인 함수
    '''

    # init
    is_raspi_connected = False
    # points = [[1065, 503], [1393, 559], [1341, 849], [951, 765]]
    # points = np.array([[853, 701], [1169, 712], [1171, 965], [762, 940]])
    points5 = np.array([[2158,754], [3312, 1168], [2588, 2125], [1301, 1304]])
    points1 = np.array([[1579, 716], [2542, 1109], [1523, 2020], [556, 1250]])
    points6 = np.array([[2164, 380], [3551, 685], [2714, 1646], [1143, 866]])
    points10 = np.array([[1374, 289], [2462, 630], [1113, 1524], [75, 762]])
    # warping_point = [

    #     [-1.0, 1.0, 0],
    #     [1.0, 1.0, 0],
    #     [1.0, -1.0, 0],
    #     [-1.0, -1.0, 0]
    # ]
    warping_img_segmentation = []
    warping_img = []

    # 데이터 불러오기
    # img_arr = get_picture_from_raspi(is_raspi_connected)
    # fname_list = ["./data_real/0818/20250818_145135.jpg",
    #               "./data_real/0818/20250818_145131.jpg",
    #               "./data_real/0818/20250818_145136.jpg",
    #               "./data_real/0818/20250818_145140.jpg",]
    fname_list = os.listdir("./data/input_data")
    img_arr = [cv2.imread(f) for f in fname_list]

    # 현재 cam3 고장에 대응하기 위한 코드 추가 (cam3 조치완료 시 삭제할 것)
    # img_arr = img_arr[:-1]

    # 각 데이터 사진 한 개마다 process 적용
    for i, img in enumerate(img_arr):
        # # 마커 찾기 및 마커기준 카메라의 rvec tvec 찾기
        # _, k, d, rvec, tvec = warp_point_finder.detect_marker.detectMarker(img, "./warp_point_finder/images")
        #
        # # warp point projection
        # warping_point_2d = []
        # for wp in warping_point:
        #     wp_2d = warp_point_finder.prjection.project(wp, k, d, rvec, tvec)
        #     warping_point_2d.append(wp_2d)

        # warping point 찾기 (SIFT 사용)
        if i == 0:
            srcimg = img
            warping_point_2d = points
        else:
            srcimg_gray = cv2.cvtColor(srcimg, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            retval, M = getHomographySift(srcimg_gray, img_gray)

            if retval:
                warping_point_2d = perspectiveTransfrom(points, M)
            else:
                warping_point_2d = []

        # 디버그용 warp point 찍어보기
        img_debug = img.copy()
        for wp in warping_point_2d.reshape(4,2):
            cv2.circle(img_debug, wp.astype(np.int16), color=(0,0,255), thickness=-1, radius=10)
        img_debug = cv2.resize(img_debug, (1600, 900))
        image_show(img_debug, f"{i}")

        # 철근 찾기
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_segmentation = seg_predict.predict(img_rgb)

        # perspective warp 실행
        img_warp = warp_perspective(
            img,
            np.array(warping_point_2d, dtype = np.float32),
            dst = np.float32([[0,0],[1024,0],[1024,1024],[0,1024]])
        )
        img_segmentation_warp = warp_perspective(
            img_segmentation,
            np.array(warping_point_2d, dtype=np.float32),
            dst=np.float32([[0, 0], [1024, 0], [1024, 1024], [0, 1024]])
        )

        warping_img.append(img_warp)
        image_show(img_warp)
        warping_img_segmentation.append(img_segmentation_warp)

    # warping segmented image 를 블러 처리 후 합치기
    image_count = len(warping_img_segmentation)
    weighted_sum_image = np.zeros_like(warping_img_segmentation[0], dtype=np.float32)
    for i, ws_img in enumerate(warping_img_segmentation):
        ws_img_blur = custom_blur(ws_img)
        weighted_sum_image += ws_img_blur * (1 / len(warping_img_segmentation))
    weighted_sum_image = weighted_sum_image.astype(np.uint8)
    image_show(weighted_sum_image)
    cv2.imwrite("./weighted_sum_image.png", weighted_sum_image)
    thresholded_weighted_sum_image = np.where(weighted_sum_image > int(255 * ((image_count - 1) / image_count)), 255, 0)
    image_show(thresholded_weighted_sum_image.astype(np.uint8))
    cv2.imwrite("./weighted_sum_thresholded_image.png", thresholded_weighted_sum_image.astype(np.uint8))


if __name__ == '__main__':
    prev_main()
    # main()