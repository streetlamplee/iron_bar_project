"""
철근 배근 상태 확인 파이프라인의 실행 진입점.

이 파일에는 목적이 다른 두 개의 파이프라인이 들어 있다.
두 파이프라인의 핵심 계산 순서(segmentation -> perspective warp -> blur -> 다중 시점
가중 합성 -> threshold)는 동일하고, "관심 영역(warp point)을 어떻게 얻는가"가 다르다.

1. run_manual_warp_pipeline(z)  : 현재 사용 중인 경로.
                                  warp point를 사람이 지정한다(하드코딩 좌표 또는 마우스 클릭).
                                  상부근/하부근 좌표를 z 비율로 보간해 높이별 단면을 만들 수 있다.
2. run_sift_warp_pipeline()     : 실험 경로. 사람이 첫 장에만 좌표를 주고,
                                  나머지 장은 SIFT Homography로 좌표를 자동 전파한다.
                                  현재 그대로는 동작하지 않는다(함수 상단의 주의사항 참고).

경로가 모두 프로젝트 루트 기준 상대경로이므로 반드시 루트에서 실행할 것.
"""

import os
import cv2
import numpy as np
import sys

# src/ 를 import 경로에 추가해야 아래 프로젝트 모듈들을 import 할 수 있다.
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_path not in sys.path:
    sys.path.append(src_path)


from etc.extension import image_show
from processing.pointClicker import PointClicker
import iron_bar_segmentation.predict as seg_predict
from processing.warp import warp_perspective
from processing.blur import custom_blur
from get.get_picture_from_raspi import get_picture_from_raspi
from processing.video_to_imageset import video_to_frame, target_frame
# 아래 두 함수는 run_sift_warp_pipeline() 에서만 사용한다.
from featureMatching.featureMatching import perspectiveTransfrom, getHomographySift


def run_manual_warp_pipeline(z = 0):
    '''
    사람이 지정한 warp point로 동작하는 파이프라인 (현재 사용 중인 실행 경로).

    :param z: 상부근(100) ~ 하부근(0) 사이의 보간 비율. target == 1 에서만 사용된다.
              z=0 이면 하부근 평면, z=100 이면 상부근 평면, 그 사이는 선형 보간된 중간 높이.

    결과물은 output/{target}/{z}/ 아래에 단계 번호 순서(1.x ~ 6.x)로 저장된다.
    '''
    # isShow    : 1이면 중간 결과를 창으로 띄운다. GUI가 없는 환경에서는 0으로 둘 것.
    # FastDebug : 1이면 아래에 하드코딩된 warp point를 사용하고,
    #             0이면 PointClicker로 매 실행마다 4점을 직접 클릭해서 지정한다.
    isShow = 0
    FastDebug = 1

    # 처리할 입력 데이터 폴더 번호. data/input_data/{target}/ 을 가리킨다.
    target = 1

    '''
    출력 폴더 준비
    z가 0이 아니면 높이별로 구분해서 output/{target}/{z}/ 에 저장한다.
    '''
    if not os.path.exists("./output"):
        os.makedirs("./output", exist_ok=True)
    if not os.path.exists(f"./output/{target}"):
        os.makedirs(f"./output/{target}", exist_ok=True)

    output_folder = f"./output/{target}"

    if z != 0:
        if not os.path.exists(f"./output/{target}/{z}"):
            os.makedirs(f"./output/{target}/{z}", exist_ok=True)
        output_folder = f"./output/{target}/{z}"


    '''
    데이터 위치 선언
    같은 장면을 서로 다른 각도에서 찍은 사진 4장(카메라 4대)을 읽어온다.
    아래 주석은 과거에 쓰던 다른 입력 경로들이다(라즈베리파이 직접 수집 / 동영상 프레임 추출).
    '''
    # data_list = get_picture_from_raspi(False)
    # data_list = target_frame('video_frame', [0, 5, 11, 19])
    # fname_list = ["./data_real/0818/20250818_145135.jpg",
    #               "./data_real/0818/20250818_145131.jpg",
    #               "./data_real/0818/20250818_145136.jpg",
    #               "./data_real/0818/20250818_145140.jpg",]
    folder_name = f"./data/input_data/{target}/"
    fname_list = os.listdir(folder_name)
    # 주의: 아래 하드코딩된 warp point는 "파일 이름 순서"와 1:1로 대응한다.
    #       sorted()는 반환값을 쓰지 않으면 정렬이 적용되지 않으므로 확인 필요.
    sorted(fname_list)
    data_list = [cv2.imread(os.path.join(folder_name,f)) for f in fname_list]

    '''
    철근 찾기
    사진 한 장마다 DeepLabv3+ 로 철근 영역을 예측한다.
    predict()는 RGB 입력을 기대하므로 BGR -> RGB 변환 후 넣는다.
    결과는 0~255 grayscale 확률맵이다.
    '''
    iron_seg_image_list = []

    for i,real_image in enumerate(data_list):
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
        iron_seg_image = seg_predict.predict(real_image)
        if isShow:
            image_show(iron_seg_image)
        cv2.imwrite(f"{output_folder}/2.{i}_seg.png", iron_seg_image)
        iron_seg_image_list.append(iron_seg_image)

    '''
    목표 구역 설정
    warp_point_list_stack 의 형태는 (사진 장수, 4, 2) 이다.
    즉 사진 한 장당 관심 영역의 네 꼭짓점 좌표를 가진다.
    점 순서는 좌상 -> 우상 -> 우하 -> 좌하 이며, 아래 warp_perspective 의 dst 순서와 맞아야 한다.
    '''
    # 아래 points* 는 과거 단일 영역 실험에 쓰던 좌표. 현재 로직에서는 사용하지 않는다.
    points5 = np.array([[2158,754], [3312, 1168], [2588, 2125], [1301, 1304]])
    points1 = np.array([[1579, 716], [2542, 1109], [1523, 2020], [556, 1250]])
    points6 = np.array([[2164, 380], [3551, 685], [2714, 1646], [1143, 866]])
    points10 = np.array([[1374, 289], [2462, 630], [1113, 1524], [75, 762]])
    if FastDebug:
        # --- FastDebug = 1 : target 별로 미리 찍어둔 좌표를 그대로 사용 ---
        if target == 1:
            # target 1만 상부근/하부근 좌표를 둘 다 가지고 있어 z 보간이 가능하다.
            # 상부
            warp_point_list_stack = np.array([[(1651, 783), (2579, 913), (2014, 1468), (793, 1141)], [(1746, 825), (2709, 923), (2584, 1643), (1226, 1371)], [(1631, 880), (2479, 1088), (1946, 1754), (865, 1301)], [(2106, 480), (3262, 635), (3070, 1281), (1443, 870)]])
            # 하부
            warp_point_list_stack_btm = np.array([[(1666, 873), (2592, 1040), (1999, 1631), (855, 1248)], [(1756, 900), (2704, 1028), (2544, 1754), (1273, 1458)], [(1648, 960), (2484, 1206), (1929, 1864), (920, 1393)], [(2114, 580), (3242, 780), (2982, 1448), (1476, 993)]])
        elif target == 2:
            warp_point_list_stack = np.array([[(1461, 1005), (2594, 678), (3563, 1100), (2061, 1696)], [(1368, 893), (2739, 833), (3337, 1623), (1095, 1736)], [(1531, 1000), (2850, 918), (3640, 1661), (1523, 1839)], [(1601, 943), (2850, 728), (3770, 1313), (1901, 1746)]])
        elif target == 3:
            warp_point_list_stack = np.array([[(1313, 708), (2502, 728), (3022, 1100), (968, 1120)], [(1361, 715), (2522, 718), (3000, 1208), (1085, 1253)], [(1626, 848), (2804, 783), (3668, 1075), (1656, 1241)], [(1441, 678), (2552, 590), (3250, 1003), (1503, 1223)]])
        elif target == 4:
            warp_point_list_stack = np.array([[(1884, 628), (2607, 437), (3495, 703), (2727, 1100)], [(1934, 1233), (2509, 993), (3330, 1261), (2757, 1694)], [(2367, 1208), (2887, 923), (3840, 1128), (3365, 1613)], [(2354, 945), (2902, 725), (3913, 888), (3445, 1281)]])
        elif target == 5:
            warp_point_list_stack = np.array([[(2041, 410), (3035, 603), (2577, 1045), (1223, 635)], [(2352, 270), (3455, 417), (3357, 875), (1749, 552)], [(2392, 230), (3395, 437), (3142, 1025), (1829, 625)], [(2051, 492), (2970, 753), (2502, 1311), (1233, 850)]])
        elif target == 6:
            # target 6은 관심 영역 크기를 두 가지로 실험했다. 필요한 쪽의 주석을 해제해서 사용한다.
            # 작은 구역
            # warp_point_list_stack = np.array([[(813, 968), (1864, 733), (2467, 1241), (1035, 1731)], [(960, 1028), (1986, 755), (2719, 1246), (1423, 1811)], [(1288, 925), (2302, 755), (2817, 1333), (1311, 1643)], [(943, 1143), (1914, 753), (2787, 1148), (1709, 1869)]])
            # 큰 구역
            warp_point_list_stack = np.array([[(705, 565), (2169, 335), (3370, 923), (1035, 1731)], [(723, 613), (2181, 345), (3485, 900), (1421, 1814)], [(1291, 547), (2727, 352), (3981, 1070), (1306, 1643)], [(517, 730), (2019, 330), (3365, 743), (1706, 1871)]])
        elif target == 7:
            warp_point_list_stack = np.array([[(1458, 995), (2592, 675), (3575, 1108), (2074, 1716)], [(1368, 893), (2739, 830), (3337, 1626), (1093, 1739)], [(1531, 1000), (2850, 918), (3643, 1661), (1521, 1839)], [(1601, 945), (2857, 725), (3775, 1316), (1901, 1746)]])
        elif target == 8:
            warp_point_list_stack = np.array([[(2367, 465), (3217, 740), (2519, 1198), (1689, 760)], [(2382, 580), (3345, 745), (2647, 1115), (1661, 810)], [(2021, 573), (2694, 930), (1754, 1281), (1211, 763)], [(1946, 638), (2699, 875), (1681, 1196), (1063, 810)]])
        elif target == 9:
            warp_point_list_stack = np.array([[(1929, 750), (2855, 1216), (1924, 2059), (980, 1226)], [(1944, 432), (2957, 883), (1884, 1653), (873, 818)], [(1989, 350), (3165, 688), (1886, 1326), (773, 600)], [(1904, 913), (2729, 1448), (1819, 2249), (993, 1373)]])
        elif target == 10:
            warp_point_list_stack = np.array([[(588, 1000), (1814, 630), (3060, 1143), (1663, 2089)], [(710, 875), (1771, 395), (3090, 743), (2189, 1678)], [(1676, 725), (2589, 675), (3152, 1196), (1764, 1296)], [(1651, 658), (2527, 522), (3215, 950), (1946, 1203)]])
        else:
            # 좌표를 아직 찍어두지 않은 target이면 FastDebug로는 진행할 수 없다.
            print(f"There is no warp point setted with option FastDebug")
            return
        num = 0
        # 상부근 좌표와 하부근 좌표를 z 비율로 선형 보간해서 원하는 높이의 평면을 만든다.
        if target == 1:
            warp_point_list_stack = warp_point_list_stack * (z / 100) + warp_point_list_stack_btm * ((100 - z) / 100)

        # 확인용: 원본 사진 위에 사용한 warp point를 빨간 점으로 찍어 1.{num}.png 로 저장
        for real_image, warp_point_list in zip(data_list, warp_point_list_stack):

            tmp = real_image.copy()
            # tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
            # tmp = cv2.resize(tmp, (1200, 900))
            for warp_point in warp_point_list:
                tmp = cv2.circle(tmp, warp_point.astype(np.int16), thickness=-1, radius = 7, color = (0, 0, 255))
            cv2.imwrite(f'{output_folder}/1.{num}.png', tmp)
            num += 1
            if isShow:
                image_show(tmp)
    else:
        # --- FastDebug = 0 : 사진마다 4점을 직접 클릭해서 좌표를 만든다 ---
        # 창에서 4점을 클릭한 뒤 Enter를 누르면 다음 사진으로 넘어간다(ESC는 중단).
        # 클릭한 좌표는 마지막에 print 되므로, 복사해서 위 FastDebug 블록에 붙여넣어 재사용할 수 있다.
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
                num += 1
            cv2.imwrite(f'{output_folder}/1.{num}.png', point_check_image)
        print(warp_point_list_stack)

    '''
    warp_perspective
    각 사진의 관심 영역(4점)을 1024x1024 정사각 평면으로 펴서,
    서로 다른 각도에서 찍은 사진들을 같은 좌표계 위에 올린다.
    '''
    warp_seg_image_list = []
    num = 0
    for iron_seg_image, warping_points, r in zip(iron_seg_image_list, warp_point_list_stack, data_list):
        # segmentation 결과를 warp -> 이후 합성에 사용
        warp_seg_image = warp_perspective(iron_seg_image,
                                          np.array(warping_points, dtype = np.float32),
                                          dst = np.float32([[0,0],[1024,0],[1024,1024],[0,1024]]))
        # 원본 사진도 같은 변환으로 warp -> 눈으로 대조하기 위한 용도
        warp_real_image = warp_perspective(r,
                                           np.array(warping_points, dtype = np.float32),
                                           dst = np.float32([[0,0], [1024,0], [1024,1024], [0, 1024]]))
        warp_seg_image_list.append(warp_seg_image)
        if FastDebug:
            pass
            # image_show(warp_seg_image)
        # 보고용: segmentation 결과 위에 warp point를 크게 찍은 이미지
        iron_seg_report = iron_seg_image.copy()
        iron_seg_report = cv2.cvtColor(iron_seg_report, cv2.COLOR_GRAY2BGR)
        for w in warping_points:
            iron_seg_report = cv2.circle(iron_seg_report, np.array(w, dtype = np.int16), 25, (0,0,255), thickness = -1)
        cv2.imwrite(f"{output_folder}/report_seg_pointed_{num}.png", iron_seg_report)
        cv2.imwrite(f'{output_folder}/4.{num}_seg_warp.png', warp_seg_image)
        cv2.imwrite(f'{output_folder}/3.{num}_warp.png', warp_real_image)
        num += 1

    '''
    erode dilate로 확인하기
    warp된 segmentation 4장을 blur(dilate/erode + median/gaussian)로 두껍게 만든 뒤 평균낸다.
    그 다음 3/4 이상(= 4장 중 3장 이상)에서 철근으로 잡힌 픽셀만 남긴다.
    한 시점에서만 보이는 노이즈를 걸러내고, 여러 시점에서 공통으로 보이는 철근만 남기는 것이 목적이다.
    주의: 가중치 1/4 과 threshold 3/4 는 카메라 4대를 전제로 하드코딩되어 있다.
    '''
    n = 0
    result = np.zeros_like(warp_seg_image_list[0]).astype(np.float32)
    for i, warp_seg_image in enumerate(warp_seg_image_list):
        # if i not in target:
        #     continue
        warp_seg_image = custom_blur(warp_seg_image)
        result += warp_seg_image * (1 / 4)
        n += 14
    print(n)
    result_seg = result.astype(np.uint8)
    cv2.imwrite(f"{output_folder}/5.result_before.png", result_seg)
    result_t = np.where(result_seg > int(255 * (4-1) / 4), 255, 0)
    if isShow:
        image_show(result_seg)
        image_show(result_t.astype(np.uint8))

    cv2.imwrite(f"{output_folder}/6.result.png", result_t.astype(np.uint8))

    '''
    점 찾기 이후 선 그리기
    교차점 검출 모델로 철근 교차점을 찾고 선을 그리던 단계. 현재는 비활성 상태이다.
    '''

    # from find_cross_point_model.predict import predict_one_image as point_predict
    # from n_draw_line import draw_line
    # point_image, point_list = point_predict(result_seg, '/home/user/PycharmProjects/iron_bar_sample_project/find_cross_point_model/models/20250624_170217/epoch02015.pth')
    #
    # line_drawing_result = draw_line(point_list)
    #
    # cv2.imwrite(f"result_{top_btm}_line.png", line_drawing_result)


    return

    # ------------------------------------------------------------------
    # 아래는 위 return 때문에 실행되지 않는 코드다 (참고용으로만 남겨둠).
    # 철근 교차점을 찾아 개수를 세려던 단계이며, 사용하려면 return을 지우고
    # real_data_filename_list / target_list 를 먼저 정의해야 한다.
    # ------------------------------------------------------------------
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


def run_sift_warp_pipeline():
    '''
    warp point를 SIFT Homography로 자동 전파하는 파이프라인 (실험 단계).

    run_manual_warp_pipeline() 과 계산 순서는 같지만,
    첫 장에만 좌표를 주고 나머지 장의 좌표는 첫 장과의 특징점 매칭으로 계산한다는 점이 다르다.
    사진마다 4점을 클릭하는 수작업을 없애는 것이 목적이다.

    주의 - 현재 이 함수는 그대로 실행되지 않는다.
      1) 아래에서 참조하는 points 변수가 정의되어 있지 않다 (points1/5/6/10 만 있음).
         사용하려면 대상 데이터에 맞는 좌표를 points 에 넣어야 한다.
      2) 입력 경로가 data/input_data 로, 사진이 아니라 target 폴더 목록을 읽는다.
         또한 폴더명을 붙이지 않고 imread 하므로 이미지가 읽히지 않는다.
    또한 z(상부근/하부근) 보간과 output/{target}/{z} 저장 규칙은 이쪽에 없다.
    '''

    # init
    is_raspi_connected = False
    # 아래 points* 는 데이터셋별로 첫 장에서 찍어둔 관심 영역 좌표다.
    # 사용할 데이터셋에 맞는 것을 골라 points 에 할당해야 한다.
    # points = [[1065, 503], [1393, 559], [1341, 849], [951, 765]]
    # points = np.array([[853, 701], [1169, 712], [1171, 965], [762, 940]])
    points5 = np.array([[2158,754], [3312, 1168], [2588, 2125], [1301, 1304]])
    points1 = np.array([[1579, 716], [2542, 1109], [1523, 2020], [556, 1250]])
    points6 = np.array([[2164, 380], [3551, 685], [2714, 1646], [1143, 866]])
    points10 = np.array([[1374, 289], [2462, 630], [1113, 1524], [75, 762]])
    # 아래는 마커 기반으로 3D 좌표를 투영해 warp point를 구하려던 이전 시도의 잔재다.
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
        # 첫 장(i == 0)은 기준 이미지로 삼고 미리 정해둔 좌표를 그대로 쓴다.
        # 두 번째 장부터는 기준 이미지와의 SIFT 매칭으로 Homography를 구해 좌표를 옮긴다.
        # 매칭에 실패하면(retval False) 좌표가 비어 이후 warp에서 문제가 생기므로 확인이 필요하다.
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

        # 디버그용 warp point 찍어보기 (창이 뜨며 키 입력 전까지 멈춘다)
        img_debug = img.copy()
        for wp in warping_point_2d.reshape(4,2):
            cv2.circle(img_debug, wp.astype(np.int16), color=(0,0,255), thickness=-1, radius=10)
        img_debug = cv2.resize(img_debug, (1600, 900))
        image_show(img_debug, f"{i}")

        # 철근 찾기 (predict는 RGB 입력을 기대한다)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_segmentation = seg_predict.predict(img_rgb)

        # perspective warp 실행
        # 원본은 BGR 그대로, segmentation 결과는 grayscale 그대로 같은 변환을 적용한다.
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
    # manual 파이프라인과 달리 카메라 대수를 입력 장수에서 그때그때 계산한다.
    # threshold도 (n-1)/n 이라 "한 시점을 제외한 모든 시점에서 검출된 픽셀"만 남는다.
    image_count = len(warping_img_segmentation)
    weighted_sum_image = np.zeros_like(warping_img_segmentation[0], dtype=np.float32)
    for i, ws_img in enumerate(warping_img_segmentation):
        ws_img_blur = custom_blur(ws_img)
        weighted_sum_image += ws_img_blur * (1 / len(warping_img_segmentation))
    weighted_sum_image = weighted_sum_image.astype(np.uint8)
    image_show(weighted_sum_image)
    # 저장 위치가 output/ 이 아니라 프로젝트 루트라는 점에 주의.
    cv2.imwrite("./weighted_sum_image.png", weighted_sum_image)
    thresholded_weighted_sum_image = np.where(weighted_sum_image > int(255 * ((image_count - 1) / image_count)), 255, 0)
    image_show(thresholded_weighted_sum_image.astype(np.uint8))
    cv2.imwrite("./weighted_sum_thresholded_image.png", thresholded_weighted_sum_image.astype(np.uint8))


if __name__ == '__main__':
    # 하부근(z=0)부터 상부근(z=100)까지 5씩 올려가며 높이별 단면을 모두 뽑는다.
    # 특정 높이 하나만 필요하면 이 반복문 대신 run_manual_warp_pipeline(원하는 z) 를 호출할 것.
    for i in range(0,101,5):
        run_manual_warp_pipeline(i)
    # run_manual_warp_pipeline()
    # run_sift_warp_pipeline()
