import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import os
import json
import getDisplayCoordinate
import image_shift
from n_randomseed import randomseed
from tqdm import tqdm
from n_camera import Camera
from n_picture import Picture
from solvePnP import solvePnP
from n_warp import warp_perspective
import itertools
import cupy as cp

matplotlib.use('TkAgg')
randomseed(42)


# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

# 3D 공간 선언
# plotter = pyvistaEnvironment.make_plotter(w=w,
#                                           h=h,
#                                           aruco_dict=aruco_dict,
#                                           MarkerPosition=MarkerPosition,
#                                           marker_size=marker_size)

# plotter.add_points(np.array(pos, dtype=np.float32), render_points_as_spheres=True, point_size=20.0, color="red")

# num_cell = pyvistaEnvironment.num_cell
# cell_size = pyvistaEnvironment.cell_size
# chess_board_center = pyvistaEnvironment.chess_board_center
# points_top = pyvistaEnvironment.points_top
# points_btm = pyvistaEnvironment.points_btm



# plotter = pv.Plotter(window_size=[w,h],off_screen=True)
# plotter.show_axes()
# plotter.background_color = (200,200,200)
#
# # 3D 공간에 철근 표현
# points_top = pyvistaEnvironment.setXYZ(20, 20, 200, True)
# points_btm = pyvistaEnvironment.setXYZ(20, 20, 200, False)
# lines = pyvistaEnvironment.makeLine(points_top)
# lines.extend(pyvistaEnvironment.makeLine(points_btm))
#
# for line in lines:
#     conn = np.hstack([[len(line)], np.arange(len(line))])
#
#     polyline = pv.PolyData(line)
#     polyline.lines = conn
#     plotter.add_mesh(polyline, color=(64,64,64), line_width = 10)
#
# # 3D 공간에 마커 만들기 (수직으로 세워서, 중앙을 바라보도록)
# for i, center in enumerate(MarkerPosition):
#     marker_id = i
#     direction = np.array(MarkerPosition[i] - [2000, 2000, 0])
#     direction[2] = 0
#     np.abs(direction)
#     marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, borderBits=1)
#     marker_img_color = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2RGB)
#     marker_img_color = marker_img_color[::-1, :, :]
#     plane = pv.Plane(center = center, direction=direction, i_size=marker_size, j_size = marker_size)
#     texture = pv.Texture(marker_img_color)
#     plotter.add_mesh(plane, texture=texture, ambient=1.0, diffuse=0.0, specular = 0.0)
#
# # 3D 공간에 chessboard 만들기
# num_cell = 8
# cell_size = 50
# chess_board = Callibration.create_chessboard(num_cell,num_cell,cell_size)
# texture_chess_board = pv.Texture(chess_board)
# chess_board_center = (-500, -500, 0)
# chess_board_plane = pv.Plane(center = chess_board_center, i_size= num_cell*cell_size, j_size = num_cell*cell_size)
# plotter.add_mesh(chess_board_plane, texture = texture_chess_board, ambient=1.0, diffuse=0.0, specular = 0.0)
clicked_points = []
def click_event_warp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x,y))

clicked_points_pnp = []
def click_event_pnp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points_pnp.append([x,y])


if __name__ == "__main__":
    '''
    카메라 calibration
    '''
    cam = Camera("chessboardImage")

    '''
    카메라 사진 준비
    
    picture_list << Picture 객체를 가지는 list
    Picture << get_picture() 와 get_cam_position() 내장 함수 사용 가능
    '''
    with open('data/data.json', 'r') as json_f:
        data = json.load(json_f)

    with open('data/data_seg.json', "r") as json_f:
        data_seg = json.load(json_f)

    if len(os.listdir('data/image_seg')) != len(data_seg.keys()):
        raise "some Image doesn't have json data. 모든 이미지의 정보가 json에 적혀 있지 않습니다."

    picture_list = []
    for idx, key in enumerate(data):
        if "image_path" not in data[key] and "cam_position" not in data[key]:
            raise "data.json is not available. data.json 파일을 확인해주세요"
        picture = Picture(data[key])
        picture_list.append(picture)

    picture_list_seg = []
    for idx, key in enumerate(data_seg):
        if "image_path" not in data_seg[key] and "cam_position" not in data_seg[key]:
            raise "data_seg.json is not available. data_seg.json 파일을 확인해주세요"
        picture = Picture(data_seg[key])
        picture_list_seg.append(picture)
    '''
    PnP data 없으면 PnP 실행 >> rvec 과 tvec 구하기
    '''
    _3d_coordinate = np.array([
        [843, -840, 100],
        [420, -830, 100],
        [420, -415, 100],
        [843, -412, 100],
    ], dtype = np.float32)          ## 사진 순서 : 첫 사진 기준 좌상 > 우상 > 우하 > 좌하 ## 하부근 z 축 값 12, 상부근 z 축 값 100
    if os.path.exists('cache/pnp.json'):
        with open('cache/pnp.json', 'r') as f:
            _2d_coordinate = json.load(f)

    else:

        _2d_coordinate = dict()

        for idx, picture in enumerate(picture_list):
            cv2.imshow("Please click the point which is pair to _3d_coordinate", picture.picture)
            cv2.setMouseCallback("Please click the point which is pair to _3d_coordinate", click_event_pnp)
            while True:
                key = cv2.waitKey(1)
                if len(clicked_points_pnp) == 4:
                    break
                elif key == 27:
                    break

            _2d_coordinate[idx] = list(clicked_points_pnp)
            clicked_points_pnp = []

        with open('cache/pnp.json', 'w') as f:
            json.dump(_2d_coordinate, f, indent=4)

    rvec_list = [] # 각 사진을 찍은 카메라에 대한 rvec (사진 순서대로)
    tvec_list = [] # 각 사진을 찍은 카메라에 대한 tvec (사진 순서대로)


    for idx, clicked_point_pnp in enumerate(_2d_coordinate.items()):
        tmp_rvec, tmp_tvec = solvePnP(cam, _3d_coordinate, np.array(clicked_point_pnp[1], dtype = np.float32))
        rvec_list.append(tmp_rvec)
        tvec_list.append(tmp_tvec)

    '''
    rvec과 tvec을 알고 있으니, 3D >> 2D 좌표로 변환가능  
    '''

    target_3d = np.array([
        # [1200, -1200, 100],
        # [0, -1200, 100],
        # [0, 0, 100],
        # [1200, 0, 100],
        [843, -840, 100],
        [420, -830, 100],
        [420, -415, 100],
        [843, -412, 100],
    ], dtype = np.float32)

    basis_image = None
    for i, (r, t) in enumerate(zip(rvec_list, tvec_list)):
        _2d_point = getDisplayCoordinate.get_display_coordinate(cam, r, t, target_3d)


        img = cv2.imread(f'data/image_seg/{i+1}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1440, 1080))
        # colormap = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        # for i, point in enumerate(_2d_point):
        #     x, y = int(point[0]), int(point[1])
        #     cv2.circle(img, (x,y), radius=3, thickness = -1, color=colormap[i])
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        # cv2.destroyWindow('img')
        '''
        x, y offset을 이용한 brutal force
        '''
        if i == 0:
            warped = warp_perspective(img, _2d_point)
            basis_image = warped
            basis_image = np.astype(basis_image, np.float32)
            continue

        offset = 2
        r = range(-offset, offset+1)

        max_num_255 = -1
        best_offset_arr = None
        _2d_point = cp.asarray(_2d_point, cp.int32)

        print(f"Start Brutal Force Searching on Image {i+1}")
        for combo in tqdm(itertools.product(r, repeat=8)):
            x1, y1, x2, y2, x3, y3, x4, y4 = combo
            offset_arr = cp.array([[x1, y1],
                                   [x2, y2],
                                   [x3, y3],
                                   [x4, y4]], dtype = cp.float32)
            _2d_point_shifted = _2d_point + offset_arr
            _2d_point_shifted = _2d_point_shifted.astype(cp.float32)
            '''
            warp perspective 실행
            '''
            _2d_point_shifted_cpu = cp.asnumpy(_2d_point_shifted)
            warped = warp_perspective(img, _2d_point_shifted_cpu)
            # cv2.imwrite(f"warp_image/{i+1}.png", warped)

            basis_iter = cp.asarray(basis_image, dtype=cp.float32)
            basis_iter = basis_iter * i / (i + 1)

            warped = cp.asarray(warped, cp.float32)
            warped = warped * 1. / (i + 1)

            target_iter = basis_iter + warped

            if cp.count_nonzero((target_iter >= 255.)) > max_num_255:
                max_num_255 = cp.count_nonzero((target_iter >= 255.))
                best_offset_arr = offset_arr
                print(f"{combo} on target")
        best_offset = best_offset_arr + _2d_point
        best_offset_cpu = cp.asnumpy(best_offset)
        warped_best = warp_perspective(img, best_offset_cpu)
        warped_best = np.astype(warped_best, np.float32)

        basis_image = basis_image / 2.
        warped_best = warped_best / 2.



        plt.figure(figsize=(20,10))

        plt.subplot(1,3,1)
        plt.imshow(basis_image.astype(np.uint8), cmap='gray')
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(warped_best.astype(np.uint8), cmap='gray')
        plt.axis('off')

        basis_image = basis_image + warped_best

        plt.subplot(1,3,3)
        plt.imshow(basis_image.astype(np.uint8), cmap='gray')
        plt.axis('off')

        plt.show()

    basis_image = np.astype(basis_image, np.uint8)
    cv2.imshow("result", basis_image)
    cv2.waitKey(0)
    cv2.destroyWindow("result")
    cv2.imwrite("res.png", basis_image)

    thres = np.where(basis_image >= 255. * (0.875), 255, 0)
    thres = thres.astype(np.uint8)

    cv2.imshow("thres", thres)
    cv2.waitKey(0)
    cv2.destroyWindow("thres")
    cv2.imwrite('res_thres.png', thres)
