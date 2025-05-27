import gc
import matplotlib
import os
import json
from tqdm import tqdm
import getDisplayCoordinate
from find_cross_point_model.predict import predict_one_image
from n_blur import custom_blur
from n_brute_force_gpu import brute_force_best_warp
from n_randomseed import randomseed
from n_camera import Camera
from n_picture import Picture
from solvePnP import solvePnP
from n_warp import warp_perspective
import torch
import torch.nn.functional as F
import itertools
import numpy as np
import cv2

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


clicked_points_pnp = []
def click_event_pnp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points_pnp) < 4:
            clicked_points_pnp.append([x,y])

# =====================================
# 🔧 4점 기반 perspective transform 행렬을 배치로 구하는 함수
# =====================================
def get_perspective_transform_batch(src_points, dst_points):
    # src_points: [N, 4, 2], dst_points: [N, 4, 2]
    # 각 조합마다 하나의 perspective 행렬을 구해 [N, 3, 3] shape로 반환

    def _build_A_matrix(src, dst):
        # A @ h = b 형태의 선형 시스템을 구성 (DLT 알고리즘)
        N = src.shape[0]
        A = torch.zeros(N, 8, 8).to(src.device)
        b = torch.zeros(N, 8).to(src.device)
        for i in range(4):
            x, y = src[:, i, 0], src[:, i, 1]
            u, v = dst[:, i, 0], dst[:, i, 1]
            A[:, 2 * i] = torch.stack(
                [x, y, torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x),
                 -u * x, -u * y], dim=1)
            A[:, 2 * i + 1] = torch.stack(
                [torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x), x, y, torch.ones_like(x),
                 -v * x, -v * y], dim=1)
            b[:, 2 * i] = u
            b[:, 2 * i + 1] = v
        return A, b

    A, b = _build_A_matrix(src_points, dst_points)  # 선형 시스템 구성
    h = torch.linalg.solve(A, b).unsqueeze(-1)  # h 벡터 구하기 → [N, 8, 1]
    H = torch.cat([h, torch.ones(h.shape[0], 1, 1).to(h.device)], dim=1)  # 마지막 항 추가 → [N, 9, 1]
    H = H.view(-1, 3, 3)  # 3x3 변환 행렬로 reshape → [N, 3, 3]
    return H


# =====================================
# 🔧 주어진 perspective 행렬로 grid를 만들어 이미지 warp
# =====================================
def warp_with_perspective_grid(img_tensor, M, out_size):
    # img_tensor: [1, C, H, W], M: [N, 3, 3]
    # out_size: (H, W) — 출력 이미지 크기
    # 출력: [N, C, H, W] — 모든 조합의 warping 결과

    N, C, H, W = M.shape[0], *img_tensor.shape[1:]

    # 2D 좌표 grid 생성 (x, y)
    y, x = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W), indexing='ij')
    ones = torch.ones_like(x)
    base_grid = torch.stack([x, y, ones], dim=-1).to(M.device)  # [H, W, 3]
    base_grid = base_grid.view(-1, 3).T  # [3, H*W] — homogeneous 좌표계

    warped_imgs = []
    for i in range(N):
        grid = M[i] @ base_grid  # perspective 변환 적용 → [3, H*W]
        grid = grid[:2] / grid[2:]  # z로 나눠서 정규화 → [2, H*W]
        grid = grid.T.view(H, W, 2)  # [H, W, 2]

        # [-1, 1]로 정규화 (F.grid_sample용)
        grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
        grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1

        # grid_sample로 warping 실행
        warped = F.grid_sample(img_tensor, grid.unsqueeze(0), align_corners=True)
        warped_imgs.append(warped)

    return torch.cat(warped_imgs, dim=0)  # [N, C, H, W] — 모든 warp 결과

if __name__ == "__main__":
    print(torch.cuda.is_available())
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
        [1170, -1210, 100],
        [0, -1185, 100],
        [0, 0, 100],
        [1227, 0, 100],
        # [843, -840, 100],
        # [420, -830, 100],
        # [420, -415, 100],
        # [843, -412, 100],
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
        [1170, -1210, 100],
        [0, -1185, 100],
        [0, 0, 100],
        [1227, 0, 100],
        # [843, -840, 100],
        # [420, -830, 100],
        # [420, -415, 100],
        # [843, -412, 100],
    ], dtype = np.float32)

    point_basis_image = None
    basis_image = None
    for i, (r, t) in enumerate(zip(rvec_list, tvec_list)):
        gc.collect()
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # this is for auto mode
        _2d_point = getDisplayCoordinate.get_display_coordinate(cam, r, t, target_3d)

        # this is for manual mode
        _2d_point = _2d_coordinate[f'{i}']
        _2d_point = np.array(_2d_point, dtype = np.float32)
        point_seg_image = cv2.imread(f'model_outputs/{i + 1}.png')
        point_seg_image = cv2.cvtColor(point_seg_image, cv2.COLOR_BGR2GRAY)
        point_seg_image = cv2.resize(point_seg_image, (1360, 1020))
        seg_image = cv2.imread(f'data/image_seg/{i + 1}.png')
        seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)
        seg_image = cv2.resize(seg_image, (1360, 1020))


        '''
        250514 test code
        '''
        for p in range(len(_2d_point)):
            point = _2d_point[p].astype(np.int32)
            cv2.circle(seg_image, point, radius=5, thickness=-1, color=(255, 255, 255))

        cv2.imwrite(f'250514/{i+1}.png', seg_image)
        # warp_image = warp_perspective(iron_img, _2d_point)
        # cv2.imwrite(f'warp_image/{i+1}.png', warp_image)
        #
        # continue

        '''
        x, y offset을 이용한 brutal force
        '''
        if i == 0:
            point_seg_warped = warp_perspective(point_seg_image, _2d_point)
            point_seg_warped = custom_blur(point_seg_warped)
            point_basis_image = point_seg_warped
            point_basis_image = np.astype(point_basis_image, np.float32)
            seg_warped = warp_perspective(seg_image, _2d_point)
            predicted_point_warped = predict_one_image(seg_warped)
            predicted_point_warped = custom_blur(predicted_point_warped)
            basis_image = predicted_point_warped
            basis_image = np.astype(basis_image, np.float32)
            '''
            250514 확인용 코드
            '''
            if not os.path.exists('250514'):
                os.mkdir('250514')
                os.mkdir('250514/iter_point')
                os.mkdir('250514/iter_line')

            cv2.imwrite(f'250514/iter_point/{i + 1}.png', np.reshape(point_seg_warped, (1024, 1024)).astype(np.uint8))
            cv2.imwrite(f'250514/iter_line/{i + 1}.png', np.reshape(seg_warped, (1024, 1024)).astype(np.uint8))
            cv2.imwrite(f'250514/iter_{i + 1}_basis_image.png', point_basis_image.astype(np.uint8))
            cv2.imwrite(f'250514/iter_{i + 1}_basis_seg_image.png', basis_image.astype(np.uint8))

            continue


        best_warped, best_seg_warped, _, _ = brute_force_best_warp(point_seg_image, seg_image, basis_image, _2d_point, offset = 8, i = i)




        # 기준 이미지와 best warped 이미지 블렌딩
        point_basis_image = point_basis_image * i / (i + 1) + np.reshape(best_warped, (1024, 1024)) * 1. / (i + 1)
        basis_image = basis_image * i / (i + 1) + np.reshape(best_seg_warped, (1024, 1024)) * 1 / (i + 1)
        # 결과 시각화
        '''
        250514 확인용 코드
        '''
        if not os.path.exists('250514'):
            os.mkdir('250514')
            os.mkdir('250514/iter_point')
            os.mkdir('250514/iter_line')

        cv2.imwrite(f'250514/iter_point/{i+1}.png', np.reshape(best_warped,(1024,1024)).astype(np.uint8))
        cv2.imwrite(f'250514/iter_line/{i+1}.png', np.reshape(best_seg_warped, (1024, 1024)).astype(np.uint8))
        cv2.imwrite(f'250514/iter_{i+1}_basis_image.png', point_basis_image.astype(np.uint8))
        cv2.imwrite(f'250514/iter_{i+1}_basis_seg_image.png', basis_image.astype(np.uint8))


        import matplotlib.pyplot as plt

        # plt.imshow(basis_image.astype(np.uint8), cmap='gray')
        # plt.axis('off')
        # plt.title("Blended Image")
        # plt.show()

    point_basis_image = np.astype(point_basis_image, np.uint8)
    # cv2.imshow("result", basis_image)
    # cv2.waitKey(10000)
    # cv2.destroyWindow("result")
    cv2.imwrite("res_point.png", point_basis_image)

    basis_image = np.astype(basis_image, np.uint8)

    cv2.imwrite('res_iron_pointwise.png', basis_image)

    thres = np.where(point_basis_image >= 254. * 0.875, 255, 0)
    thres_seg = np.where(basis_image >= 254. * 0.875, 255, 0)

    thres = thres.astype(np.uint8)
    thres_seg = thres_seg.astype(np.uint8)

    # cv2.imshow("thres", thres)
    # cv2.waitKey(10000)
    # cv2.destroyWindow("thres")
    cv2.imwrite('res_point_thres.png', thres)
    cv2.imwrite('res_iron_pointwise_thres.png', thres_seg)
