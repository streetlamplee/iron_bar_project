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

matplotlib.use('TkAgg')
randomseed(42)

'''
해당 코드 내용
1. z offset을 통해 찾고자 함
'''

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
    # z_offset = [z for z in range(-10, 11)]
    z_offset = [0]
    z_result = []
    z_warp = []
    for z in z_offset:
        target_3d = np.array([
            [843, -840, 100 + z],
            [420, -830, 100 + z],
            [420, -415, 100 + z],
            [843, -412, 100 + z],
        ], dtype = np.float32)

        for i, (r, t) in enumerate(zip(rvec_list, tvec_list)):
            _2d_point = getDisplayCoordinate.get_display_coordinate(cam, r, t, target_3d)
            img = cv2.imread(f'data/image_seg/{i+1}.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (1440, 1080))
            # for j in range(len(_2d_point)):
            #     center = _2d_point[j].astype(np.int32)
            #     img = cv2.circle(img, center, radius = 3, color = (255,0,0), thickness = -1)
            # cv2.imshow("2D test", img)
            # cv2.waitKey(100)
            # cv2.destroyWindow("2D test")

            warped = warp_perspective(img, _2d_point)
            cv2.imwrite(f"z_offset/{i+1}_warp_{z}.png", warped)





        '''
        warp perspective 실행
        '''
        if not os.path.exists('z_offset') or len(os.listdir('z_offset')) == 0:
            warp_image_list = []
            for idx, (picture, picture_seg) in enumerate(zip(picture_list, picture_list_seg)):
                cam_position = picture.cam_position
                image = picture.picture
                image_seg = picture_seg.picture
                # warp_target_area_2d = get_display_coordinate(camera=cam, _3d_point=warp_target_area)

                cv2.imshow("Please click the area to warp", image)
                cv2.setMouseCallback("Please click the area to warp", click_event_warp)
                while True:
                    key = cv2.waitKey(1)
                    if len(clicked_points) == 4:
                        break
                    elif key == 27:
                        break

                warped_image = warp_perspective(image_seg, np.array(clicked_points, dtype = np.float32))
                clicked_points = []
                cv2.imshow("debug", warped_image)
                cv2.waitKey(0)
                cv2.destroyWindow("debug")
                warp_image_list.append(warped_image)

            '''
            warp 된 이미지 저장
            '''
            if not os.path.exists('warp_image'):
                os.mkdir('warp_image')
            for i, w in enumerate(warp_image_list):
                cv2.imwrite(f'warp_image/{i+1}.png', w)
        else:
            warp_image_list = []
            for i in os.listdir('z_offset'):
                if i.endswith(f'_{z}.png'):
                    tmp = cv2.imread(f'z_offset/{i}')
                    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                    warp_image_list.append(tmp)
            z_warp.append(warp_image_list)

    '''
    warp 된 이미지에 한 가중치로 더하기
    '''
    z = -0
    for warp_image_list in z_warp:
        weight_value = 1. / len(warp_image_list)

        weighted_sum_image = warp_image_list[0] * weight_value

        for i in range(1, len(warp_image_list)):
            image_in_iter = warp_image_list[i]
            weighted_sum_image += image_in_iter * weight_value

        # cv2.imshow("weighted_sum_image", weighted_sum_image)
        # cv2.waitKey(100)
        # cv2.destroyWindow("weighted_sum_image")
        cv2.imwrite(f"res_{z}.png", weighted_sum_image)

        '''
        가중치로 더한 이미지에 threshold를 걸어서 확인하기
        '''
        res = weighted_sum_image
        if len(res.shape) == 3:
            res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

        res = np.where((res > 127), 255, 0)
        res = res.astype(np.uint8)

        # cv2.imshow("thres", res)
        # cv2.waitKey(100)
        # cv2.destroyWindow("thres")

        cv2.imwrite(f"res_threshold_{z}.png", res)
        z += 1
        '''
        250502 warp 이미지 최적화
        '''
        continue
        basis_image = warp_image_list[0]

        for w in range(1, len(warp_image_list)):
            warp_in_iter = warp_image_list[w]

            basis_image = basis_image // 2
            warp_in_iter = warp_in_iter // 2

            # brutal force?
            warped_cell_count = 11
            shift = int(1024 / warped_cell_count / 2)
            num_of_cell_255 = -1
            best_offset_x = -1
            best_offset_y = -1
            best_warp_shifted = None
            best_target = None
            for x in range(-shift, shift+1):
                for y in range(-shift, shift+1):
                    warp_in_iter_shifted = image_shift.image_shift(warp_in_iter, x, y)
                    target_in_iter = basis_image + warp_in_iter_shifted
                    # cv2.imshow("warp shifted", warp_in_iter_shifted)
                    # cv2.imshow("basis", basis_image)
                    # cv2.imshow("target iter",target_in_iter)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    if num_of_cell_255 < np.count_nonzero(target_in_iter == 254):
                        num_of_cell_255 = np.count_nonzero(target_in_iter == 254)
                        best_offset_x = x
                        best_offset_y = y
                        best_warp_shifted = warp_in_iter_shifted
                        best_target = target_in_iter
                        print(f'find new best offset. x = {x}, y = {y}, num_255 = {num_of_cell_255}')

            plt.figure(figsize=(20,10))
            plt.subplot(2,3,1)
            plt.title("basis image")
            plt.imshow(basis_image, cmap='gray')
            plt.axis('off')

            plt.subplot(2,3,2)
            plt.title('warp_in_iter')
            plt.imshow(warp_in_iter, cmap='gray')
            plt.axis('off')

            plt.subplot(2,3,5)
            plt.title('warp_in_iter_shifted')
            plt.imshow(best_warp_shifted, cmap='gray')
            plt.axis('off')

            plt.subplot(2,3,3)
            plt.title('target_in_iter')
            plt.imshow(best_target, cmap='gray')
            plt.axis('off')

            plt.show()

            basis_image = best_target


        cv2.imwrite("res_shifted.png", basis_image)









    raise "code after this is not available"
if False:
    z_offset = list(range(0,1))
    for z in tqdm(z_offset):
        original = []
        warped = []

        for index, (p,f,u) in enumerate(zip(pos, focal, up)): ## 결국 카메라 위치에 대한 value를 묶어서 카메라를 정의하고자 함
            x_offset = np.random.randint(-100, 101)
            y_offset = np.random.randint(-100, 101)



            cam.position = p
            cam.focal_point = f
            cam.up = u
            plotter.show(auto_close=False)
            plotter.render()

            before_compute_bias_position = p + np.array([x_offset, y_offset, 0])
            input_area = np.array([points_top[8][8],
                                   points_top[8][11],
                                   points_top[11][8],
                                   points_top[11][11]])
            # 수정 전
            warp_, origin_ = warpPerspective(plotter=plotter, points=input_area,w=w, h=h, z_offset=z, real_camera_position=before_compute_bias_position)


            # rendered_img = plotter.screenshot(return_img=True)
            # rendered_img_gray = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2GRAY)
            if f"ret{index}" not in cache.cache.keys() and f"points{index}" not in cache.cache.keys():
                ret, points = ComputeCamPositionBias(MarkerPosition, plotter, w, h, marker_size, aruco_dict, camera_matrix, dist_coeff)
                cache.set_cache(f"ret{index}", ret)
                cache.set_cache(f"points{index}", points)
                cache.save_cache()
            else:
                ret = cache.get_cache(f"ret{index}")
                points = cache.get_cache(f"points{index}")
                points = np.array(points)
            # points = np.mean(points,axis=0).reshape((3,))
            result = []
            for isSquare, point in zip(ret, points):
                if isSquare:
                    result.append(point)
            # points = np.mean(points, axis=0)
            points = np.mean(np.array(result), axis = 0)
            print(f"mean of predict cam_pos")
            print(points)

            diff = points - p
            predict_x_offset = diff[0]
            predict_y_offset = diff[1]
            predict_z_offset = diff[2]
            cam.position = p
            cam.focal_point = f#_ + np.array([diff[0], diff[1], diff[2]])
            cam.up = u
            plotter.show(auto_close=False)
            plotter.render()

            after_compute_bias_position = points.reshape(3,)
            warp, origin = warpPerspective(plotter=plotter, points=input_area,w=w, h=h, z_offset=z, real_camera_position=after_compute_bias_position)

            # 비교군 생성용 코드


            warped.append(warp)
            original.append(origin)


        #########################################################################
        weight = (1 / len(warped))
        alpha_blended = warped[0] * weight
        cv2.imwrite(f"tmp_warp/warp0.png", warped[0])
        for i in range(1,len(warped)):
            alpha_blended = cv2.addWeighted(alpha_blended.astype(np.uint8), 1.0, warped[i], weight, 0)
            cv2.imwrite(f"tmp_warp/warp{i}.png", warped[i])
        # threshold = np.min(warped, axis=(0, 1, 2)) + (np.max(warped, axis=(0, 1, 2)) - np.min(warped, axis=(0, 1, 2))) * (1 / (2*len(warped)))
        threshold = np.array([64,64,64]) + (np.max(warped, axis=(0, 1, 2)) - np.array([64,64,64])) * (1 / (2*len(warped)))
        alpha_blended_Under_threshold = np.where(alpha_blended <= threshold, 127, 255).astype(np.uint8)

        plt.figure(figsize=(9,9))
        for i, (o, w, o_, w_) in enumerate(zip(original, warped, original_, warped_)):
            plt.subplot(5,4,4*i+1)
            plt.imshow(cv2.cvtColor(o, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.subplot(5,4,4*i+2)
            plt.imshow(cv2.cvtColor(o_, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.subplot(5,4,4*i+3)
            plt.imshow(cv2.cvtColor(w, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.subplot(5,4,4*i+4)
            plt.imshow(cv2.cvtColor(w_, cv2.COLOR_BGR2RGB))
            plt.axis("off")

        plt.subplot(5,4,4 * len(warped)+1)
        plt.imshow(cv2.cvtColor(alpha_blended.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.axis("off")
        cv2.imwrite(f"addWeighted/{z}_afterComputeBias.png", alpha_blended)

        plt.subplot(5,4,4*len(warped)+2)
        plt.imshow(cv2.cvtColor(alpha_blended_, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"addWeighted/{z}_beforeComputeBias.png", alpha_blended_)
        plt.axis("off")

        plt.subplot(5,4,4 * len(warped)+3)
        plt.imshow(cv2.cvtColor(alpha_blended_Under_threshold.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.axis("off")
        cv2.imwrite(f"alphaBlendThreshold/{z}_afterComputeBias.png", alpha_blended_Under_threshold)

        plt.subplot(5,4,4*len(warped)+4)
        plt.imshow(cv2.cvtColor(alpha_blended_Under_threshold_, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"alphaBlendThreshold/{z}_beforeComputeBias.png", alpha_blended_Under_threshold_)
        plt.axis("off")


        plt.show()
        plt.waitforbuttonpress(0)
        #########################################################################



    plotter.close()

# forward = np.array(plotter.camera.direction)
# up = np.array(plotter.camera.up)
# right = np.cross(up, forward)
#
# R = np.column_stack((right,up, -forward))
# print("R :")
# print(R)
# t = -1 * (R @ np.array(plotter.camera.position))
# print("t :")
# print(t)
#
# extrinsicM = np.column_stack((R, t))
# extrinsicM = np.append(extrinsicM, [0,0,0,1])
# print("extrinsicM:")
# print(extrinsicM)
#
# # print("model_transform_matrix:")
# # print(plotter.camera.model_transform_matrix)
#
# yaw = np.degrees(np.arctan2(forward[0], forward[2]))
# pitch = np.degrees(np.arcsin(forward[1]))
#
# print(f"yaw: {yaw}, pitch: {pitch}, roll: {plotter.camera.roll}")

# plotter.render()
# plotter.store_image = True  # last_image and last_image_depth
# plotter.close()
# img = plotter.last_image()
# img = np.array(img, dtype=np.uint8)

# plt.imshow(img)
# plt.show()



#
#
# cx = intrinsic[0,2]
# # get screen image
# cy = intrinsic[1,2]
# f = intrinsic[0,0]
#
# # convert the principal point to window center (normalized coordinate system) and set it
# wcx = -2*(cx - float(w)/2) / w
# wcy =  2*(cy - float(h)/2) / h
# plotter.camera.SetWindowCenter(wcx, wcy)
#
# # convert the focal length to view angle and set it
# view_angle = 180 / math.pi * (2.0 * math.atan2(h/2.0, f))
# plotter.camera.SetViewAngle(view_angle)
#
#
# #
# # extrinsics
# #
#
# # apply the transform to scene objects
# plotter.camera.SetModelTransformMatrix(trans_to_matrix(extrinsic))
#
# # the camera can stay at the origin because we are transforming the scene objects
# plotter.camera.SetPosition(0, 0, 0)
#
# # look in the +Z direction of the camera coordinate system
# plotter.camera.SetFocalPoint(0, 0, 1)
#
# # the camera Y axis points down
# plotter.camera.SetViewUp(0,-1,0)
#
#
# #
# # near/far plane
# #
#
# # ensure the relevant range of depths are rendered
# # depth_min = 0.1
# # depth_max = 100
# # p.camera.SetClippingRange(depth_min, depth_max)
# # # depth_min, depth_max = p.camera.GetClippingRange()
# plotter.renderer.ResetCameraClippingRange()
#
# plotter.show()
# plotter.render()
# plotter.store_image = True  # last_image and last_image_depth
# plotter.close()
#
#
# # get screen image
# img = plotter.last_image
#
# # get depth
# # img = p.get_image_depth(fill_value=np.nan, reset_camera_clipping_range=False)
# # img = plotter.last_image_depth
#
# plt.figure()
# plt.imshow(img)
# plt.title('Depth image')
# plt.xlabel('X Pixel')
# plt.ylabel('Y Pixel')
# plt.show()