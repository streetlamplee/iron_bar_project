import numpy as np
import cv2
import math

def ComputeCamPositionBias(MarkerPosition:np.array,
                           plotter:pv.Plotter,
                           w, h, marker_size,
                           aruco_dict,
                           camera_matrix,
                           dist_coeff,
                           rv=None,
                           tv=None,
                           display=True):
    useCrop = True
    predictCamPositions = []
    ret_li = []
    for f in MarkerPosition:
        markerImagePosition = []
        plotter.camera.focal_point = f
        plotter.camera.up = (0., 0., 1.)
        focal_length = (h / 2.0) / math.tan(math.radians(plotter.camera.view_angle) / 2.0)
        # plotter.camera.zoom(1.5)

        plotter.disable_shadows()
        plotter.show(auto_close=False)
        plotter.render()

        img = plotter.screenshot(return_img=True)
        input = img
        m = GetObjectPixelInImage(plotter.camera.position, f, marker_size, focal_length)
        if useCrop:
            border_top = int(h/2 - m) if h/2-m >= 0 else 0
            border_btm = int(h/2 + m) if h/2+m < h else h
            border_left = int(w/2-m) if w/2-m >= 0 else 0
            border_right = int(w/2+m) if w/2+m < w else w
            input = input[border_top : border_btm, border_left : border_right]
        input_gray = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)

        params = cv2.aruco.DetectorParameters()
        c, idx, reject = cv2.aruco.detectMarkers(input_gray, aruco_dict, parameters=params)
        if idx is not None:
            for corner in c: # c는 마커의 각 꼭지점의 2D 좌표
                corner = np.array(corner).reshape((4,2))
                if useCrop:
                    adjustCroppedArr = np.array([ [int(w/2 - m), int(h/2 - m)]  ] * 4)
                    corner = corner + adjustCroppedArr
                markerImagePosition.extend(corner.astype(np.float64))

                # print(f"markerCenterPosition: {f}")
                # print(f"detectPosition: {(x, y)}")
                img_markers = cv2.aruco.drawDetectedMarkers(input.copy(), c, idx)
                if useCrop:
                    img_result = img.copy()
                    img_result[border_top : border_btm, border_left : border_right] = img_markers
                    if display:
                        cv2.imshow('Detected Markers', img_result)
                else:
                    if display:
                        cv2.imshow('Detected Markers', img_markers)
                if display:
                    cv2.waitKey(1000)
            if display:
                cv2.destroyWindow("Detected Markers")


            dir = np.array([2000, 2000, 0]) - f
            dir[2] = 0

            markerCornerWorldPosition = get_marker_corners_in_world(f, marker_size, dir)


            # 반환: 4x3 numpy array, 순서는
            # [좌측 상단, 우측 상단, 우측 하단, 좌측 하단]
            # (각 꼭지점이 world 좌표에서의 3D 위치)
            ## 여기까지 검증 완료 >> get_marker_corners_in_world 함수 정상 작동
            camera_matrix = camera_matrix.astype(np.float32)

            #######################################################################
            # 250401 3D 좌표와 2D 좌표의 순서가 맞는지 확인해보기 (완료시 삭제할 것)
            #######################################################################
            #
            # p = plotter.camera.position
            # fp = plotter.camera.focal_point
            # u = plotter.camera.up
            # test_plotter = pv.Plotter(off_screen=True, window_size=[w,h])
            # test_camera = test_plotter.camera
            # test_camera.position = p
            # test_camera.focal_point = fp
            # test_camera.up = u
            #
            # test_img2 = img.copy()
            # test_img2 = np.array(test_img2)
            # colorMap = ("red", "blue", "black", "green")
            # ids = 0
            # for dim3pos, dim2pos in zip(markerCornerWorldPosition, np.asarray(markerImagePosition)):
            #
            #     dim2pos = dim2pos.astype(np.int32)
            #
            #     test_plotter.add_points(dim3pos,render_points_as_spheres=True,point_size=10.0, color = colorMap[ids])
            #
            #     cv2.circle(test_img2, dim2pos, 10, (0,0,255), 5)
            #     cv2.putText(test_img2, str(ids), dim2pos-np.array([-20,-0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            #
            #     ids += 1
            # test_plotter.show(auto_close=False)
            # test_plotter.render()
            # test_img = test_plotter.screenshot(return_img=True)
            # test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
            # for pos in np.array(markerImagePosition):
            #     pos = pos.astype(np.int32)
            #     cv2.circle(test_img, pos, 10, (0,0,255), 5)
            # test_img = np.asarray(test_img)
            #
            # cv2.imshow("test_img", test_img)
            # cv2.imshow("test_img2", test_img2)
            #
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()





            #######################################################################
            # 250401 3D 좌표와 2D 좌표의 순서가 맞는지 확인해보기 (완료시 삭제할 것)
            #######################################################################

            ##########################################################
            #test

            np_marker_image_position = np.asarray(markerImagePosition)
            corner1_x = np_marker_image_position[0][0]
            corner1_y = np_marker_image_position[0][1]
            corner2_x = np_marker_image_position[1][0]
            corner4_y = np_marker_image_position[3][1]

            diff_x = corner2_x - corner1_x
            diff_y = corner4_y - corner1_y

            if diff_y >=100 or diff_x >= 100:
                ret = 1
            else:
                ret = 0

            ret_li.append(ret)



            ##########################################################

            retval, rvec, tvec = cv2.solvePnP(markerCornerWorldPosition,
                                              np.asarray(markerImagePosition),
                                              camera_matrix,
                                              dist_coeff.reshape(5,-1),
                                              # useExtrinsicGuess=True,
                                              # rvec=rv,
                                              # tvec=tv,
                                              flags=0)

            if retval:
                # print(f"rvec: {rvec.flatten()}")
                # print(f"tvec: {tvec.flatten()}")
                R, _ = cv2.Rodrigues(rvec)
                R_inv = np.linalg.inv(R)
                predictCamPosition = -R_inv.dot(tvec)
                # print(f"predict cam_pos:")
                # print(f"{predictCamPosition}")

                predictCamPositions.append(predictCamPosition.tolist())

            else:
                print(f"Failed to estimate Camera Pose")

            # cv2.imshow("drawAxis", img)
            # cv2.waitKey(0)
            # cv2.destroyWindow("drawAxis")
            ###########################################

        else:
            img_reject = img.copy()
            if reject is not None:
                for candidate in reject:
                    # candidate shape: (1, 4, 2); candidate[0]는 4개의 코너 좌표
                    candidate[0] = candidate[0] + np.array([[int(w/2 - m),int(h/2 - m)],[int(w/2 - m),int(h/2 - m)],[int(w/2 - m),int(h/2 - m)],[int(w/2 - m),int(h/2 - m)]])
                    pts = candidate[0].astype(int)
                    # 다각형을 빨간색 선 (BGR: (0,0,255)), 두께 2로 그림
                    cv2.polylines(img_reject, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            if display:
                cv2.imshow('Rejected Markers', img_reject)
                cv2.waitKey(1000)
                cv2.destroyWindow('Rejected Markers')


    print(f"cam_pos: {plotter.camera.position}")
    return ret_li, predictCamPositions

def GetObjectPixelInImage(cameraPosition, objectPosition, objectSize, focal_length):
    if type(cameraPosition) != np.ndarray:
        cameraPosition = np.array(cameraPosition)
    if type(objectPosition) != np.ndarray:
        objectPosition = np.array(objectPosition)
    distance = np.linalg.norm(cameraPosition - objectPosition)

    res = (focal_length * objectSize) / distance

    return res

def get_marker_corners_in_world(center,
                                marker_size,
                                marker_normal,
                                world_up=np.array([0, 0, 1], dtype=np.float32)) -> np.array:
    """
    center: 마커의 중심점 (3D 좌표, numpy array)
    marker_size: 마커의 한 변 길이
    marker_normal: 마커가 속한 평면의 법선 벡터 (3D, numpy array)
    world_up: 월드에서의 up 방향 (기본값: [0, 1, 0])

    반환: 4x3 numpy array, 순서는
         [좌측 상단, 우측 상단, 우측 하단, 좌측 하단]
         (각 꼭지점이 world 좌표에서의 3D 위치)
    """
    # marker_normal을 정규화
    marker_normal = marker_normal / np.linalg.norm(marker_normal)

    # marker의 up 방향을 계산: world_up을 marker 평면에 사영
    marker_up = world_up - np.dot(world_up, marker_normal) * marker_normal
    norm_marker_up = np.linalg.norm(marker_up)
    if norm_marker_up < 1e-6:
        raise ValueError("marker_normal과 world_up이 평행하여 marker의 up 방향을 정의할 수 없습니다.")
    marker_up = marker_up / norm_marker_up

    # 오른쪽 방향: marker_up과 marker_normal의 외적 (오른손 좌표계)
    marker_right = np.cross(marker_up, marker_normal)
    marker_right = marker_right / np.linalg.norm(marker_right)

    half_size = marker_size / 2.0

    # 각 꼭지점 계산:
    # 좌측 상단: center - marker_right*half_size + marker_up*half_size
    # 우측 상단: center + marker_right*half_size + marker_up*half_size
    # 우측 하단: center + marker_right*half_size - marker_up*half_size
    # 좌측 하단: center - marker_right*half_size - marker_up*half_size
    top_left = center - marker_right * half_size + marker_up * half_size
    top_right = center + marker_right * half_size + marker_up * half_size
    bottom_right = center + marker_right * half_size - marker_up * half_size
    bottom_left = center - marker_right * half_size - marker_up * half_size

    top_left = tuple(top_left)
    top_right = tuple(top_right)
    bottom_right = tuple(bottom_right)
    bottom_left = tuple(bottom_left)

    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float64)
    return corners
