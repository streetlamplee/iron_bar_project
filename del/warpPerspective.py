import numpy as np
import cv2
import math

def warpPerspective(plotter:pv.Plotter,
                    real_camera_position,
                    save_img = False,
                    points:np.array = None,
                    w=1600,
                    h=900,
                    z_offset = 0
                    ):
    # print(points)
    display_points = []
    vtkPoints = []
    position = plotter.camera.position
    focal_point = plotter.camera.focal_point
    up = plotter.camera.up
    view_angle = plotter.camera.GetViewAngle()
    if points is None:
        return None
    for point in points:
        if z_offset != 0:
            point = point + np.array([0,0,z_offset])
        # 방식 3 코드
        camera_params = {
            'position': position,  # 카메라 위치
            'focal_point': focal_point,  # 바라보는 점
            'view_up': up,  # 업 벡터
            'view_angle': view_angle,  # 수직 시야각 (degree)
            'near': 0.1,
            'far': 100
        }
        # 뷰포트: (x, y, width, height)
        viewport = (0, 0, w, h)

        # 월드 좌표의 한 점 (예: 원점)
        world_pt = list(point)
        display_coord = get_computed_display_value(world_pt, camera_params, viewport, real_camera_position)
        display_coord = [int(display_coord[0]), int(viewport[3] - display_coord[1])]

        # print(f"plotter method: {[x, viewport[3]-y]}")
        # print(f"vtk method: {point_2d}")
        # print(f"gpt method: {display_coord}")
        # print("")

        vtkPoints.append(display_coord)

    vtkPoints = np.array(vtkPoints, dtype=np.float32)
    dst = np.float32([[0,0],[512,0],[0,512],[512,512]])
    M = cv2.getPerspectiveTransform(vtkPoints, dst)

    srcImg = plotter.screenshot(return_img=True)
    srcImg = np.array(srcImg)

    warpedImg = cv2.warpPerspective(srcImg, M, (512,512))

    if save_img:
        cv2.imwrite(f"original_{z_offset}_{plotter.camera.up[0]}.png", srcImg)
        cv2.imwrite(f"warped_{z_offset}_{plotter.camera.up[0]}.png", warpedImg)


    return warpedImg, srcImg
    # else:
    #     img = plotter.screenshot(return_img=True)
    #     dst = np.float32([[0,0],[1024,0],[0,1024],[1024,1024]])
    #     M = cv2.getPerspectiveTransform(points, dst)
    #     w = cv2.warpPerspective(img, M, (1024,1024))
    #
    #     if save_img:
    #         cv2.imwrite(f"Orgnl_Marker_{z_offset}_{plotter.camera.up[0]}.png",img)
    #         cv2.imwrite(f"Wrped_Marker_{z_offset}_{plotter.camera.up[0]}.png",w)
    #
    #     return w, img

def get_computed_display_value(world_point, camera, viewport, real_camera_position):
    """
    world_point: [x, y, z] 월드 좌표의 3D 점
    camera: dict 형식으로, 카메라의 정보를 포함
            { 'position': [x, y, z],
              'focal_point': [x, y, z],
              'view_up': [x, y, z],
              'view_angle': 수직 시야각 (deg),
              'near': near 평면,
              'far': far 평면 }
    viewport: (x, y, width, height) 형태의 뷰포트 설정
              보통 (0, 0, window_width, window_height)

    반환: (x_disp, y_disp) 디스플레이 좌표 (픽셀 단위)
    """
    global projection
    # 1. 모델뷰 행렬 계산 (카메라의 좌표계로 변환)
    eye = np.array(camera['position'], dtype=np.float32)
    eye_assume = real_camera_position

    target = np.array(camera['focal_point'], dtype=np.float32)
    target_assume = target + (real_camera_position - eye)

    up = np.array(camera['view_up'], dtype=np.float32)

    modelview = look_at(eye_assume, target_assume, up)

    # 2. 투영 행렬 계산 (원근 투영)
    fov = camera['view_angle']  # 수직 시야각 (deg)
    near = camera.get('near', 0.1)
    far = camera.get('far', 1000.0)
    # viewport의 width, height로부터 종횡비 계산
    _, _, vp_width, vp_height = viewport
    aspect = vp_width / vp_height
    projection = perspective_projection_matrix(fov, aspect, near, far)

    # 3. 세계 좌표를 동차 좌표(homogeneous coordinate)로 변환
    wp = np.array([world_point[0], world_point[1], world_point[2], 1.0], dtype=np.float32)

    # 4. 모델뷰와 투영 행렬을 적용하여 클립 좌표 계산
    clip_coord = projection.dot(modelview.dot(wp))

    # 5. perspective division (동차 좌표를 3D로 환원)
    ndc = clip_coord[:3] / clip_coord[3]  # ndc: Normalized Device Coordinates, 각 값이 [-1, 1]

    # 6. 뷰포트 변환: ndc를 실제 디스플레이 좌표로 매핑 (보통 x: 왼쪽~오른쪽, y: 아래~위)
    #    viewport: (x, y, width, height), 여기서 (x,y)는 뷰포트의 왼쪽 아래 코너라고 가정
    x_disp = viewport[0] + (ndc[0] + 1) * (viewport[2] / 2.0)
    y_disp = viewport[1] + (ndc[1] + 1) * (viewport[3] / 2.0)

    ####
    # cache = main.cache
    # rvec = cache.get_cache("rvec")
    # tvec = cache.get_cache("tvec")
    # camera_matrix = cache.get_cache("camera_matrix")
    # dist_coeffs = cache.get_cache("dist_coeff")
    #
    # img_point = cv2.projectPoints(world_point, rvec, tvec, camera_matrix, dist_coeffs)
    # img_point = img_point.reshape(-1, 2)[0]
    #
    # return img_point
    ####

    return (x_disp, y_disp)

def look_at(eye, target, up):
    """
    eye: 카메라 위치
    target: 카메라가 바라보는 점 (focal point)
    up: 카메라의 업 벡터
    """
    # 카메라 좌표계 (오른손 좌표계)에서,
    # zaxis는 카메라의 "뒤쪽" 방향 (eye에서 target을 빼면, 카메라가 바라보는 방향의 반대)
    zaxis = normalize(eye - target)
    xaxis = normalize(np.cross(up, zaxis))
    yaxis = np.cross(zaxis, xaxis)

    # 카메라의 회전과 이동을 포함하는 4x4 행렬 (카메라 좌표로 변환)
    # (OpenGL 스타일 LookAt 행렬)
    mat = np.array([[xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
                    [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
                    [zaxis[0], zaxis[1], zaxis[2], -np.dot(zaxis, eye)],
                    [0, 0, 0, 1]])
    return mat


def perspective_projection_matrix(fov_deg, aspect, near, far):
    """
    fov_deg: 수직 시야각 (degree)
    aspect: 종횡비 (width / height)
    near, far: near와 far 클리핑 평면
    """
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1
    return m

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm