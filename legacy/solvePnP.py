import numpy as np
import cv2
from n_camera import Camera


def solvePnP(camera:Camera, object_point, image_point):
    camera_matrix = camera.camera_matrix
    dist_coeffs = camera.dist_coeffs

    retval, rvec, tvec = cv2.solvePnP(object_point, image_point, camera_matrix, dist_coeffs)
    if retval:
        return rvec, tvec
    else:
        raise "solvePnP function can't return proper value"
