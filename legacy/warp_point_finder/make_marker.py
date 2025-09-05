import cv2
import numpy as np
from extension import image_show

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def make_marker(id:int = 0):
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_5X5_100"])
    imageSize = 256

    markerImage = cv2.aruco.generateImageMarker(arucoDict, id, imageSize)
    image_show(markerImage)
    cv2.imwrite(f"./marker{id}.png", markerImage)

def make_marker_grid():
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_5X5_100"])
    X = 3.0
    Y = 3.0
    markerLength = 0.05
    markerSeperation = 0.01

    board = cv2.aruco.GridBoard(
        (3, 3), markerLength, markerSeperation, arucoDict
    )

    img = np.zeros((1700,1700), np.uint8)

    board.generateImage((1700,1700), img, 1, 1)
    image_show(img)
    return img

if __name__ == "__main__":
    # make_marker(0)
    # make_marker(1)
    # make_marker(2)
    # make_marker(3)
    cv2.imwrite("./gridboard.jpg",make_marker_grid())