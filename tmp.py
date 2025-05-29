from find_cross_point_model.predict import predict_one_image
import cv2
from extension import image_show
from n_warp import warp_perspective
import numpy as np

image = cv2.imread('./data/image_seg/1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
points = np.array([
        [
            380,
            445
        ],
        [
            1085,
            459
        ],
        [
            1166,
            848
        ],
        [
            238,
            834
        ]
    ], dtype=np.float32)
image = warp_perspective(image, points)

output = predict_one_image(image)
image_show(image)
image_show(output)
