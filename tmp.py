from find_cross_point_model.predict import predict_one_image
import cv2
from extension import image_show
from n_warp import warp_perspective
import numpy as np

image = cv2.imread('./data/image_seg/1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (1360,1020))
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

output, points = predict_one_image(image)
output = output.astype(np.uint8)
image_show(image)
image_show(output)
