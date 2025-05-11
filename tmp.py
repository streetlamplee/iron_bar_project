# import json
#
# dict = {
#     1 : {
#         "image_path" : "data/image/1.jpg",
#         "cam_position" : [620, 3000, 2000]
#     },
#     2 : {
#         "image_path" : "data/image/2.jpg",
#         "cam_position" : [-1380, 3000, 2000]
#     },
#     3 : {
#         "image_path" : "data/image/3.jpg",
#         "cam_position" : [-1380, 2500, 2000]
#     },
#     4 : {
#         "image_path" : "data/image/4.jpg",
#         "cam_position" : [-2880, 2500, 2000]
#     },
#     5 : {
#         "image_path" : "data/image/5.jpg",
#         "cam_position" : [-2880, 1000, 2000]
#     }
#
# }
#
# with open("data/data_seg.json", "w") as f:
#     json.dump(dict, f, indent=4)

import cv2
import numpy as np
res = cv2.imread("res.png")
res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

res = np.where((res > 255 // 8 * 5), 255, 0)
res = res.astype(np.uint8)

cv2.imshow("thres", res)
cv2.waitKey(0)
cv2.destroyWindow("thres")
cv2.imwrite("res_thres.png", res)
