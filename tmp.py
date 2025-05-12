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
#
# import cv2
# import numpy as np
# res = cv2.imread("res.png")
# res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
#
# res = np.where((res > 255 // 8 * 5), 255, 0)
# res = res.astype(np.uint8)
#
# cv2.imshow("thres", res)
# cv2.waitKey(0)
# cv2.destroyWindow("thres")
# cv2.imwrite("res_thres.png", res)
import os.path

import cv2
import numpy as np
img = cv2.imread('data/image/1.jpg')
seg = cv2.imread('data/image_seg/1.png')
if not os.path.exists('train/train_data'):
    os.mkdir('train/train_data')
if not os.path.exists('train/train_mask'):
    os.mkdir('train/train_mask')

idx = 1
while True:
    i = np.random.randint(0, img.shape[0] - 512)
    j = np.random.randint(0, img.shape[1] - 512)

    img_ = img[i:i+512, j:j+512]
    seg_ = seg[i:i+512, j:j+512]
    cv2.imwrite(f'train/train_data/{idx}.png', img_)
    cv2.imwrite(f'train/train_mask/{idx}.png', seg_)
    idx += 1
    if len(os.listdir('train/train_data')) >= 8:
        break
