import json
import os
import cv2

import extension
from find_cross_point_model.dataset import calculate_tensor
from find_cross_point_model.predict import extract_keypoints_from_tensor

with open('find_cross_point_model/train/data.json', 'r') as f:
    json_file = json.load(f)

for key in json_file.keys():
    if key == 'len':
        continue

    image_path = json_file[key]['filename']
    mask = json_file[key]['mask']
    image = cv2.imread(os.path.join('find_cross_point_model', image_path))
    c_tensor = calculate_tensor(image.shape[0], int(image.shape[0] / 32), mask)
    c_tensor = c_tensor.unsqueeze(0).permute(0,3,1,2)
    mask_after_f = extract_keypoints_from_tensor(c_tensor, image.shape[0])
    for m in mask_after_f:
        h, w, o = m
        if o >= 0.5:
            cv2.circle(image, (int(h),int(w)), radius = 2, thickness = -1, color = (0,0,255))
    extension.image_show(image)