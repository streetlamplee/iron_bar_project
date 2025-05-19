import numpy as np
import cv2
import os
import json

class clickHandler:
    def __init__(self):
        self.clicked_points = []
    def handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append([x,y])

def make_data(path:str, output_folder:str, num:int):
    if not os.path.exists(path):
        raise RuntimeError(f'No Directory Found : {path}')

    if not os.path.exists(f'{output_folder}'):
        os.makedirs(f'{output_folder}', exist_ok=True)
        os.mkdir(f'{output_folder}/image')

    image_filename_list = os.listdir(path)

    crop_height = 256
    crop_width = 256
    json_s = {}
    json_s['len'] = 0
    cnt = 0

    while True:
        random_idx = np.random.randint(0, len(image_filename_list))
        image_filename = image_filename_list[random_idx]
        if not image_filename.endswith(('.png','.jpg')):
            continue

        json_value = {}

        image = cv2.imread(os.path.join(path, image_filename))

        h, w, _ = image.shape

        crop_h = np.random.randint(0, h - crop_height)
        crop_w = np.random.randint(0, w - crop_width)

        image_crop = image[crop_h:crop_h + crop_height, crop_w:crop_w + crop_width]

        click = clickHandler()
        image_crop_copy = image_crop.copy()
        cv2.imshow("Please click the Crossed point", image_crop_copy)
        cv2.setMouseCallback("Please click the Crossed point", click.handler)
        while True:
            key = cv2.waitKey(1)
            if len(click.clicked_points) != 0:
                cv2.circle(image_crop_copy, click.clicked_points[-1], radius=3, thickness=-1, color = (0,0,255))
            cv2.imshow("Please click the Crossed point", image_crop_copy)

            if key == 27:
                break

        json_value['filename'] = os.path.join(f'{output_folder}/image', f"{cnt}.png")

        cv2.imwrite(os.path.join(f'{output_folder}/image', f"{cnt}.png"), image_crop)

        json_value['mask'] = click.clicked_points
        json_s['len'] += len(click.clicked_points)

        json_s[f'{cnt}'] = json_value
        cnt += 1

        if len(json_s) >= num+1:
            break

    with open(os.path.join(output_folder, 'data.json'), 'w') as json_file:
        json.dump(json_s, json_file, indent=1)

if __name__ == "__main__":
    make_data('../warp_image', 'valid', 2)