import os
import json
import numpy as np
import cv2
import extension
import torch
from model import pointFindingModel
from torchvision import transforms
from nms import nms

def predict(input_image:np.ndarray, m = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if m is None:
        model_folder = './models'
        model_filename = os.path.join(model_folder, extension.get_latest_pth_file(model_folder, '.pth'))
        checkpoint = torch.load(model_filename)
        model = pointFindingModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
    else:
        model = pointFindingModel()
        checkpoint = torch.load(m)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
    t_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    t_image = torch.tensor(t_image, dtype = torch.float32)
    t_image /= 255
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    t_image = t_image.permute(2,0,1).unsqueeze(0)
    t_image = normalize(t_image)
    t_image = t_image.to(device)
    with torch.no_grad():
        output = model(t_image)
        print(output.shape)
        output = torch.sigmoid(output)
        keypoints = nms(output, 7)
    res = input_image.copy()
    for keypoint in keypoints:
        h, w, o = keypoint
        h = int(h)
        w = int(w)
        if o >= .5:
            cv2.circle(res, (h,w), 2, (0,0,255), -1)
    os.makedirs('result', exist_ok=True)
    # cv2.imwrite(os.path.join('result', test_image), res)
    # extension.image_show(res)
    return res, input_image

class clickHandler:
    def __init__(self):
        self.clicked_points = []
    def handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append([x,y])

def clicked_point_finder(image:np.ndarray):
    click = clickHandler()
    image_copy = image.copy()
    cv2.imshow("Please click the Crossed point", image_copy)
    cv2.setMouseCallback("Please click the Crossed point", click.handler)
    while True:
        key = cv2.waitKey(1)
        if len(click.clicked_points) != 0:
            cv2.circle(image_copy, click.clicked_points[-1], radius=3, thickness=-1, color=(0, 0, 255))
        cv2.imshow("Please click the Crossed point", image_copy)

        if key == 13:
            break
    cv2.destroyWindow('Please click the Crossed point')
    return click.clicked_points

def make_more_data(num:int, data_path:str, origin_data_json_path:str):
    extension.set_seed(0)
    data_list = os.listdir(data_path)
    with open(origin_data_json_path, 'r') as f:
        data_json = json.load(f)
    before_data_len = len(data_json)
    if 'len' in data_json.keys():
        cnt = before_data_len - 1
    else:
        cnt = before_data_len
    while True:
        data_file_name_iter = os.path.join(data_path, data_list[np.random.randint(0, len(data_list))])
        image_iter = cv2.imread(data_file_name_iter)
        random_x = np.random.randint(0, image_iter.shape[0] - 256 + 1)
        random_y = np.random.randint(0, image_iter.shape[1] - 256 + 1)
        image_iter = image_iter[random_x: random_x + 256, random_y: random_y + 256]
        res_iter, image = predict(image_iter)
        extension.image_show(res_iter)
        if int(input()):
            clicked_point = clicked_point_finder(image)
            sub_dict = {
                'filename': f'train/image/{cnt}.png',
                'mask': clicked_point
            }
            if 'len' in data_json.keys():
                data_json['len'] += len(clicked_point)
            data_json[f'{cnt}'] = sub_dict


            cv2.imwrite(origin_data_json_path.replace('data.json', f'image/{cnt}.png'), image)
            with open(origin_data_json_path, 'w') as f:
                json.dump(data_json, f, indent=2)
            cnt += 1


        if len(data_json) >= before_data_len + num:
            break

    return

if __name__ == "__main__":
    make_more_data(5, '../warp_image', 'data/data.json')
    '''
    374개 까지 진행할 것
    '''