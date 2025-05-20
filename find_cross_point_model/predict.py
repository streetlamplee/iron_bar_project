import os
from torchvision import transforms
import numpy as np
import extension
from find_cross_point_model.model import pointFindingModel
import torch
import cv2

def extract_keypoints_from_tensor(tensor, image_size, max_points_per_cell=4, threshold=0.5):
    B, C, H, W = tensor.shape
    assert B == 1, "배치 크기는 1"
    grid_size = H
    stride = image_size // grid_size

    keypoints = []

    # 텐서 shape: (H, W, C)
    tensor = tensor.squeeze(0).permute(1, 2, 0)

    for gy in range(grid_size):
        for gx in range(grid_size):
            cell = tensor[gy, gx]  # shape: (C,)
            for i in range(max_points_per_cell):
                offset = i * 3
                rel_x = cell[offset + 0]
                rel_y = cell[offset + 1]
                objectness = cell[offset + 2]


                abs_x = min((gx + rel_x.item()) * stride, image_size - 1)
                abs_y = min((gy + rel_y.item()) * stride, image_size - 1)
                keypoints.append((abs_x, abs_y, float(objectness)))

    return keypoints


def main():
    test_image_folder = './valid/image'
    test_image_list = os.listdir(test_image_folder)
    model_folder = './models'
    model_folder = os.path.join(model_folder, sorted(os.listdir(model_folder))[-1])
    model_filename = os.path.join(model_folder, sorted(os.listdir(model_folder))[-1])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_filename)
    model = pointFindingModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    test_image_list.sort()
    for test_image in test_image_list:
        t_image = cv2.imread(os.path.join(test_image_folder, test_image))
        t_image = cv2.cvtColor(t_image, cv2.COLOR_BGR2RGB)
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
            keypoints = extract_keypoints_from_tensor(output, 256, 4, 0.5)
        res = cv2.imread(os.path.join(test_image_folder, test_image))
        for keypoint in keypoints:
            h, w, o = keypoint
            h = int(h)
            w = int(w)
            if o >= .5:
                cv2.circle(res, (h,w), 2, (0,0,255), -1)
        os.makedirs('result', exist_ok=True)
        cv2.imwrite(os.path.join('result', test_image), res)
        extension.image_show(res, title=test_image)





if __name__ == "__main__":
    main()