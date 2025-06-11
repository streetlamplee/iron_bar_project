import numpy as np
import cv2
import torch
from torchvision import transforms
import os
import extension
from image_2_ironbar_model import data_making
from model import pointFindingModel
from PIL import Image

def predict(image_list, model_file = None):
    input = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for image in image_list:
        image_tensor = torch.tensor(image, dtype = torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor /= 255.
        norm = transforms.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        image_tensor = norm(image_tensor)
        input.append(image_tensor)

    input = torch.concat(input, dim = 0)
    input = input.unsqueeze(0)
    input_height = input.shape[2]
    input_width = input.shape[3]

    model = pointFindingModel()

    if model_file is None:
        model_folder = '/home/user/PycharmProjects/iron_bar_sample_project/image_2_ironbar_model/models'
        model_filename = os.path.join(model_folder, extension.get_latest_pth_file(model_folder, '.pth'))
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    input = input.to(device)

    with torch.no_grad():
        output_logit = model(input)
        output = torch.sigmoid(output_logit)


    output = output.squeeze(0) # 4, 8, 8

    result = np.zeros((input_height, input_width), dtype = np.uint8)

    point_list = []

    for h in range(output.shape[1]):
        for w in range(output.shape[2]):
            point_obj = output[0, h, w]
            x_value = output[1, h, w]
            y_value = output[2, h, w]

            if point_obj >= 0.5:
                cv2.circle(result,(int(32 * h + 32 * y_value), int(32 * w + 32 * x_value)), radius=5, thickness = -1, color = (255,255,255))
                point_list.append((float(point_obj), int(32 * h + 32 * y_value), int(32 * w + 32 * x_value)))
    return output, result, point_list

if __name__ == '__main__':
    image_list = []
    image_filename_list = []
    for i in range(5):
        image = cv2.imread(f'../warp_image/{i+1}.png')
        image_filename_list.append(f'../warp_image/{i+1}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_list.append(image)
    output, result, point_list = predict(image_list)

    comp = np.zeros_like(image, dtype = np.float32)
    for i in range(len(image_list)):
        comp += image_list[i] / len(image_list)

    _, res, _ = data_making.make_target(image_filename_list, 1024)

    cv2.imwrite('test.png', res)
    cv2.imwrite('result.png', result)

    for h in range(output.shape[1]):
        for w in range(output.shape[2]):
            point_obj = output[0, h, w]
            x_value = output[1, h, w]
            y_value = output[2, h, w]

            if point_obj >= 0.5:
                cv2.circle(res,(int(32 * h + 32 * y_value), int(32 * w + 32 * x_value)), radius=5, thickness = -1, color = (255,255,255))
    cv2.imwrite('test_point.png', res)