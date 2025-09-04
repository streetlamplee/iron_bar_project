import numpy as np
import torch
import cv2
import torch.nn as nn
from torchvision.models.resnet import resnet101, ResNet101_Weights
import torch.nn.functional as F
from torchvision.transforms import v2
from iron_bar_segmentation.model import DeepLabv3Plus
import etc.extension as extension
import os


def get_model(num_classes):
    model = DeepLabv3Plus(num_classes)
    return model

def predict(input):
    model_file = extension.get_latest_pth_file("src/iron_bar_segmentation/models", '.pth')
    # checkpoint = torch.load('/home/user/PycharmProjects/iron_bar_sample_project/iron_bar_segmentation/models/20250616_174956/epoch00231.pth', map_location=torch.device('cpu'))
    checkpoint = torch.load(os.path.join("src/iron_bar_segmentation/models", model_file), map_location=torch.device('cpu'))
    print(os.path.join('./iron_bar_segmentati1on/models', model_file))
    model = get_model(1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_input = torch.tensor(input, dtype=torch.float32)
    test_input /= 255
    normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    test_input =  normalize(test_input.permute(2,0,1))
    test_input = test_input.unsqueeze(0)
    test_input = test_input.to(device)
    with torch.no_grad():
        output = model(test_input)
        print(output.shape)
        output = torch.sigmoid(output)
    result = (output.squeeze(0).permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
    return result

if __name__ == '__main__':
    input = cv2.imread('/home/user/PycharmProjects/iron_bar_sample_project/data_real/0.jpg')
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    output = predict(input)

    output = cv2.resize(output, (1200, 900))
    input = cv2.resize(cv2.cvtColor(input, cv2.COLOR_RGB2BGR), (1200, 900))
    extension.image_show(input)
    extension.image_show(output)
    output_thres = np.where((output > 127), 255, 0).astype(np.uint8)
    extension.image_show(output_thres)