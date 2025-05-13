import os.path

import torch
from torch.utils.checkpoint import checkpoint

from  model_architecture import get_model
import cv2
from torchvision.transforms import v2


def predict(model=None,test_data="data/image/1.jpg", recent=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model == None and recent:
        model_folder = os.listdir('models')
        model_folder.sort()
        model_folder = model_folder[-1]
        model_filename = os.listdir(f'models/{model_folder}')
        model_filename.sort()
        model_filename = model_filename[-1]
        checkpoint = torch.load(f"models/{model_folder}/{model_filename}", map_location=torch.device(device))
        print(f'[predict.py] ready to predict with model "models/{model_folder}/{model_filename}"')
    elif model != None:
        checkpoint = torch.load(model, map_locatioin=torch.device(device))
    else:
        raise "Choose model to predict OR set recent arg True"
    model = get_model(2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    test_input = cv2.imread(test_data)
    test_input = cv2.cvtColor(test_input, cv2.COLOR_BGR2RGB)
    test_input = torch.tensor(test_input, dtype=torch.float32)
    test_input /= 255
    normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    test_input =  normalize(test_input.permute(2,0,1))
    test_input = test_input.unsqueeze(0)
    test_input = test_input.to(device)
    with torch.no_grad():
        output = model(test_input)
        print(output.shape)
        output = torch.softmax(output, 1)
        output = torch.argmax(output, dim=1)
    res = output.squeeze(0).cpu().detach().numpy()*255
    if not os.path.exists('model_outputs'):
        os.mkdir('model_outputs')
    cnt = len(os.listdir('model_outputs'))
    cv2.imwrite(f'model_outputs/{cnt}.png', res)
    # cv2.imshow('',res)
    # cv2.waitKey(0)
    # cv2.destroyWindow('')

if __name__ == '__main__':
    predict()