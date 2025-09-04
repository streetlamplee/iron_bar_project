import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import json

def line_data_processing(h_lines, v_lines, image_size = (256, 256), grid_size = (8,8)):
    res = torch.zeros(size=(4, *grid_size), dtype = torch.float32)
    grid_height = image_size[0] / grid_size[0]
    grid_width = image_size[1] / grid_size[1]
    for h_line in h_lines:
        grid_num_line = int(h_line // grid_height)
        value_line = (h_line % grid_height) / grid_height

        res[0, grid_num_line, :] = 1.0
        res[1, grid_num_line, :] = value_line

    for v_line in v_lines:
        grid_num_line = int(v_line // grid_width)
        value_line = (v_line % grid_width) / grid_width

        res[2, :, grid_num_line] = 1.0
        res[3, :, grid_num_line] = value_line

    return res

def point_data_processing(image_size, grid_size, keypoints, max_points_per_cell=1):
    stride = image_size // grid_size
    grid_size = int(grid_size)
    stride = int(stride)
    target = torch.zeros((grid_size, grid_size, max_points_per_cell * 3), dtype=torch.float32)

    for x, y in keypoints:
        # 셀 위치 계산
        cell_x = int(x // stride)
        cell_y = int(y // stride)

        # 셀 내 상대 좌표
        rel_x = (x % stride) / stride
        rel_y = (y % stride) / stride

        # 이 셀에 이미 채워진 슬롯 개수 확인
        for i in range(max_points_per_cell):
            offset = i * 3
            if target[cell_y, cell_x, offset + 0] == 0:  # objectness가 비어 있음
                target[cell_y, cell_x, offset + 0] = 1.0
                target[cell_y, cell_x, offset + 1] = rel_x
                target[cell_y, cell_x, offset + 2] = rel_y
                break  # 이 점은 처리 완료
        # else:  # 선택적으로 경고 출력 가능
        #     print(f"Cell ({cell_y}, {cell_x}) already full, skipping extra keypoint")

    return target



class PointDataset(Dataset):
    def __init__(self, data_json:dict, transform = None, input_size=256):
        self.data_json = data_json
        self.transform = transform
        self.input_size = 256
        self.basic_trans = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                               ])

    def __len__(self):
        return len(self.data_json) - 1

    def __getitem__(self, idx):
        image_list = []
        for image in self.data_json[f'{idx}']['images']:
            image = Image.open(image).convert('RGB')
            if self.transform:
                image = self.transform(image)

            image = self.basic_trans(image)

            image_list.append(image)

        input = torch.stack(image_list, dim = 0)

        # target = line_data_processing(self.data_json[f'{idx}']['target_h'],
        #                          self.data_json[f'{idx}']['target_v'],
        #                          image_size=(256,256),
        #                          grid_size=(8, 8))

        target = point_data_processing(256,
                                       8,
                                       self.data_json[f'{idx}']['points']
                                       )
        target = target.permute(2,0,1)
        return input, target