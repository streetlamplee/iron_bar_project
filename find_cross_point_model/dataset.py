import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import json

def caculate_tensor(image_size, grid_size, keypoint):
    """
        image_size: int, 예: 256
        grid_size: int, 예: 16
        keypoint: tuple, 예: (135.2, 72.5) - 정답 점 좌표 (픽셀 기준)

        반환값: target tensor, shape = (16, 16, 3)
        """
    stride = image_size // grid_size
    x, y = keypoint

    # 셀 위치 계산
    cell_x = int(x // stride)
    cell_y = int(y // stride)

    # 셀 내 상대 좌표
    rel_x = (x % stride) / stride
    rel_y = (y % stride) / stride

    # 타겟 텐서 초기화
    target = torch.zeros((grid_size, grid_size, 3), dtype=torch.float32)

    # 정답 셀에 값 채우기
    target[cell_y, cell_x, 0] = rel_x  # x 오프셋
    target[cell_y, cell_x, 1] = rel_y  # y 오프셋
    target[cell_y, cell_x, 2] = 1.0  # objectness

    return target

def calculate_tensor(image_size, grid_size, keypoints, max_points_per_cell=4):
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
            if target[cell_y, cell_x, offset + 2] == 0:  # objectness가 비어 있음
                target[cell_y, cell_x, offset + 0] = rel_x
                target[cell_y, cell_x, offset + 1] = rel_y
                target[cell_y, cell_x, offset + 2] = 1.0
                break  # 이 점은 처리 완료
        # else:  # 선택적으로 경고 출력 가능
        #     print(f"Cell ({cell_y}, {cell_x}) already full, skipping extra keypoint")

    return target

class PointDataset(Dataset):
    def __init__(self, data_json:dict, transform = None, input_size=256):
        self.data_json = data_json
        self.transform = transform
        self.input_size = 256

    def __len__(self):
        return len(self.data_json) - 1

    def __getitem__(self, idx):
        # 이미지와 마스크 파일 경로
        original_idx = idx % (len(self.data_json)-1)
        data = self.data_json[f'{original_idx}']
        image_path = data['filename']

        # 이미지와 마스크 로드
        image = Image.open(image_path).convert("RGB")
        points = data['mask']

        if self.transform :
            image, points = self.transform(image, points)
            Normalization = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            image = Normalization(image)

        image = image.float()

        size = int(self.input_size / 32)
        mask = calculate_tensor(256, size, points)
        mask = mask.permute(2,0,1)
        # test = extract_keypoints_from_tensor(mask.unsqueeze(0).permute(0,3,1,2), 256)
        return image, mask