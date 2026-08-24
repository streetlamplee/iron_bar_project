"""
교차점 검출 모델용 Dataset.

정답이 "점 좌표 목록"이라 그대로는 학습에 쓸 수 없어서,
이미지를 격자로 나눈 텐서 형태로 변환해 넣어준다.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import json

def calculate_tensor(image_size, grid_size, keypoints, max_points_per_cell=6):
    """
    점 좌표 목록을 모델이 학습할 수 있는 격자 형태의 정답 텐서로 바꾼다.

    :param image_size: 원본 이미지 한 변의 길이
    :param grid_size: 격자 칸 수 (한 변 기준)
    :param keypoints: 정답 점 목록 [(x, y), ...]
    :param max_points_per_cell: 한 칸이 담을 수 있는 점의 최대 개수
    :return: (grid_size, grid_size, max_points_per_cell * 3) 텐서
             채널은 점마다 (상대 x, 상대 y, 점 있음 여부) 3개씩이다.
    """
    stride = image_size // grid_size
    grid_size = int(grid_size)
    stride = int(stride)
    target = torch.zeros((grid_size, grid_size, max_points_per_cell * 3), dtype=torch.float32)

    for x, y in keypoints:
        # 이 점이 어느 칸에 속하는지 계산
        # 셀 위치 계산
        cell_x = int(x // stride)
        cell_y = int(y // stride)

        # 칸 안에서의 위치를 0~1 비율로 저장한다 (칸 크기와 무관하게 학습되도록).
        # 셀 내 상대 좌표
        rel_x = (x % stride) / stride
        rel_y = (y % stride) / stride

        # 한 칸에 점이 여러 개 올 수 있으므로 빈 슬롯을 찾아 채운다.
        # 슬롯이 다 찼으면 그 점은 버려진다.
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
    """
    JSON에 기록된 (이미지 경로, 점 좌표 목록)을 읽어 학습용 텐서로 만들어 주는 Dataset.
    """
    def __init__(self, data_json:dict, transform = None, input_size=256):
        self.data_json = data_json
        self.transform = transform
        self.input_size = input_size

    def __len__(self):
        # JSON에 데이터 개수를 담은 "len" 항목이 함께 들어 있어 1을 뺀다.
        return len(self.data_json) - 1

    def __getitem__(self, idx):
        # 이미지와 마스크 파일 경로
        original_idx = idx % (len(self.data_json) - 1)
        data = self.data_json[f'{original_idx}']
        image_path = data['filename']

        # 이미지와 마스크 로드
        image = Image.open(image_path).convert("RGB")
        points = data['mask']

        # 이미지를 변형하면 점 좌표도 같이 움직여야 하므로 둘을 함께 넘긴다.
        if self.transform:
            image, points = self.transform(image, points)
            Normalization = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            image = Normalization(image)

        image = image.float()

        # backbone을 지나며 해상도가 1/32로 줄어든다. 그 크기가 곧 격자 칸 수가 된다.
        size = int(self.input_size / 32)
        mask = calculate_tensor(512, size, points, 1)
        # (H, W, C) -> (C, H, W) : 모델 출력과 축 순서를 맞춘다.
        mask = mask.permute(2, 0, 1)
        # test = extract_keypoints_from_tensor(mask.unsqueeze(0).permute(0,3,1,2), 256)
        return image, mask