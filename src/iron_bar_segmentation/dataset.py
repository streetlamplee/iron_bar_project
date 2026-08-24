"""
철근 사진과 정답 마스크를 짝지어 모델에 넣어주는 Dataset.
"""

from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image


class ironbar_dataset(Dataset):
    def __init__(self, image_path, mask_path, transform = None):
        """
        :param image_path: 원본 사진 경로 목록
        :param mask_path: 정답 마스크 경로 목록 (철근=흰색, 배경=검정)
        :param transform: 사진과 마스크에 동일하게 적용할 augmentation
        """
        self.image_path = image_path
        self.mask_path = mask_path
        self.transform = transform
        self.norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        # 사진과 마스크가 어긋나지 않도록 양쪽 모두 이름순으로 정렬한 뒤 같은 순번을 꺼낸다.
        # (매번 정렬하므로 데이터가 많아지면 느려질 수 있다.)
        # 마스크는 흑백 한 장이면 되므로 'L'로 변환한다.
        image = Image.open(sorted(self.image_path)[idx]).convert('RGB')
        mask = Image.open(sorted(self.mask_path)[idx]).convert('L')

        # 회전/뒤집기 같은 변형은 사진과 마스크에 똑같이 적용되어야 한다.
        if self.transform:
            image, mask = self.transform(image, mask)
        # 사진만 ImageNet 통계로 정규화한다 (마스크는 0/1 값이므로 건드리지 않는다).
        image = self.norm(image)

        image = image.float()
        # 마스크를 0/1 이진값으로 확정한다 (jpg 압축 등으로 생긴 중간 밝기 값을 정리).
        mask = (mask > 0).long()

        return image, mask