"""
학습용 데이터 augmentation.

철근 사진은 수가 적어서, 회전/좌우·상하 반전/색감 변화를 무작위로 주어
같은 사진을 여러 상황처럼 보이게 만들어 학습시킨다.

핵심 제약: 위치를 바꾸는 변형(회전, 반전)은 사진과 마스크에 반드시 똑같이 적용해야 한다.
그래서 torchvision 변환을 그대로 쓰지 않고 직접 구현했다.
"""

import random
import numpy as np
import torch
import cv2
import PIL
from PIL import Image

from torchvision import transforms

class custom_transforms():
    def __init__(self, mode):
        """
        :param mode: 'train'이면 augmentation 적용, 'valid'면 텐서 변환만 수행한다.
                     (검증 결과는 매번 같아야 비교가 가능하므로 무작위 변형을 넣지 않는다.)
        """
        # 색감 변화는 사진에만 적용한다. 마스크의 값이 바뀌면 안 되기 때문이다.
        self.train_transforms = transforms.Compose([
            transforms.ColorJitter(brightness= .2,
                                   contrast=.2,
                                   saturation=.1,
                                   hue=.0),
            transforms.ToTensor(),
        ])
        self.valid_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.mode = mode

    def __call__(self, input_image, input_mask):
        """사진과 마스크를 함께 받아 함께 변형한 뒤 텐서로 돌려준다."""
        if self.mode == 'valid':
            image = self.valid_transforms(input_image)
            mask = self.valid_transforms(input_mask)
            return image, mask
        else:
            # 위치 변형은 OpenCV로 처리하므로 PIL -> numpy 로 바꾼다.
            input_image = np.array(input_image, dtype = np.uint8)
            input_mask = np.array(input_mask, dtype = np.uint8)
            image, mask = self.geometric(input_image, input_mask)

            # 색감 변형(ColorJitter)은 PIL 이미지를 요구하므로 되돌린다.
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)

            image = self.train_transforms(image)
            to_tensor = transforms.ToTensor()
            mask = to_tensor(mask)
            return image, mask


    def geometric(self, input_image, input_data):
        """위치를 바꾸는 변형들을 순서대로 적용한다 (사진과 마스크에 동일하게)."""
        image, data = self.random_rotation(input_image, input_data, 30, 0.5)
        image, data = self.random_Hflip(image, data, 0.5)
        image, data = self.random_Vflip(image, data, 0.5)
        return image, data


    def random_rotation(self, image, mask, degree, p):
        """확률 p로 -degree ~ +degree 사이만큼 회전시킨다. 촬영 각도가 매번 다른 상황을 흉내낸다."""
        if random.random() < p:
            angle = random.uniform(-degree, degree)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)

            # BORDER_REFLECT_101: 회전 후 빈 모서리를 검게 두지 않고 주변 무늬로 채운다.
            rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            return image, mask
        else:
            return image, mask

    def random_Hflip(self, image, mask, p):
        """확률 p로 좌우 반전"""
        if random.random() < p:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

            return image, mask
        else:
            return image, mask

    def random_Vflip(self, image, mask, p):
        """확률 p로 상하 반전"""
        if random.random() < p:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
            return image, mask
        else:
            return image, mask