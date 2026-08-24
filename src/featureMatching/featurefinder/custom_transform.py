"""
교차점 검출 모델용 augmentation.

이미지를 회전하거나 뒤집으면 교차점 좌표도 같은 규칙으로 옮겨야 한다.
정답이 이미지가 아니라 좌표 목록이라서, 변형마다 좌표 변환을 직접 계산한다.
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
        :param mode: 'train'이면 회전/반전 augmentation 적용, 'valid'면 텐서 변환만 수행
        """
        self.train_transforms = transforms.Compose([
            # transforms.ColorJitter(brightness= .1,
            #                        contrast=.1,
            #                        saturation=0,
            #                        hue=0),
            transforms.ToTensor(),
        ])
        self.valid_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.mode = mode

    def __call__(self, input_image, input_data):
        """
        :param input_image: 입력 이미지
        :param input_data: 교차점 좌표 목록. 이미지와 함께 변형된다.
        """
        if self.mode == 'valid':
            image = self.valid_transforms(input_image)
            return image, input_data
        else:
            image, data = self.geometric(input_image, input_data)
            image = self.train_transforms(image)
            return image, data


    def geometric(self, image, data):
        """이미지를 움직이는 변형들. 좌표도 같은 규칙으로 함께 옮긴다."""
        image, data = self.random_rotation(image, data, 30, 0.5)
        image, data = self.random_Hflip(image, data, 0.5)
        image, data = self.random_Vflip(image, data, 0.5)
        return image, data


    def random_rotation(self, image, points, degree, p):
        """
        확률 p로 이미지를 회전시키고, 점 좌표에도 같은 회전 행렬을 곱해 위치를 맞춘다.
        회전으로 이미지 밖으로 밀려난 점은 목록에서 제외한다.
        """
        r = np.random.uniform(0, 1)
        if r <= p:
            if isinstance(image, Image.Image):
                image = np.array(image.convert("RGB"))
            assert image.shape[0] == image.shape[1], ''
            h, w = image.shape[:2]
            center = (h//2, w//2)

            angle = random.uniform(-degree, degree)

            rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)

            rotated_image = cv2.warpAffine(
                image,
                rot_mat,
                (w, h),
                flags = cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0,0,0)
            )

            # 이미지에 쓴 것과 같은 회전 행렬을 점에도 적용한다.
            rotated_points = []
            for (x, y) in points:
                point_vec = np.array([x, y, 1.0])
                rotate_x, rotate_y = rot_mat @ point_vec

                # 회전 후 화면 밖으로 나간 점은 버린다.
                if 0<= rotate_x < w and 0 <= rotate_y < h:
                    rotated_points.append([rotate_x, rotate_y])
            rotated_image = Image.fromarray(rotated_image)
            return rotated_image, rotated_points
        else:

            return image, points

    def random_Hflip(self, image, points, p):
        """확률 p로 좌우 반전. 점 좌표도 함께 뒤집는다."""
        r = np.random.uniform(0, 1)
        if r < p:
            if isinstance(image, Image.Image):
                image = np.array(image.convert("RGB"))
            h, w = image.shape[:2]
            flipped_image = cv2.flip(image, 1)

            flipped_points = []
            for x, y in points:
                new_y = w - 1 - y
                if 0 <= x < h and 0 <= new_y < w:
                    flipped_points.append([x, new_y])
            flipped_image = Image.fromarray(flipped_image)
            return flipped_image, flipped_points

        else:
            return image, points

    def random_Vflip(self, image, points, p):
        """확률 p로 상하 반전. 점 좌표도 함께 뒤집는다."""
        r = np.random.uniform(0, 1)
        if r < p:
            if isinstance(image, Image.Image):
                image = np.array(image.convert("RGB"))
            h, w = image.shape[:2]
            flipped_image = cv2.flip(image, 0)
            flipped_points = []
            for x, y in points:
                new_x = h - 1 - x
                if 0 <= new_x < w and 0 <= y < h:
                    flipped_points.append([new_x, y])

            flipped_image = Image.fromarray(flipped_image)
            return flipped_image, flipped_points
        else:
            return image, points