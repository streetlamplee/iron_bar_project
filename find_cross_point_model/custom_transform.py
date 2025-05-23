import random
import numpy as np
import torch
import cv2
import PIL
from PIL import Image

from torchvision import transforms

class custom_transforms():
    def __init__(self, mode):
        self.train_transforms = transforms.Compose([
            transforms.ColorJitter(brightness= .5,
                                   contrast=.2,
                                   saturation=.1,
                                   hue=(-.3, .3)),
            transforms.ToTensor(),
        ])
        self.valid_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.mode = mode

    def __call__(self, input_image, input_data):
        if self.mode == 'valid':
            image = self.valid_transforms(input_image)
            return image, input_data
        else:
            image, data = self.geometric(input_image, input_data)
            image = self.train_transforms(image)
            return image, data


    def geometric(self, input_image, input_data):
        image, data = self.random_rotation(input_image, input_data, 30, 0.5)
        return image, data


    def random_rotation(self, image, points, degree, p):
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

            rotated_points = []
            for (x, y) in points:
                point_vec = np.array([x, y, 1.0])
                rotate_x, rotate_y = rot_mat @ point_vec

                if 0<= rotate_x < w and 0 <= rotate_y < h:
                    rotated_points.append([rotate_x, rotate_y])
            rotated_image = Image.fromarray(rotated_image)
            return rotated_image, rotated_points
        else:

            return image, points
