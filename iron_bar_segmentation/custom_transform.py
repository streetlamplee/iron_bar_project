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
        if self.mode == 'valid':
            image = self.valid_transforms(input_image)
            mask = self.valid_transforms(input_mask)
            return image, mask
        else:
            input_image = np.array(input_image, dtype = np.uint8)
            input_mask = np.array(input_mask, dtype = np.uint8)
            image, mask = self.geometric(input_image, input_mask)

            image = Image.fromarray(image)
            mask = Image.fromarray(mask)

            image = self.train_transforms(image)
            to_tensor = transforms.ToTensor()
            mask = to_tensor(mask)
            return image, mask


    def geometric(self, input_image, input_data):
        image, data = self.random_rotation(input_image, input_data, 30, 0.5)
        image, data = self.random_Hflip(image, data, 0.5)
        image, data = self.random_Vflip(image, data, 0.5)
        return image, data


    def random_rotation(self, image, mask, degree, p):
        if random.random() < p:
            angle = random.uniform(-degree, degree)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)

            rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            return image, mask
        else:
            return image, mask

    def random_Hflip(self, image, mask, p):
        if random.random() < p:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

            return image, mask
        else:
            return image, mask

    def random_Vflip(self, image, mask, p):
        if random.random() < p:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
            return image, mask
        else:
            return image, mask