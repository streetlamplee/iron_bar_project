import PIL.Image
import numpy as np
import cv2

import torch.nn as nn
import torchvision.transforms as transforms

class custom_transforms():
    def __init__(self):
        self.random_int_list = np.random.random(5).tolist()
        pass

    def __call__(self, image, points):
        rand_int1, rand_int2, rand_int3, rand_int4, rand_int5 = self.random_int_list
        height, width = image.size
        if rand_int1 > 0.5:
            # horizontalFlip
            image = transforms.RandomHorizontalFlip(p=1.0)
            points = [[h, width - w] for h, w in points]

        if rand_int2 > 0.5:
            # VertivalFlip
            image = transforms.RandomVerticalFlip(p=1.0)
            points = [[height - h, w] for h, w in points]

        image = transforms.ToTensor()

        return image, points

if __name__ == "__main__":
    transform = custom_transforms()
