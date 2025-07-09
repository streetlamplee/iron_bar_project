from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image


class ironbar_dataset(Dataset):
    def __init__(self, image_path, mask_path, transform = None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.transform = transform
        self.norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = Image.open(sorted(self.image_path)[idx]).convert('RGB')
        mask = Image.open(sorted(self.mask_path)[idx]).convert('L')

        if self.transform:
            image, mask = self.transform(image, mask)
        image = self.norm(image)

        image = image.float()
        mask = (mask > 0).long()

        return image, mask