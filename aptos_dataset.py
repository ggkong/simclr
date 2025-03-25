import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class APTOSSimCLRDataset(Dataset):
    def __init__(self, image_dir, image_size=224):
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        self.transform = A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            A.RandomBrightnessContrast(p=0.8),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        xi = self.transform(image=image)['image']
        xj = self.transform(image=image)['image']
        return xi, xj
