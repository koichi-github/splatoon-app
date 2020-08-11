import glob
import random
import os
import sys
import csv
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler



def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class MyDataset(Dataset):
    def __init__(self, cfg_file_path, transform=None, img_size=256, imgpath=False):
        with open(cfg_file_path, "r") as f:
            reader = csv.reader(f)
            data = np.array([row for row in reader])
        
        self.images_path = data[:, 0]
        self.labels = np.array(list(map(int, data[:, 1])))

        self.img_size = img_size
        self.n_data = len(self.images_path)
        self.imgpath = imgpath

        self.transform = transform if transform else transforms.ToTensor()


    def __getitem__(self, index):

        img_path = self.images_path[index % len(self.images_path)].rstrip()
        label = self.labels[index % len(self.labels)]

        # Extract image as PyTorch tensor
        img = self.transform(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        if self.imgpath:
            return img_path, img, label
        else:
            return img, label


    def __len__(self):
        return self.n_data
