import numpy as np
from PIL import Image
from random import shuffle

import torch
from torch.utils import data


class LowLightDataset(data.Dataset):

    def __init__(self, image_files=None, image_size=256):
        self.image_files = image_files
        self.image_size = image_size
        shuffle(self.image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_path = self.image_files[item]
        image = Image.open(image_path)
        image = image.resize(
            (self.image_size, self.image_size), Image.ANTIALIAS)
        image = (np.asarray(image) / 255.0)
        image = torch.from_numpy(image).float()
        image_data = image.permute(2, 0, 1)
        return image_data
