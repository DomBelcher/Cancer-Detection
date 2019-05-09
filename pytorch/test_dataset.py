import os

import pandas as pd
import cv2
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, root_dir, label, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir('{}/{}'.format(root_dir, label))
        self.label = label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = '{}/{}/{}'.format(self.root_dir,
                                         self.label,
                                         self.files[idx])

        # print(img_name)
        img = cv2.imread(img_name)
        # img = img[24:-24, 24:-24, :]
        # img = np.moveaxis(img, -1, 0)

        if self.transform:
            img = self.transform(img)

        return {'image': img, 'filename': self.files[idx]}, self.label #.astype(float)

