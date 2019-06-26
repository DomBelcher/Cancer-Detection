import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class CancerDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = '{}/{}.tif'.format(self.root_dir,
                                      self.data_frame.iloc[idx, 0])
        img = cv2.imread(img_name)
        # img = img[24:-24, 24:-24, :]
        # img = np.moveaxis(img, -1, 0)

        if self.transform:
            img = self.transform(img)

        return img, self.data_frame.iloc[idx, 1] #.astype(float)