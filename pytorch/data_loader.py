from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np

from dataset import CancerDataset


def loader (image_path, label_path, batch_size, validation_split, shuffle_dataset=True, random_seed=42):
    dataset = CancerDataset(label_path, image_path)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    return train_loader, validation_loader