import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
from torch.utils.data import Dataset
import cv2
import numpy as np
import torchvision.transforms as transforms
import torchbearer
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchbearer import Trial
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1a = nn.Conv2d(3, 16, (3, 3), padding=1)
        self.conv1b = nn.Conv2d(16, 16, (3, 3), padding=1)

        self.conv2a = nn.Conv2d(16, 32, (4, 4), padding=1, stride=2)
        self.conv2b = nn.Conv2d(32, 32, (3, 3), padding=1)

        self.conv3a = nn.Conv2d(32, 64, (4, 4), padding=1, stride=2)
        self.conv3b = nn.Conv2d(64, 64, (3, 3), padding=1)

        self.fc1 = self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #         print(x.shape)
        out = self.conv1a(x)
        #         print(out.shape)
        out = F.relu(out)
        out = self.conv1b(out)
        out = F.relu(out)

        out = self.conv2a(out)
        out = F.relu(out)
        out = self.conv2b(out)
        out = F.relu(out)

        out = self.conv3a(out)
        out = F.relu(out)
        out = self.conv3b(out)
        out = F.relu(out)

        #         print(out.shape)
        out = out.view(out.shape[0], -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        #         print(out)

        return out

data_path = '../data'
labels_path = '{}/train_labels/train_labels.csv'.format(data_path)

train_labels = pd.read_csv(labels_path)

print(len(train_labels))


class TrainDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = '{}/{}.tif'.format(self.root_dir,
                                      self.data_frame.iloc[idx, 0])
        img = cv2.imread(img_name)
        img = img[24:-24, 24:-24, :]
        img = np.moveaxis(img, -1, 0)

        return torch.Tensor(img), self.data_frame.iloc[idx, 1].astype(float)


dataset = TrainDataset(labels_path, '{}/train'.format(data_path))
batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed= 42

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

model = TestModel()

model.train()

loss_function = nn.BCELoss()
optimiser = optim.Adam(model.parameters())

device = "cuda:0" if torch.cuda.is_available() else "cpu"
trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, test_generator=validation_loader)
trial.run(epochs=10)
results = trial.evaluate(data_key=torchbearer.TEST_DATA)
print(results)