import torch
from torch import nn
from torch import optim
import torchvision.transforms as transforms
import numpy as np

from models.resnet_model_3c import TestModel
from data_loader_2 import load_test_dataset
from test_dataset import TestDataset
import transforms as tfs

trans = transforms.Compose([
    tfs.PillowToNumpy(),
    # tfs.RandomRotation(range=(0, 360)),
    tfs.Downsize(),
    tfs.Normalise(),
    # tfs.ChannelShift(),
    transforms.ToTensor()
])

test_loader = load_test_dataset(batch_size=128, transforms=trans)

dataset_0 = TestDataset('{}'.format(data_path), label=0, transform=transform)
dataset_1 = TestDataset('{}'.format(data_path), label=1, transform=transform)

model = TestModel()
state_path = './weights/resnet_model_3c/train_1.weights'
state_dict = torch.load(state_path)
model.load_state_dict(state_dict)

prediction_dict = {}

correct = 0
total = 0
incorrect = 0

for i0 in range(len(dataset_0)):
    sample, l = dataset_0[i0]
    data = sample['img']
    filename = sample['filename']

    prediction = np.argmax(model(data))

    prediction_dict[filename] = prediction

    if prediction == l:
        correct +=1
    else:
        incorrect += 1

    total += 1

for i1 in range(len(dataset_1)):
    sample, l = dataset_0[i1]
    data = sample['img']
    filename = sample['filename']

    prediction = np.argmax(model(data))

    prediction_dict[filename] = prediction

    if prediction == l:
        correct += 1
    else:
        incorrect += 1

    total += 1

print('Corrent %:', correct/total)
print('Inorrent %:', incorrect/total)
