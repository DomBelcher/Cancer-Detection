import pickle

import torch
from torch import nn
from torch import optim
import torchvision.transforms as transforms
import numpy as np

from models.resnet_model_3c import TestModel
from data_loader_2 import test_dataset
from test_dataset import TestDataset
import transforms as tfs

np_transorms = transforms.Compose([
    # transforms.CenterCrop(48),
    # tfs.PillowToNumpy(),
    # tfs.RandomRotation(range=(0, 360)),
    tfs.Downsize(),
    # tfs.Normalise(),
    # tfs.ChannelShift(),
    transforms.ToTensor()
])

pillow_transforms = transforms.Compose([
    transforms.CenterCrop(48),
    transforms.ToTensor()
])


test_loader = test_dataset(batch_size=128, transform=pillow_transforms)

data_path = '../data/train/Testing'
dataset_0 = TestDataset('{}'.format(data_path), label=0, transform=np_transorms)
dataset_1 = TestDataset('{}'.format(data_path), label=1, transform=np_transorms)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

model = TestModel()
model.to(device)
state_path = './weights/resnet_model_3c/train_1.weights'
state_dict = torch.load(state_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

prediction_dict = {}

correct_0 = 0
correct_1 = 0
total_0 = 0
total_1 = 0
incorrect_0 = 0
incorrect_1 = 0

'''
print('Batch 0')
for i0 in range(len(dataset_0)):
    sample, l = dataset_0[i0]
    data = sample['image'].unsqueeze(0).float().to(device)
    filename = sample['filename']

    prediction = torch.argmax(model(data)).item()
    # print(prediction)

    prediction_dict[filename] = prediction

    if prediction == l:
        # print('correct')
        correct_0 +=1
    else:
        # print('incorrect')
        incorrect_0 += 1

    total_0 += 1


print('Batch 1')
for i1 in range(len(dataset_1)):
    sample, l = dataset_1[i1]
    data = sample['image'].unsqueeze(0).float().to(device)
    filename = sample['filename']

    prediction = torch.argmax(model(data)).item()
    # print('prediction')
    # print(model(data))
    prediction_dict[filename] = prediction

    if prediction == l:
        correct_1 += 1
    else:
        incorrect_1 += 1

    total_1 += 1

pickle.dump(prediction_dict, open('predictions.p', 'wb'))

print('Correct 0 %:', correct_0/total_0)
print('Correct 1 %:', correct_1/total_1)
print('Total Correct %:', (correct_0 + correct_1)/(total_0 + total_1))

'''

n_classes = 2

confusion_matrix = torch.zeros(n_classes, n_classes)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # print(images.to(device))
        # images.cuda()
        # images.type(torch.cuda.FloatTensor)
        # print(labels.to(device))
        # labels.type(torch.cuda.FloatTensor)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1


print(confusion_matrix/confusion_matrix.sum(axis=1))

print('Test Accuracy: %d %%' % (
        100 * correct / total))


