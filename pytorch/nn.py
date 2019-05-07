import os

import torch
import torchbearer
from torch import nn
from torch import optim
from torchbearer import Trial
import torchvision.transforms as transforms

from models.resnet_model_3a import TestModel
from data_loader import loader
from data_loader_2 import load_datasets
import transforms as tfs
from helpers import deactivate_layer

data_path = '../Data'
labels_path = '{}/train_labels/train_labels.csv'.format(data_path)
#
# train_labels = pd.read_csv(labels_path)
#
# print(len(train_labels))

trans = transforms.Compose([
    tfs.RandomRotation(range=(0, 360)),
    tfs.Downsize(),
    tfs.Normalise(),
    # tfs.ChannelShift(),
    transforms.ToTensor()
])


batch_size = 128
validation_split = .2
shuffle_dataset = True
p = 1

# train_loader, validation_loader = loader('{}/train'.format(data_path), labels_path, batch_size, validation_split, p=p, transform=trans)
train_loader, validation_loader = load_datasets(batch_size=128, transforms=trans)

model = TestModel()
model.train()

'''
state_path = './weights/resnet_model_3/train_0.1.weights'
state_dict = torch.load(state_path)
# state_dict.pop('fc1.weight')
# state_dict.pop('fc1.bias')
state_dict.pop('fc2.weight')
state_dict.pop('fc2.bias')
# state_dict['fc3'] = None
model.load_state_dict(state_dict, strict=False)
to_deactivate = ['conv1a', 'conv1b', 'conv1c', 'conv2a', 'conv2b', 'conv2c', 'conv3a', 'conv3b', 'conv3c']
for l in to_deactivate:
    deactivate_layer(model, l)
'''

loss_function = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters())

device = "cuda:0" if torch.cuda.is_available() else "cpu"
trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, test_generator=validation_loader)
trial.run(epochs=10)
results = trial.evaluate(data_key=torchbearer.TEST_DATA)
print(results)

model_name = 'resnet_model_3b'
weights_path = './weights/{}/'.format(model_name)

if not os.path.exists(weights_path):
        os.makedirs(weights_path)

w_filepath = './weights/{}/train_{}.weights'.format(model_name, p)
torch.save(model.state_dict(), w_filepath)
