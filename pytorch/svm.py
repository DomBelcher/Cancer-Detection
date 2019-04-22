import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics

from models.resnet_model_2_headless import TestModel
from data_loader import loader
import transforms as tfs

model = TestModel()
model.load_state_dict(torch.load('./weights/test.weights'))
model.eval()
model = model.float()

trans = transforms.Compose([
    # tfs.RandomRotation(range=(0, 360)),
    tfs.Downsize(),
    tfs.Normalise(),
    # tfs.ChannelShift(),d
    transforms.ToTensor()
])

batch_size = 1
validation_split = .2
shuffle_dataset = True

data_path = '../Data'
labels_path = '{}/train_labels/train_labels.csv'.format(data_path)
train_loader, validation_loader = loader('{}/train'.format(data_path), labels_path, batch_size, validation_split, p=1, transform=trans)

# inputs, labels = next(iter(train_loader))
# print(labels)
# print(inputs)

# print(model(inputs.float()))
# print(labels)

try:
    features = np.load('./features/resnet_2_features')
    labels = np.load('./features/resnet_2_labels')
except IOError:
    features = np.zeros((0, 128))
    labels = np.zeros(0)

    for i, data in enumerate(train_loader, 0):
        inputs, label = data
        inputs, label = Variable(inputs).float(), Variable(label).float()
        inputs.requires_grad = False

        outputs = model(inputs)

        features = np.append(features, outputs.detach().numpy(), axis=0)
        labels = np.append(labels, label)

    np.save('./features/resnet_2_features', features)
    np.save('./features/resnet_2_labels', labels)

try:
    v_features = np.load('./features/resnet_2_features_v')
    v_labels = np.load('./features/resnet_2_labels_v')
except IOError:
    v_features = np.zeros((0, 128))
    v_labels = np.zeros(0)

    for i, data in enumerate(validation_loader, 0):
        inputs, label = data
        inputs, label = Variable(inputs).float(), Variable(label).float()
        inputs.requires_grad = False

        outputs = model(inputs)

        v_features = np.append(v_features, outputs.detach().numpy(), axis=0)
        v_labels = np.append(v_labels, label)

    np.save('./features/resnet_2_features_v', v_features)
    np.save('./features/resnet_2_labels_v', v_labels)

svc = SVC(C=25, gamma='scale')
svc.fit(features, labels)

print('Validation Data Report')
train_pred = svc.predict(v_features)
print(metrics.classification_report(v_labels, train_pred))