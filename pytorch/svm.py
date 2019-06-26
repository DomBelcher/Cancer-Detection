import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from joblib import dump

from models.resnet_model_3a_headless import TestModel
from data_loader import loader
import transforms as tfs

model = TestModel()
model.load_state_dict(torch.load('./weights/resnet_model_3a/train_0.1.weights'), strict=False)
model.eval()
model = model.float()

trans = transforms.Compose([
    # tfs.RandomRotation(range=(0, 360)),
    tfs.Downsize(),
    tfs.Normalise(),
    # tfs.ChannelShift(),d
    transforms.ToTensor()
])

p = 0.1

batch_size = 64
validation_split = .2
shuffle_dataset = True

data_path = '../Data'
labels_path = '{}/train_labels/train_labels.csv'.format(data_path)
train_loader, validation_loader = loader('{}/train'.format(data_path), labels_path, batch_size, validation_split, p=p, transform=trans)

# inputs, labels = next(iter(train_loader))
# print(labels)
# print(inputs)

# print(model(inputs.float()))
# print(labels)

n_features = 512

try:
    features = np.load('./features/resnet_3a_features_{}.npy'.format(p))
    labels = np.load('./features/resnet_3a_labels_{}.npy'.format(p))
    print('Loaded training data')
except IOError:
    print('Failed to load training data')
    features = np.zeros((0, n_features))
    labels = np.zeros(0)

    for i, data in enumerate(train_loader, 0):
        inputs, label = data
        inputs, label = Variable(inputs).float(), Variable(label).float()
        inputs.requires_grad = False

        outputs = model(inputs)

        features = np.append(features, outputs.detach().numpy(), axis=0)
        labels = np.append(labels, label)

    np.save('./features/resnet_3a_features_{}'.format(p), features)
    np.save('./features/resnet_3a_labels_{}'.format(p), labels)

try:
    v_features = np.load('./features/resnet_3a_features_{}_v.npy'.format(p))
    v_labels = np.load('./features/resnet_3a_labels_{}_v.npy'.format(p))
    print('Loaded validatiom data')
except IOError:
    print('Failed to load validation data')
    v_features = np.zeros((0, n_features))
    v_labels = np.zeros(0)

    for i, data in enumerate(validation_loader, 0):
        inputs, label = data
        inputs, label = Variable(inputs).float(), Variable(label).float()
        inputs.requires_grad = False

        outputs = model(inputs)

        v_features = np.append(v_features, outputs.detach().numpy(), axis=0)
        v_labels = np.append(v_labels, label)

    np.save('./features/resnet_3a_features_{}_v'.format(p), v_features)
    np.save('./features/resnet_3a_labels_{}_v'.format(p), v_labels)

print('Training SVM')
svc = SVC(C=25, gamma='scale', verbose=1)
svc.fit(features, labels)

print('Training Data Report')
train_pred = svc.predict(features)
print(metrics.classification_report(labels, train_pred))

print('Validation Data Report')
valid_pred = svc.predict(v_features)
print(metrics.classification_report(v_labels, valid_pred))

dump(svc, './classifiers/resnet-svm_{}.joblib'.format(p))