import torch
import torchbearer
from torch import nn
from torch import optim
from torchbearer import Trial
import torchvision.transforms as transforms

from models.test_model_2 import TestModel
from data_loader import loader
import transforms as tfs

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


batch_size = 12
validation_split = .2
shuffle_dataset = True

train_loader, validation_loader = loader('{}/train'.format(data_path), labels_path, batch_size, validation_split, p=1, transform=trans)

model = TestModel()

model.train()

loss_function = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters())

device = "cuda:0" if torch.cuda.is_available() else "cpu"
trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, test_generator=validation_loader)
trial.run(epochs=10)
results = trial.evaluate(data_key=torchbearer.TEST_DATA)
print(results)

torch.save(model.state_dict(), "./weights/test.weights")