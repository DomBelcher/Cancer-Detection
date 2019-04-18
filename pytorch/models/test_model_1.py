from torch import nn
import torch.nn.functional as F

maps = 32

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1a = nn.Conv2d(3, maps, (3, 3), padding=1)
        self.conv1b = nn.Conv2d(maps, maps, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(maps)

        self.conv2a = nn.Conv2d(maps, maps * 2, (4, 4), padding=1, stride=2)
        self.conv2b = nn.Conv2d(maps * 2, maps * 2, (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(maps * 2)

        self.conv3a = nn.Conv2d(maps * 2, maps * 4, (4, 4), padding=1, stride=2)
        self.conv3b = nn.Conv2d(maps * 4, maps * 4, (3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(maps * 4)

        self.fc1 = self.fc1 = nn.Linear((maps * 4) * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1a(x)
        out = F.relu(out)
        out = self.conv1b(out)
        out = F.relu(out)
        out = self.bn1(out)

        out = self.conv2a(out)
        out = F.relu(out)
        out = self.conv2b(out)
        out = F.relu(out)
        out = self.bn2(out)

        out = self.conv3a(out)
        out = F.relu(out)
        out = self.conv3b(out)
        out = F.relu(out)
        out = self.bn3(out)

        #         print(out.shape)
        out = out.view(out.shape[0], -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        # out = self.sigmoid(out)

        #         print(out)

        return out