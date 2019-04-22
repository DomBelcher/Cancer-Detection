from torch import nn
import torch.nn.functional as F

maps = 32

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1a = nn.Conv2d(3, maps, (3, 3), padding=1)
        self.conv1b = nn.Conv2d(maps, maps, (3, 3), padding=1)
        self.conv1c = nn.Conv2d(3, maps, (1, 1))
        self.bn1 = nn.BatchNorm2d(maps)

        self.conv2a = nn.Conv2d(maps, maps * 2, (3, 3), padding=1)
        self.conv2b = nn.Conv2d(maps * 2, maps * 2, (3, 3), padding=1)
        self.conv2c = nn.Conv2d(maps, maps * 2, (1, 1))
        self.bn2 = nn.BatchNorm2d(maps * 2)

        self.conv3a = nn.Conv2d(maps * 2, maps * 4, (3, 3), padding=1)
        self.conv3b = nn.Conv2d(maps * 4, maps * 4, (3, 3), padding=1)
        self.conv3c = nn.Conv2d(maps * 2, maps * 4, (1, 1))
        self.bn3 = nn.BatchNorm2d(maps * 4)

        self.fc1 = self.fc1 = nn.Linear((maps * 4) * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 2)

        self.max_pool = nn.MaxPool2d(2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out_ = self.conv1a(x)
        out_ = F.relu(out_)
        out_ = self.conv1b(out_)
        out = self.conv1c(x)
        out = out + out_
        out = F.relu(out)
        out = self.max_pool(out)
        out = self.bn1(out)

        # print(out.shape)
        out_ = self.conv2a(out)
        out_ = F.relu(out_)
        out_ = self.conv2b(out_)
        # print(out.shape)
        out = self.conv2c(out)
        out = out + out_
        out = F.relu(out)
        out = self.max_pool(out)
        out = self.bn2(out)

        out_ = self.conv3a(out)
        out_ = F.relu(out_)
        out_ = self.conv3b(out_)
        out = self.conv3c(out)
        out = out + out_
        out = F.relu(out)
        out = self.max_pool(out)
        out = self.bn3(out)

        # print(out.shape)
        out = out.view(out.shape[0], -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = nn.Dropout(0.25)(out)
        out = self.fc2(out)
        # out = self.sigmoid(out)

        #         print(out)

        return out