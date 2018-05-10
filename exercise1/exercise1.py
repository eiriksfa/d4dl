
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class SimpleNet(nn.Module):
    """
    Task 2
    Several layers of 3x3 conv filters, stride 1, and 2x2 max pooling, stride 2
    ReLU activation functions, one fully connected layer with 512 neurons, one softmax
    cross entropy loss, sgd
    """

    def _build_vgg16(self):
        # .double needed, probably because of the use of numpy arrays (?)
        layers = [nn.Conv2d(3, 32, kernel_size=3, padding=1).double(), nn.BatchNorm2d(32).double(), nn.ReLU(inplace=True).double(),
                  #nn.Conv2d(32, 32, kernel_size=3, padding=1).double(), nn.BatchNorm2d(32).double(), nn.ReLU(inplace=True).double(),
                  self.pool,
                  nn.Conv2d(32, 64, kernel_size=3, padding=1).double(), nn.BatchNorm2d(64).double(), nn.ReLU(inplace=True).double(),
                  #nn.Conv2d(64, 64, kernel_size=3, padding=1).double(), nn.BatchNorm2d(64).double(), nn.ReLU(inplace=True).double(),
                  self.pool,
                  nn.Conv2d(64, 128, kernel_size=3, padding=1).double(), nn.BatchNorm2d(128).double(), nn.ReLU(inplace=True).double(),
                  nn.Conv2d(128, 128, kernel_size=3, padding=1).double(), nn.BatchNorm2d(128).double(), nn.ReLU(inplace=True).double(),
                  # nn.Conv2d(256, 256, kernel_size=3, padding=1).double(), nn.ReLU(inplace=True).double(),
                  self.pool,
                  nn.Conv2d(128, 256, kernel_size=3, padding=1).double(), nn.BatchNorm2d(256).double(), nn.ReLU(inplace=True).double(),
                  nn.Conv2d(256, 256, kernel_size=3, padding=1).double(), nn.BatchNorm2d(256).double(), nn.ReLU(inplace=True).double(),
                  # nn.Conv2d(512, 512, kernel_size=3, padding=1).double(), nn.ReLU(inplace=True).double(),
                  self.pool,
                  nn.Conv2d(256, 256, kernel_size=3, padding=1).double(), nn.BatchNorm2d(256).double(), nn.ReLU(inplace=True).double(),
                  #nn.Conv2d(256, 256, kernel_size=3, padding=1).double(), nn.BatchNorm2d(256).double(), nn.ReLU(inplace=True).double(),
                  # nn.Conv2d(512, 512, kernel_size=3, padding=1).double(), nn.ReLU(inplace=True).double(),
                  self.pool,
                  nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layers = self._build_vgg16()
        self.fc1 = nn.Linear(256, 512).double()
        self.fc3 = nn.Linear(512, 1024).double()
        self.fc2 = nn.Linear(1024, 43).double()

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)))
        x = F.dropout(F.relu(self.fc3(x)))

        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


# This function trains the neural network for one epoch
def train(epoch, net, train, optimizer, criterion, device):
    net.train()
    for batch_idx, (data, target) in enumerate(train):
        # Move the input and target data on the GPU
        data, target = data.to(device), target.to(device)
        # Zero out gradients from previous step
        optimizer.zero_grad()
        # Forward pass of the neural net
        output = net(data)
        # Calculation of the loss function
        loss = criterion(output, target)
        # Backward pass (gradient computation)
        loss.backward()
        # Adjusting the parameters according to the loss function
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train.dataset),
                100. * batch_idx / len(train), loss.item()))


def test(net, val, device, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val.dataset),
        100. * correct / len(val.dataset)))
    return correct


# Not very large dataset, so I will pre-process (no memory issues)
def pre_process(path, fname, load=False):
    # Question: When should 0.5 be used instead of the actual mean/std?
    mr, mg, mb = 0.33999, 0.31190, 0.32119  # 0.0, 0.0, 0.0
    sr, sg, sb = 0.09047, 0.07557, 0.07636  # 0.0, 0.0, 0.0

    images = []
    if not load:
        set = [pth for pth in path.iterdir()]
        data = None
        for dir in set:
            csv = [file for file in dir.iterdir() if file.suffix == '.csv'][0]  # Not the fastest way to 'search', cba
            df = pd.read_csv(csv, sep=';')
            df['Stem'] = [dir.stem for _ in range(len(df))]
            if data is None:
                data = df
            else:
                data = data.append(df)
        c = 1
        for i in range(len(data)):  # len(data)
            print(i)
            item = data.iloc[i]
            name = item['Stem'] + '/' + item['Filename']
            p = path.joinpath(name)
            img = io.imread(p)
            img = transform.resize(img, (32, 32))
            # for h in img:
            #     for w in h:
            #         tmp = mr
            #         mr += (w[0] - tmp) / c
            #         sr = (w[0] - tmp) * (w[0] -mr)
            #         tmp = mg
            #         mg += (w[1] - tmp) / c
            #         sg = (w[1] - tmp) * (w[1] - mg)
            #         tmp = mb
            #         mb += (w[2] - tmp) / c
            #         sb = (w[2] - tmp) * (w[2] - mb)
            #         c += 1

            images.append((img, item['ClassId']))
        images = np.array(images)
        np.save(fname, images)
    else:
        images = np.load(fname)
    np.random.shuffle(images)
    tf = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])#mean=[mr, mg, mb], std=[sr, sg, sb])
    ])
    return images, tf


class ImageSet(Dataset):

    def __init__(self, data, tf):
        self.tf = tf
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        itm = self.data[item]
        img = itm[0]
        # img = img.astype(np.float)
        return self.tf(img), itm[1]


if __name__ == '__main__':

    # Pre-process
    path = Path.cwd().joinpath('data/GTSRB_Training/GTSRB/Final_Training/Images')
    dpath = Path.cwd().joinpath('data/data.npy')
    data, tf = pre_process(path, dpath, load=True)
    split = int(np.round(len(data)*0.8))
    training = data[0:split]
    validation = data[split:len(data)]

    ts = ImageSet(training, tf)
    vs = ImageSet(validation, tf)
    ts_loader = torch.utils.data.DataLoader(ts, batch_size=128)
    vs_loader = torch.utils.data.DataLoader(vs, batch_size=128)

    print("Finished preprocessing data")
    device = 'cuda'
    net = SimpleNet()
    net = net.to(device)
    net = torch.nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    accuracy = []
    for e in range(1, 7):
        train(e, net, ts_loader, optimizer, criterion, device)
        a = test(net, vs_loader, device, criterion)
        accuracy.append(a)

    plt.figure()
    plt.plot(np.arange(1, 7), accuracy)
    plt.savefig('plot.png')


