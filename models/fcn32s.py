
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from skimage import io
import numpy as np
import matplotlib.pyplot as plt


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN32s(nn.Module):

    def __init__(self):
        super(FCN32s, self).__init__()
        self.n_class = 3
        self.layers = self._build_layers(self.n_class)

        # init weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):  # TODO: batchnorm inits?
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # Bilinear (No changes?)
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = self.layers(x)
        # TODO: Debug/Test Resize etc
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
        return h

    @classmethod
    def _build_layers(cls, n_class):
        # .double needed, probably because of the use of numpy arrays (?)
        layers = [
                    # Conv1 # TODO: Padding?
                    nn.Conv2d(3, 32, kernel_size=3, padding=100), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
                    # Conv2
                    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
                    # Conv3
                    nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
                    # Conv4
                    nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
                    # Conv5
                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/32
                    # fc6
                    nn.Conv2d(256, 2048, 7), nn.ReLU(inplace=True), nn.Dropout2d(),
                    # fc7
                    nn.Conv2d(2048, 2048, 7), nn.ReLU(inplace=True), nn.Dropout2d(),
                    # fc8
                    nn.Conv2d(2048, n_class, 1),
                    # Transpose convolution
                    nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)

                    # nn.AvgPool2d(kernel_size=1, stride=1)
                 ]
        return nn.Sequential(*layers)


# Functions used for testing the model (consider moving into a util file or something, for use in all models)
def train(net, optimizer, criterion, device, data, target):
    net.train()
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


def load_images():
    pass


if __name__ == '__main__':
    tf = transforms.Compose([
        transforms.ToTensor()
    ])

    device = 'cuda'
    net = FCN32s()
    net = net.to(device)
    net = torch.nn.DataParallel(net)

    img = io.imread('C:/Users/eirik/PycharmProjects/d4dl2/testimg/2.png')
    target = io.imread('C:/Users/eirik/PycharmProjects/d4dl2/testimg/1.png')
    img, target = tf(img), tf(target)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    accuracy = []
    for e in range(1, 30):
        train(net, optimizer, criterion, device, img, target)
