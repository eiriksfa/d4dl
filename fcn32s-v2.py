
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
import torch.nn.functional as F
from dataloader import ImageSet, ImageSet2
from torch.utils.data import DataLoader

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

# from https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/trainer.py
def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    print(log_p.size())
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

class FCN32s(nn.Module):
    def __init__(self, n_class=3):
        super(FCN32s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4
        # conv3
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
        # conv5
        self.conv5_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32
        # fc6
        self.fc6 = nn.Conv2d(256, 2048, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(2048, 2048, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(2048, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 32, stride=32, bias=False)
        #self.upscore = nn.UpsamplingBilinear2d(scale_factor=2)

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
        h = x
        print(h.shape)
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)
        print(h.shape)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        print(h.shape)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)
        print(h.shape)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        print(h.shape)

        h = self.score_fr(h)
        print(h.shape)

        h = self.upscore(h)
        print(h.shape)
        #h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return h


# Functions used for testing the model (consider moving into a util file or something, for use in all models)
def train(net, optimizer, criterion, device, train):
    net.train()
    for batch_idx, (data, target) in enumerate(train):
        # Move the input and target data on the GPU
        data, target = data.to(device), target.to(device)
        # Zero out gradients from previous step
        #print(target)
        optimizer.zero_grad()
        # Forward pass of the neural net
        output = net(data)
        # Calculation of the loss function
        loss = criterion(output, target)
        #loss = cross_entropy2d(output, target.squeeze()) #2D version
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

    # ts = ImageSet()
    # dl = DataLoader(ts, batch_size=4)

    ts = ImageSet2()
    dl = DataLoader(ts, batch_size=1)
    print(dl)

    #img = io.imread('/home/novian/term2/dl4ad/repo2/d4dl/testimg/2.png')
    #target = io.imread('/home/novian/term2/dl4ad/repo2/d4dl/testimg/1.png')

    # img, target = tf(img), tf(target)
    # img.unsqueeze_(0)
    # #target.unsqueeze_(0)
    # print(img.shape)
    # print(target.shape)
    #criterion = nn.CrossEntropyLoss() // using 2d version instead
    criterion = nn.NLLLoss2d()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    accuracy = []
    for e in range(1, 30):
        train(net, optimizer, criterion, device, dl)
