
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import utility
from distutils.version import LooseVersion
import torch.nn.functional as F
from dataloader import ImageSet
from torch.utils.data import DataLoader
import cv2
import time


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


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class FCN32s(nn.Module):
    def __init__(self, n_class=3):
        super(FCN32s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.batch1_1 = nn.BatchNorm2d(32)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.batch1_2 = nn.BatchNorm2d(32)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.batch2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.batch2_2 = nn.BatchNorm2d(64)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4
        # conv3
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.batch3_1 = nn.BatchNorm2d(128)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.batch3_2 = nn.BatchNorm2d(128)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.batch3_3 = nn.BatchNorm2d(128)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.batch4_1 = nn.BatchNorm2d(256)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.batch4_2 = nn.BatchNorm2d(256)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.batch4_3 = nn.BatchNorm2d(256)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
        # conv5
        self.conv5_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.batch5_1 = nn.BatchNorm2d(256)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.batch5_2 = nn.BatchNorm2d(256)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(256, 256, 3)
        self.batch5_3 = nn.BatchNorm2d(256)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32
        # fc6
        self.fc6 = nn.Conv2d(256, 2048, 7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(2048, 2048, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(2048, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)
        #self.upscore = nn.UpsamplingBilinear2d(scale_factor=2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
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
        h = self.relu1_1(self.batch1_1(self.conv1_1(h)))
        h = self.relu1_2(self.batch1_2(self.conv1_2(h)))
        h = self.pool1(h)

        h = self.relu2_1(self.batch2_1(self.conv2_1(h)))
        h = self.relu2_2(self.batch2_2(self.conv2_2(h)))
        h = self.pool2(h)

        h = self.relu3_1(self.batch3_1(self.conv3_1(h)))
        h = self.relu3_2(self.batch3_2(self.conv3_2(h)))
        h = self.relu3_3(self.batch3_3(self.conv3_3(h)))
        h = self.pool3(h)

        h = self.relu4_1(self.batch4_1(self.conv4_1(h)))
        h = self.relu4_2(self.batch4_2(self.conv4_2(h)))
        h = self.relu4_3(self.batch4_3(self.conv4_3(h)))
        h = self.pool4(h)

        h = self.relu5_1(self.batch5_1(self.conv5_1(h)))
        h = self.relu5_2(self.batch5_2(self.conv5_2(h)))
        h = self.relu5_3(self.batch5_3(self.conv5_3(h)))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore(h)
        #h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return h


# Functions used for testing the model (consider moving into a util file or something, for use in all models)
def train(epoch, net, optimizer, criterion, device, train):
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
        #loss = cross_entropy2d(output, target.squeeze()) #2D version
        # Backward pass (gradient computation)
        loss.backward()
        # Adjusting the parameters according to the loss function
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train.dataset),
                100. * batch_idx / len(train), loss.item()))


# https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/utils.py
def intersectionAndUnion(pred, lab, numClass):
    pred = np.asarray(pred).copy()
    lab = np.asarray(lab).copy()

    pred += 1
    lab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pred = pred * (lab > 0)

    # Compute area intersection:
    intersection = pred * (pred == lab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(pred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(lab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


def test(net, criterion, device, val, save, pre):
    net.eval()
    test_loss = 0
    correct = 0
    intersect_all = 0
    union_all = 0
    with torch.no_grad():
        imgname = 1
        for data, target in val:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred = torch.squeeze(pred)

            if save:  # one per batch as is
                for i in range(len(output)):
                    out = utility.output_labels_to_image(output[i].cpu())
                    out = cv2.normalize(out, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                    img = data[i].cpu().numpy().transpose((1, 2, 0))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                    cv2.addWeighted(img, 0.7, out, 0.3, 0, out)
                    cv2.imwrite('images/out_' + pre + '_' + str(imgname) + '_' + str(i) + '.png', out)
            imgname += 1
            if val.batch_size == 1:
                intr, uni = intersectionAndUnion(pred, target, 3)
                intersect_all += intr
                union_all += uni
            else:
                for i in range(0, len(pred)):
                    intr, uni = intersectionAndUnion(pred[i], target[i], 3)
                    intersect_all += intr
                    union_all += uni

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, (len(val.dataset)*128*256),
        100. * correct / (len(val.dataset)*128*256)))
    iou = intersect_all / union_all
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(i, _iou))
    print("Mean IoU : {:.4}".format(iou.mean()))
    return 100. * correct / (len(val.dataset)*128*256)


def fp_video(net, device, dl):
    # NB: Asserts a batch size of 1 atm
    net.eval()
    with torch.no_grad():
        t = time.time()
        for batch_idx, (data, target, name) in enumerate(dl):
            data = data.to(device)
            output = net(data)
            name = str(name).split("'")[1]
            print(name)
            for i in range(len(output)):
                out = utility.output_labels_to_image(output[i].cpu())
                out = cv2.normalize(out, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                img = data[i].cpu().numpy().transpose((1, 2, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                cv2.addWeighted(img, 0.7, out, 0.3, 0, out)
                cv2.imwrite('images/' + str(name) + '.png', out)
        print(time.time()-t)


def main():
    tf = transforms.Compose([
        transforms.ToTensor()
    ])

    device = 'cuda'
    net = FCN32s()
    net = net.to(device)
    net = torch.nn.DataParallel(net)

    # ts = ImageSet(1)
    # vs = ImageSet(2, False)
    # # tts = ImageSet(3, False)
    # dl = DataLoader(ts, batch_size=24)
    # vl = DataLoader(vs, batch_size=1)
    # tl = DataLoader(tts, batch_size=12)

    testset = ImageSet(0, False, True)
    # tts = ImageSet(3, False)
    # dl = DataLoader(ts, batch_size=24)
    # vl = DataLoader(vs, batch_size=1)
    testloader = DataLoader(testset, batch_size=1)
    # single_pass(net, device, vl)

    # criterion = nn.NLLLoss2d()
    net.load_state_dict(torch.load('snapshots/snapshot_26.pt'))
    criterion = CrossEntropyLoss2d()
    # optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    # accuracy = []
    # for e in range(1, 100):
    #     train(e, net, optimizer, criterion, device, dl)
    #     a = test(net, criterion, device, vl, False, 'v')
    #     torch.save(net.state_dict(), '/mnt/disks/data/d4dl/snapshots/snapshot_' + str(e) + '.pt')
    #     accuracy.append(a)
    # torch.save(dict, 'snapshots/snapshot.pt')
    # plt.plot(accuracy, range(len(accuracy)))
    # plt.savefig('images/plot.png')
    # test(net, criterion, device, dl, True, 'd')
    fp_video(net, device, testloader)
    #test(net, criterion, device, vl, True, 'v')
    # test(net, criterion, device, tl, True, 't')

    # torch.save(net.state_dict(), 'snapshots/model.pt')


if __name__ == '__main__':
    pass
