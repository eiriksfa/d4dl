
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
from models.deeplab_v3_plus import DeepLabv3_plus


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


# def test(net, criterion, device, val, save, pre):
#     net.eval()
#     test_loss = 0
#     correct = 0
#     intersect_all = 0
#     union_all = 0
#     with torch.no_grad():
#         imgname = 1
#         for data, target in val:
#             data, target = data.to(device), target.to(device)
#             t = time.time()
#             output = net(data)
#             print(time.time() - t)
#             test_loss += criterion(output, target).item()  # sum up batch loss
#             pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#             pred = torch.squeeze(pred)
#
#             if save:  # one per batch as is
#                 for i in range(len(output)):
#                     out = utility.output_labels_to_image(output[i].cpu())
#                     out = cv2.normalize(out, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
#                     out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
#                     img = data[i].cpu().numpy().transpose((1, 2, 0))
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                     img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
#                     cv2.addWeighted(img, 0.7, out, 0.3, 0, out)
#                     cv2.imwrite('images/out_' + pre + '_' + str(imgname) + '_' + str(i) + '.png', out)
#             imgname += 1
#             if val.batch_size == 1:
#                 intr, uni = intersectionAndUnion(pred, target, 3)
#                 intersect_all += intr
#                 union_all += uni
#             else:
#                 for i in range(0, len(pred)):
#                     intr, uni = intersectionAndUnion(pred[i], target[i], 3)
#                     intersect_all += intr
#                     union_all += uni
#
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(val.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, (len(val.dataset)*128*256),
#         100. * correct / (len(val.dataset)*128*256)))
#     iou = intersect_all / union_all
#     for i, _iou in enumerate(iou):
#         print('class [{}], IoU: {}'.format(i, _iou))
#     print("Mean IoU : {:.4}".format(iou.mean()))
#     return 100. * correct / (len(val.dataset)*128*256)


def main():
    tf = transforms.Compose([
        transforms.ToTensor()
    ])

    device = 'cuda'
    net = DeepLabv3_plus(num_classes=2)
    net = net.to(device)
    net = torch.nn.DataParallel(net)

    ts = ImageSet(1)
    vs = ImageSet(2, False)
    # tts = ImageSet(3, False)
    dl = DataLoader(ts, batch_size=1)
    vl = DataLoader(vs, batch_size=4)
    # tl = DataLoader(tts, batch_size=12)

    #single_pass(net, device, vl)

    #criterion = nn.NLLLoss2d()
    #net.load_state_dict(torch.load('snapshots/snapshot_1.pt'))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0025, momentum=0.9)
    accuracy = []
    for e in range(22, 40):
        train(e, net, optimizer, criterion, device, dl)
    #     a = test(net, criterion, device, vl, False, 'v')
    #     torch.save(net.state_dict(), '/mnt/disks/data/d4dl/d4dl/snapshots/snapshot_' + str(e) + '.pt')
    #     accuracy.append(a)
    #torch.save(dict, 'snapshots/snapshot.pt')
    # plt.plot(accuracy, range(len(accuracy)))
    # plt.savefig('images/plot.png')
    # test(net, criterion, device, dl, True, 'd')
    #test(net, criterion, device, vl, True, 'v')
    # test(net, criterion, device, tl, True, 't')

    # torch.save(net.state_dict(), 'snapshots/model.pt')


if __name__ == '__main__':
    main()
