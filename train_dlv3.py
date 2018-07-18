
from models.deeplabv3plus import DeepLabV3Plus
from utils.loss import CrossEntropyLoss2d
from torchvision import transforms
from dataloader import ImageSet
from torch.utils.data import DataLoader
import torch
import cv2
import numpy as np
import utility


def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.int32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(new_target).long()


def train(epoch, net, dl, optimizer, criterion, device):
    net.train()
    # When do you want to freeze batch norms?
    net.module.freeze_bn()
    for batch_idx, (data, target) in enumerate(dl):
        data = data.to(device)
        # Zero out gradients from previous step
        optimizer.zero_grad()
        # Forward pass of the neural net
        outputs = net(data)
        # Calculation of the loss function
        loss = 0
        # Upscale ?
        for output in outputs:
            # Resize target for {100%, 75%, 50%, Max} outputs
            target_ = resize_target(target, output.size(2))
            target_ = target_.to(device)
            # Compute crossentropy loss
            loss += criterion(output, target_)

        #loss = cross_entropy2d(output, target.squeeze()) #2D version
        # Backward pass (gradient computation)
        loss.backward()
        # Adjusting the parameters according to the loss function
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dl.dataset),
                100. * batch_idx / len(dl), loss.item()))


def test(net, criterion, device, val, save, pre):
    net.eval()
    test_loss = 0
    with torch.no_grad():
        imgname = 1
        for data, target in val:
            data = data.to(device)

            outputs = net(data)
            for output in outputs:
                # Resize target for {100%, 75%, 50%, Max} outputs
                target_ = resize_target(target, output.size(2))
                target_ = target_.to(device)
                # Compute crossentropy loss
                test_loss += criterion(output, target_)

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

    test_loss /= len(val.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def main():
    device = torch.device('cuda')
    tf = transforms.Compose([
        transforms.ToTensor()
    ])
    net = DeepLabV3Plus(n_classes=4, n_blocks=[3, 4, 23, 3], pyramids=[6, 12, 18])
    net = net.to(device)
    net = torch.nn.DataParallel(net)

    #net.load_state_dict(torch.load('snapshots/snapshot_99.pt'))

    ts = ImageSet(1)
    vs = ImageSet(2, False)
    dl = DataLoader(ts, batch_size=8)
    vl = DataLoader(vs, batch_size=8)

    criterion = CrossEntropyLoss2d()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5.0e-4)

    for it in range(1, 100):
        train(it, net, dl, optimizer, criterion, device)
        test(net, criterion, device, vl, False, 't')
        torch.save(net.state_dict(), '/mnt/disks/data/d4dl/d4dl/snapshots/snapshot_' + str(it) + '.pt')


if __name__ == '__main__':
    main()
