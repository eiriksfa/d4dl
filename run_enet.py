
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from dataloader import ImageSet
from models.enet import ENet
from train_enet import Train
from test_enet import Test
from utils.iou import IoU
from utils.enet import enet_weighing, median_freq_balancing
import utils

num_classes = 2


def load_dataset():
    print("\nLoading dataset...\n")

    train_set = ImageSet(1)
    train_loader = data.DataLoader(train_set, batch_size=2, shuffle=True, num_workers=9)
    val_set = ImageSet(2, False)
    val_loader = data.DataLoader(val_set, batch_size=2, shuffle=True, num_workers=9)

    # Get class weights from the selected weighing technique
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")
    class_weights = 0
    # enet
    # class_weights = enet_weighing(train_loader, num_classes)
    # mfb
    # class_weights = median_freq_balancing(train_loader, num_classes)
    # else
    class_weights = None

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float()

    print("Class weights:", class_weights)

    return train_loader, val_loader, class_weights


def train(train_loader, val_loader, class_weights):
    model = ENet(num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=5e-4,
        weight_decay=2e-4)
    lr_updater = lr_scheduler.StepLR(optimizer, 10, 1e-7)  # Large dataset, decaying every 10 epochs..
    ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    model = model.cuda()
    criterion = criterion.cuda()

    # model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
    #        model, optimizer, args.save_dir, args.name)
    # print("Resuming from model: Start epoch = {0} "
    #       "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    start_epoch = 0
    best_miou = 0
    train = Train(model, train_loader, optimizer, criterion, metric, use_cuda=True)
    val = Test(model, val_loader, criterion, metric, use_cuda=True)
    n_epochs = 200
    for epoch in range(start_epoch, n_epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        lr_updater.step()
        epoch_loss, (iou, miou) = train.run_epoch(iteration_loss=True)

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
              format(epoch, epoch_loss, miou))

        if (epoch + 1) % 10 == 0 or epoch + 1 == n_epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, (iou, miou) = val.run_epoch(iteration_loss=True)

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                  format(epoch, loss, miou))

            # Print per class IoU on last epoch or if best iou
            if epoch + 1 == n_epochs or miou > best_miou:
                for class_iou in iou:
                    print(class_iou)

            # Save the model if it's the best thus far
            if miou > best_miou:
                print("\nBest model thus far. Saving...\n")
                best_miou = miou
                torch.save(model.state_dict(), '/mnt/disks/data/d4dl/snapshots/snapshot_' + str(epoch) + '.pt')
    return model


def test():
    pass


def main():
    n_classes = 2  # Road, other
    train_loader, val_loader, class_weights = load_dataset()
    train(train_loader, val_loader, class_weights)
    # test()..


if __name__ == '__main__':
    main()
