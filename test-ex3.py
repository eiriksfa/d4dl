from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform, filters, exposure
from skimage.util import random_noise
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import Image
import ImageDraw

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class CityScapeDataset(Dataset):
    """CityScape dataset"""

    def __init__(self, root_dir_img, root_dir_gt, gt_type, transform=None):
        """
        Args :
            roto_dir_img (string) : Directory to real images
            root_dir_gt (string) : Directory to ground truth data of the images
            gt_type (String) : Either "gtCoarse" or "gtFine"
            transform (callable, optoonal) : Optional transform to be applied on a sample
        """
        self.root_dir_img = root_dir_img
        self.root_dir_gt = root_dir_gt
        self.transform = transform
        self.gt_type = gt_type

        tmp = []
        for cityfolder in os.listdir(self.root_dir_img):
            for filename_ori in os.listdir(os.path.join(self.root_dir_img, cityfolder)):
                # print(filename_ori)
                filename_general = filename_ori.replace("leftImg8bit.png", "")
                tmp.append([filename_general, cityfolder])

        self.idx_mapping = tmp

    def __len__(self):
        return len(self.idx_mapping)

    def __getitem__(self, idx):
        # idx is translated to city folder and

        # variable for syntax shortening
        rt_im = self.root_dir_img
        rt_gt = self.root_dir_gt
        fn = self.idx_mapping[idx][0]  # filename
        cf = self.idx_mapping[idx][1]  # city folder
        gtt = self.gt_type

        # complete path for each file
        img_real_fn = os.path.join(rt_im, cf, fn + "leftImg8bit.png")
        img_color_fn = os.path.join(rt_gt, cf, fn + gtt + "_color.png")
        img_polygon_fn = os.path.join(rt_gt, cf, fn + gtt + "_polygons.json")

        # read the file
        img_real = io.imread(img_real_fn)
        img_color = io.imread(img_color_fn)
        with open(img_polygon_fn) as f:
            img_polygon = json.load(f)
        f.close()

        # creating sample tuple
        sample = {
            'image': img_real,
            'gt_color': img_color,
            'gt_polygon': img_polygon
        }

        # transform the sample (if any)
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample into Tensors"""

    def __call__(self, sample):
        image = sample['image']
        gt_color = sample['gt_color']
        gt_polygon = sample['gt_polygon']

        return {
            'image': torch.from_numpy(image),
            'gt_color': torch.from_numpy(gt_color),
            'gt_polygon': gt_polygon
        }


class OnlyRoads(object):
    """	Recreate ground truth only for road class and non-road class."""

    def __call__(self, sample):
        image = sample['image']
        gt_color = sample['gt_color']
        gt_polygon = pd.DataFrame(sample['gt_polygon'])

        h, w = gt_polygon['imgHeight'][0], gt_polygon['imgWidth'][0]
        polygon_road = []
        for item in gt_polygon.itertuples(index=True):
            label = getattr(item, 'objects')['label']
            if label == 'road':
                polygon = getattr(item, 'objects')['polygon']
                tmp = []
                for i in polygon:
                    tmp.append((i[0], i[1]))
                polygon_road.append(tmp)

        poly = Image.new('RGB', (w, h), (0, 0, 0))
        pdraw = ImageDraw.Draw(poly)
        for pl in polygon_road:
            pdraw.polygon(pl, fill=(255, 0, 0))

        poly2 = np.array(poly)

        return {
            'image': image,
            'gt_color': poly2,
            'gt_polygon': sample['gt_polygon']
        }


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        gt_color = sample['gt_color']
        gt_polygon = sample['gt_polygon']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), order=0)
        gt_col = transform.resize(gt_color, (new_h, new_w), order=0)

        return {'image': img,
                'gt_color': gt_col,
                'gt_polygon': gt_polygon}


class Rotate(object):
    """Rotate an image to the desired angle.

    Args:
        rotate_val (int): Desired rotation value, in degree.
    """

    def __init__(self, rotate_val):
        assert isinstance(rotate_val, (int))
        self.rotate_val = rotate_val

    def __call__(self, sample):
        image = sample['image']
        gt_color = sample['gt_color']
        gt_polygon = sample['gt_polygon']

        img = transform.rotate(image, self.rotate_val, resize=True, order=0)
        gt_col = transform.rotate(gt_color, self.rotate_val, resize=True, order=0)

        return {'image': img,
                'gt_color': gt_col,
                'gt_polygon': gt_polygon}


class FlipLR(object):
    """Flip the image left to right"""

    def __call__(self, sample):
        image = sample['image']
        gt_color = sample['gt_color']
        gt_polygon = sample['gt_polygon']

        img = np.fliplr(image).copy()
        gt_col = np.fliplr(gt_color).copy()

        return {'image': img,
                'gt_color': gt_col,
                'gt_polygon': gt_polygon}


class Blur(object):
    """Blur an image, simulation of rainy or foggy weather.

    Args:
        blur_val (int): Desired blur value.
    """

    def __init__(self, blur_val):
        assert isinstance(blur_val, (int))
        self.blur_val = blur_val

    def __call__(self, sample):
        image = sample['image']
        gt_color = sample['gt_color']
        gt_polygon = sample['gt_polygon']

        img = filters.gaussian(image, sigma=self.blur_val)

        return {'image': img,
                'gt_color': gt_color,
                'gt_polygon': gt_polygon}


class ContrastSet(object):
    """Change a contrast of an image, simulation of very light/dark condition.

    Args:
        val (tuple): Desired stretch range of the distribution.
    """

    def __init__(self, val):
        assert isinstance(val, (tuple))
        self.val = val

    def __call__(self, sample):
        image = sample['image']
        gt_color = sample['gt_color']
        gt_polygon = sample['gt_polygon']

        img = exposure.rescale_intensity(image, (self.val[0], self.val[1]))

        return {'image': img,
                'gt_color': gt_color,
                'gt_polygon': gt_polygon}


# ------------------------------------------------------------

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
    def __init__(self, n_class=3):
        super(FCN32s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=100)
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
        self.fc6 = nn.Conv2d(256, 2048, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(2048, 2048, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(2048, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 32, stride=32, bias=False)

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

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

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

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore(h)
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return h


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
        Rescale(128),
        ToTensor()
    ])
    city_dataset = CityScapeDataset(root_dir_img='../../../data/cityscape-mini/leftImg8bit/train',
                                root_dir_gt='../../../data/cityscape-mini/gtFine/train',
                                gt_type='gtFine', transform=compose_tf
                                )
    print(len(city_dataset))

    device = 'cuda'
    net = FCN32s()
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    train_loader = torch.utils.data.DataLoader(city_dataset,
                                           batch_size=1, shuffle=True,
                                           num_workers=4, pin_memory=True)
    print(len(train_loader))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    accuracy = []
    for e in range(1, 30):
        train(net, optimizer, criterion, device, img, target.long())


