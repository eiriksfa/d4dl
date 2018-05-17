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
			for filename_ori in os.listdir(os.path.join(self.root_dir_img,cityfolder)):
				#print(filename_ori)
				filename_general = filename_ori.replace("leftImg8bit.png","")
				tmp.append([filename_general,cityfolder])

		self.idx_mapping = tmp

	def __len__(self):
		return len(self.idx_mapping)

	def __getitem__(self, idx):
		# idx is translated to city folder and

		#variable for syntax shortening
		rt_im = self.root_dir_img
		rt_gt = self.root_dir_gt
		fn = self.idx_mapping[idx][0] #filename
		cf = self.idx_mapping[idx][1] #city folder
		gtt = self.gt_type

		#complete path for each file
		img_real_fn = os.path.join( rt_im, cf, fn + "leftImg8bit.png")
		img_color_fn = os.path.join( rt_gt, cf, fn + gtt + "_color.png")
		img_polygon_fn = os.path.join( rt_gt, cf, fn + gtt + "_polygons.json")

		#read the file
		img_real = io.imread(img_real_fn)
		img_color = io.imread(img_color_fn)
		with open(img_polygon_fn) as f:
			img_polygon = json.load(f)
		f.close()

		#creating sample tuple
		sample = {
			'image' : img_real,
			'gt_color' : img_color,
			'gt_polygon' : img_polygon
		}

		#transform the sample (if any)
		if self.transform:
			sample = self.transform(sample)

		return sample

class ToTensor(object):
	"""Convert ndarrays in sample into Tensors"""
	def __call__(self, sample):
		image = sample['image'] 
		gt_color = sample['gt_color']
		gt_polygon = sample['gt_polygon']

		
		return{
			'image' : torch.from_numpy(image),
			'gt_color' : torch.from_numpy(gt_color),
			'gt_polygon' : gt_polygon
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
			if label=='road':
				polygon = getattr(item, 'objects')['polygon']
				tmp = []
				for i in polygon:
					tmp.append((i[0] , i[1]))
				polygon_road.append(tmp)

		poly = Image.new('RGB',(w,h), (0,0,0))
		pdraw = ImageDraw.Draw(poly)
		for pl in polygon_road:
			pdraw.polygon(pl, fill=(255,0,0))

		poly2 = np.array(poly)

		return{
			'image' : image,
			'gt_color' : poly2,
			'gt_polygon' : gt_polygon
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
        		'gt_color' : gt_col,
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
        		'gt_color' : gt_col,
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
        		'gt_color' : gt_col,
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
        		'gt_color' : gt_color,
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

        img = exposure.rescale_intensity(image,(self.val[0],self.val[1]))

        return {'image': img,
        		'gt_color' : gt_color,
        		'gt_polygon': gt_polygon}
#------------------------------------------------------------

compose_tf = transforms.Compose([
								OnlyRoads(),
								Rescale(100),
								Rotate(0),
								FlipLR(),
								Blur(1),
								ContrastSet((0,3)),
								ToTensor()
								])

city_dataset = CityScapeDataset( root_dir_img='../../../data/cityscape/leftImg8bit/train',
								 root_dir_gt='../../../data/cityscape/gtFine/train',
								 gt_type='gtFine', transform=compose_tf
								)
print(len(city_dataset))

for i in range(len(city_dataset)):
	sample = city_dataset[i]
	print(i, sample['image'].shape, 
		sample['gt_color'].shape) 
	plt.imshow(sample['image'])
	plt.pause(1)
	plt.imshow(sample['gt_color'])


	plt.pause(1)
	

train_loader = torch.utils.data.DataLoader(city_dataset, 
                                            batch_size=64, shuffle=True,
                                            num_workers=4, pin_memory=True)
print(len(train_loader))



def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(100)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(train_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(output)
