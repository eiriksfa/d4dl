from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import cv2
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
		img_instancelds_fn = os.path.join( rt_gt, cf, fn + gtt + "_instanceIds.png")
		img_labelids_fn = os.path.join( rt_gt, cf, fn + gtt + "_labelIds.png")
		img_polygon_fn = os.path.join( rt_gt, cf, fn + gtt + "_polygons.json")

		#read the file
		img_real = io.imread(img_real_fn)
		img_color = io.imread(img_color_fn)
		img_instancelds = io.imread(img_instancelds_fn)
		img_labelids = io.imread(img_labelids_fn)
		with open(img_polygon_fn) as f:
			img_polygon = json.load(f)
		f.close()

		#creating sample tuple
		sample = {
			'image' : img_real,
			'gt_color' : img_color,
			'gt_instancelds' : img_instancelds,
			'gt_label' : img_labelids,
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
		gt_instancelds = sample['gt_instancelds']
		gt_label = sample['gt_label']
		gt_polygon = sample['gt_polygon']

		image = image.transpose((2,0,1))
		return{
			'image' : torch.from_numpy(image),
			'gt_color' : torch.from_numpy(gt_color),
			'gt_instancelds' : torch.from_numpy(gt_instancelds),
			'gt_label' : torch.from_numpy(gt_label),
			'gt_polygon' : gt_polygon
		}

class OnlyRoads(object):
	def __call__(self, sample):
		image = sample['image'] 
		gt_color = sample['gt_color']
		gt_instancelds = sample['gt_instancelds']
		gt_label = sample['gt_label']
		gt_polygon = sample['gt_polygon']

		imgH = gt_polygon['imgHeight']
		imgW = gt_polygon['imgWidth']
		poly_json = gt_polygon['objects']['label'=='road']['polygon']
		poly_seq = []
		for i in poly_json:
			poly_seq.append((i[0] , i[1]))

		poly = Image.new('RGBA',(imgW,imgH), (0,0,0,255))
		pdraw = ImageDraw.Draw(poly)
		pdraw.polygon(poly_seq, fill=(255,0,0,255))

		poly2 = np.array(poly)

		return{
			'image' : image,
			'gt_color' : poly2,
			'gt_instancelds' : gt_instancelds,
			'gt_label' : gt_label,
			'gt_polygon' : gt_polygon
		}



		#TODO make a process to create new groundtruth with only road class


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
        gt_instancelds = sample['gt_instancelds']
        gt_label = sample['gt_label']
        gt_polygon = sample['gt_polygon']

        #print(gt_color.shape)
        #print(gt_color[1000])
        #with open('color-ori.txt','w') as file:
        #	file.write(gt_color)

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
        	if h > w:
        		new_h, new_w = self.output_size * h / w, self.output_size
        	else:
        		new_h, new_w = self.output_size, self.output_size * w / h
        else:
        	new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        gt_col = transform.resize(gt_color, (new_h, new_w))
        gt_instlds = transform.resize(gt_instancelds, (new_h, new_w))
        gt_lab = transform.resize(gt_label, (new_h, new_w))

        #print(gt_col.shape)
        #print(gt_col[1000])
        #with open('color-tf.txt','w') as file:
       # 	file.write(gt_col)

        return {'image': img,
        		'gt_color' : gt_col,
        		'gt_instancelds' : gt_instlds,
        		'gt_label' : gt_lab,
        		'gt_polygon': gt_polygon}

#------------------------------------------------------------

compose_tf = transforms.Compose([
								Rescale(200),
								OnlyRoads(),
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
		sample['gt_color'].shape, 
		sample['gt_instancelds'].shape, 
		sample['gt_label'].shape)
	plt.imshow(sample['gt_color'])
	#print(sample['image'])
	#print(sample['gt_color'])
	#print(sample['gt_polygon']['objects']['label'=='road']['polygon'])


	plt.pause(1)

	if i==10:
		break
	

train_loader = torch.utils.data.DataLoader(city_dataset, 
                                            batch_size=64, shuffle=True,
                                            num_workers=4, pin_memory=True)
print(len(train_loader))



#for i in range(len(city_dataset)):
#	sample = city_dataset[i]
#	print(i, sample['image'].shape, 
#		sample['gt_color'].shape, 
#		sample['gt_instancelds'].shape, 
#		sample['gt_label'].shape)
#	plt.imshow(sample['gt_color'])
#	plt.pause(0.1)
	#break
	#print(sample['image'])
	#ax = plt.subplot(1, 4, i + 1)
	#plt.tight_layout()
	#ax.set_title('Sample #{}'.format(i))
	#ax.axis('off')


