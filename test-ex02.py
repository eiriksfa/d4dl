from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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
			filename_ori = os.listdir(os.path.join(self.root_dir_img,cityfolder))
			filename_general = filename_ori.replace("leftImg8bit.png","")
			tmp.append(filename_general)

		self.idx_mapping = tmp

	def __len__(self):
		return len(self.idx_mapping)

	def __getitem__(self, idx):
		# idx is translated to city folder and

		#variable for syntax shortening
		rt_im = self.root_dir_img
		rt_gt = self.root_dir_gt
		fn = self.idx_mapping[idx]
		gtt = self.gt_type

		#complete path for each file
		img_real_fn = os.path.join( rt_im, fn , "leftImg8bit.png")
		img_color_fn = os.path.join( rt_gt, fn, gtt, "_color.png")
		img_instancelds_fn = os.path.join( rt_gt, fn, gtt, "_instanceIds.png")
		img_labelids_fn = os.path.join( rt_gt, fn, gtt, "_labelIds.png")
		img_polygon_fn = os.path.join( rt_gt, fn, gtt, "_polygons.json")

		#read the file
		img_real = io.imread(img_real_fn)
		img_color = io.imread(img_color_fn)
		img_instancelds = io.imread(img_instancelds_fn)
		img_labelids = io.imread(img_labelids_fn)
		with open(img_polygon_fn) as f:
			img_polygon = json.load(f)

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

