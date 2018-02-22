import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class KITTIDataset(Dataset):
    """KITTI dataset."""

    def __init__(self, frame_selections_file, images_dir, poses_dir):
        """
        Args:
			 frame_selections_file (string): Path to the csv file with rows: (
				sequence# (00-21), camera# (image_2 or image_3), start frame# (00000-max_frame-length), length)
			images_dir (string): Directory with all the images
			poses_dir (string): Directory with all the camera poses
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.frame_selections = pd.read_csv(frame_selections_file, header=None)
        self.images_dir = images_dir
        self.poses_dir = poses_dir
        print("initialized")

    def __len__(self):
        #print(len(self.frame_selections[0]))
        return len(self.frame_selections[0])

    def __getitem__(self, idx):
        print(idx)
        #print("getting item")
    	# Construct image path (except for frame)
        image_dir = self.images_dir + "{:02d}/{:s}/".format(self.frame_selections[0][idx], self.frame_selections[1][idx])
        images = []

    	# Load in all images in sequence 
        for i in range(self.frame_selections[2][idx], self.frame_selections[2][idx] + self.frame_selections[3][idx] + 1):
            images.append(io.imread(image_dir + "{:06d}.png".format(i)))

        # Set poses as the corresponding list from the poses_dir 
        poses = []
        with open(self.poses_dir + "{:02d}.txt".format(self.frame_selections[0][idx])) as poses_file:
            for i, l in enumerate(poses_file):
                if i < self.frame_selections[2][idx]:
                    continue
                if i >= self.frame_selections[2][idx] + self.frame_selections[3][idx]:
                    break
                pose = [float(i) for i in l.split()]
                poses.append(pose)

    	# Return sample
        sample = {'images': np.array(images), 'gt': np.array(poses)}
        print(sample['images'].shape, sample['gt'].shape)
        return sample

class RedDotDataset(Dataset):
    """RedDot dataset."""

    def __init__(self, base_dir):
        """
        Args:
            base_dir (string): Directory with all npy files for the images and the gt
        """
        self.base_dir = base_dir
        print("initialized")

    def __len__(self):
        l = int(len([name for name in os.listdir(self.base_dir) if os.path.isfile(self.base_dir + name)])/2)
        #print(l)
        return l

    def __getitem__(self, idx):
        #print("getting item")
        images = np.load(self.base_dir + "/{}_img.npy".format(idx))
        gt = np.load(self.base_dir + "/{}_pos.npy".format(idx))

        sample = {'images': images, 'gt': gt}
        #print(sample['images'].shape, sample['gt'].shape)
        return sample

