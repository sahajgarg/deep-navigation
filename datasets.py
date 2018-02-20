import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


## See if I need to define collating -- I don't think I should need to 
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

    def __len__(self):
        return len(self.frame_selections)

    def __getitem__(self, idx):
    	# Construct image path (except for frame)
        image_dir = self.images_dir + "{:02d}/{:s}/".format(frame_selections[0][idx], frame_selections[1][idx])
        images = []

    	# Load in all images in sequence 
        for i in range(frame_selections[2][idx], frame_selections[2][idx] + frame_selections[3][idx] + 1):
            images.append(io.imread(image_dir + "{:06d}.png".format(i)))

    	# Set poses as the corresponding list from the poses_dir 
        poses = []
        with open(self.poses_dir + "{:02d}.txt".format(frame_selections[0][idx])) as poses_file:
            for i, l in enumerate(poses_file):
                if i < frame_selections[2][idx]:
                    continue
                if i >= frame_selections[2][idx] + frame_selections[3][idx]:
                    break
                pose = [float(i) for i in l.split()]
                poses.append(pose)

    	# Return sample
        sample = {'images': np.array(images), 'gt_poses': np.array(poses)}
        return sample



    	"""
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        return sample
        """
