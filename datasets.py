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
        self.frame_selections = pd.read_csv(frame_selections_file)
        self.images_dir = images_dir
        self.poses_dir = poses_dir

    def __len__(self):
        return len(self.frame_selections)

    def __getitem__(self, idx):
    	# Construct image path (except for frame)
    	# Load in all images in sequence 
    	# Set poses as the corresponding list from the poses_dir 
    	# Return sample

    	"""
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        return sample
        """
        pass
