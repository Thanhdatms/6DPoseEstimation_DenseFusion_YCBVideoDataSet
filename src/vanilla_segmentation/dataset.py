import os
import random
import copy
import numpy as np
import numpy.ma as ma
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
import torch
import torch.utils.data as data




class SegmentationDataSet(data.Dataset):
    def __init__(self, root_dir, txtlist, use_noise=False, transform=None):
        """
        Args:
            root_path: root path contain full data sorce
            txtlist: txt file contain file list
            use_noise: turn on augumentation
            transform: 
        """
        self.root_dir = root_dir
        self.txtlist = txtlist
        self.use_noise = use_noise
        self.transform = transform

        # read file path
        with open(txtlist, 'r') as f:
            self.path = [line.strip() for line in f if line.strip()]

        self.real_path = [p for p in self.path if p.startswith('data/')]
        self.len_data = len(self.path)
        self.back_len = len(self.real_path)

        

        # Augmentation
        self.color_jiter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

