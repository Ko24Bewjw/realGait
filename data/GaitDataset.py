# -*- coding: utf-8 -*-
# @Time : 13:13 2024/3/14
# @Author : Allen-wan
# @File : GaitDataset.py
# @Software : PyCharm
import os
import torch
import numpy as np
import torch.utils.data as tordata
from PIL import Image
from torchvision import transforms

import os
import cv2
import numpy as np
import torch
import torch.utils.data as tordata
from torchvision import transforms


class OptimizedGaitDataset(tordata.Dataset):
    """
    Optimized dataset class for gait recognition.
    """

    def __init__(self, seq_dir, labels, seq_types, views, resolution=64, frame_num=30, transform=None):
        # Dataset initialization
        self.seq_dir = seq_dir
        self.labels = labels
        self.seq_types = seq_types
        self.views = views
        self.resolution = resolution
        self.frame_num = frame_num
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resolution, resolution)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Creating index mapping for labels
        self.label_set = sorted(list(set(self.labels)))
        self.index_dict = {label: idx for idx, label in enumerate(self.label_set)}

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.seq_dir)

    def __getitem__(self, index):
        # Loading and returning a single sample from the dataset
        print(f"sele.seq_dir{self.seq_dir}")
        path = self.seq_dir[index]
        print(f"path==={path}")
        frames = self.load_frames(path, self.frame_num)

        # Apply transformations
        frames = torch.stack([self.transform(frame) for frame in frames])

        label_index = self.index_dict[self.labels[index]]
        return frames, label_index, self.seq_types[index], self.views[index]

    def load_frames(self, dir_path, num_frames):
        # Load specified number of frames from a directory
        if isinstance(dir_path, bytes):
            dir_path = dir_path.decode('utf-8')  # 或者使用你的文件系统的实际编码
        print(f"dir_path=={dir_path}")
        try:
            frame_files = sorted([f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg'))])
        except Exception as e:
            print(f"Error when listing directory {dir_path}: {e}")
            # Optionally, re-raise the error if you want the script to stop here

        selected_files = self.select_frames(frame_files, num_frames)
        frames = [cv2.imread(os.path.join(dir_path, file), cv2.IMREAD_GRAYSCALE) for file in selected_files]
        return [np.expand_dims(frame, axis=2) for frame in frames]  # Adding an extra dimension

    def select_frames(self, frame_files, num_frames):
        # Select a subset of frames
        total_frames = len(frame_files)
        stride = max(1, total_frames // num_frames)
        selected_frames = frame_files[::stride][:num_frames]
        return selected_frames

