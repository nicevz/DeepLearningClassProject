import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


class RandomGenerator(object):

    def __init__(self, adj):
        self.adj = adj

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int8))
        if self.adj:
            pad_240_to_256 = torch.nn.ConstantPad2d((8, 8, 8, 8), 0)
            image = pad_240_to_256(image)
            label = pad_240_to_256(label)

        sample = {'image': image, 'label': label}
        return sample


class MSD_Dataset(Dataset):

    def __init__(self, base_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(
            os.path.join(base_dir,
                         'lists/' + self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + f"/{vol_name}.npy.h5"
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample
