from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import pickle
import torch
import torch.utils.data as data
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from copy import deepcopy
import random

from typing import Any, Callable, Optional, Tuple
from utils.conf import base_path_dataset as base_path
from utils.noiser import *
import deeplake

class ANIMAL(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    #base_folder ='
    #url = ""
    #filename = "cifar-10-python.tar.gz"
    #tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = ['data_batch_1.bin']
    
    test_list = ['test_batch.bin']
    


    def __init__(self, root, train=True,
                 transform=None, target_transform=None,               
                 noise_type=None,
                 noise_path=None, 
                 noise=None,
                 noise_rate=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.noise_path = noise_path
        self.noise_type = noise_type
        self.noise = noise
        self.noise_rate = noise_rate

        if download:
            os.environ['DEEPLAKE_DOWNLOAD_PATH']= self.root
            if not os.path.exists(os.path.join(self.root, "hub_activeloop_animal10n-train")):
                deeplake.dataset("hub://activeloop/animal10n-train", access_method='download')
            if not os.path.exists(os.path.join(self.root, "hub_activeloop_animal10n-test")):    
                deeplake.dataset("hub://activeloop/animal10n-test", access_method='download')           
        
        if self.train:
            if os.path.exists(os.path.join(self.root, "hub_activeloop_animal10n-train")):
                downloaded = deeplake.dataset(os.path.join(self.root, "hub_activeloop_animal10n-train"))
            else:
                downloaded = deeplake.load("hub://activeloop/animal10n-train")
            dataloader = downloaded.pytorch(num_workers=2, batch_size=32, shuffle=False)
    
        else:
            if os.path.exists(os.path.join(self.root, "hub_activeloop_animal10n-test")):
                downloaded = deeplake.dataset(os.path.join(self.root, "hub_activeloop_animal10n-test"))
            else:
                downloaded = deeplake.load("hub://activeloop/animal10n-test")
            dataloader = downloaded.pytorch(num_workers=2, batch_size=32, shuffle=False)

        #deeplake.dataset()
        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for data in dataloader:
            self.data.append(data['images'])
            self.targets.append(data['labels'])
            # also indexes available if necessary

        self.data = np.vstack(self.data).reshape(-1, 3, 64, 64)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.vstack(self.targets).reshape(-1)

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            gt_targets = self.target_transform(gt_targets)
            
        return img, target #, gt_targets
    
    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")