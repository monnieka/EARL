import copy
import timm

from backbone.vit import vit_base_patch16_224
assert timm.__version__ == '0.9.8', 'Please install timm version 0.9.8 (don\'t trust timm\'s defaults)'
from copy import deepcopy
import os
import sys
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn.functional as F
import numpy as np
from utils.autoaugment import ImageNetPolicy
from utils.conf import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from torch.utils.data import Dataset
import torch.nn as nn
import yaml
import pickle
from torchvision.transforms.functional import InterpolationMode
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
import scipy.io
from glob import glob
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
import requests
import zipfile
import io
import google_drive_downloader as gdd
import json
import pandas as pd

from utils.noiser import add_symmetric_noise
#from torchdata.datapipes.iter import FileLister, FileOpener

def load_noisy_targets(dataset, download=True):
    
    if dataset.train:
        filepath = os.path.join(base_path(), dataset.root.split('/')[-1], dataset.noise)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fname, url = 'preds_eurosat.npy','downloadpath'

        if not os.path.exists(os.path.join(filepath, fname)):
            if download:
                print('Downloading noisy labels...')
                from onedrivedownloader import download
                download(url, filename=os.path.join(filepath, fname))
            else:
                raise RuntimeError('file not found. You can use download=True to download it')
        assert os.path.exists(os.path.join(filepath, fname))
        dataset.noisy_targets = np.load(os.path.join(filepath, fname))['arr_0'].tolist()

class MyEuroSat(Dataset): 

    def __init__(self, root, split='train', transform=None, noise=None, noise_rate=None,
                 target_transform=None) -> None:

        self.root = root
        self.split = split
        assert split in ['train', 'test', 'val'], 'Split must be either train, test or val'
        self.train = split == 'train'
        self.transform = transform
        self.target_transform = target_transform
        self.totensor = transforms.Compose([transforms.Resize(224, interpolation=3),
                                            transforms.CenterCrop(224),
                                           transforms.ToTensor()])
        self.noise = noise
        self.noise_rate = noise_rate

        if not os.path.exists(root):
            print('Preparing dataset...', file=sys.stderr)
            # downlaod zip from https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1
            # and extract to ../data/eurosat
            r = requests.get('https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1')
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(root)
            # move base_path() + 'EuroSAT_RGB/*' to base_path() + 'eurosat'
            os.system(f'mv {root}/EuroSAT_RGB/* {root}')
            os.system(f'rmdir {root}/EuroSAT_RGB')

            # create DONE file
            with open(self.root + '/DONE', 'w') as f:
                f.write('')
        
            gdd.GoogleDriveDownloader.download_file_from_google_drive(file_id='1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o',
                                                                      dest_path=self.root + '/split.json')

        self.class_names = self.get_class_names()

        self.data = self.data_split[0].values
        self.targets = self.data_split[1].values

        if self.train:
            if noise_rate is not None:
                self.noisy_targets = deepcopy(self.targets)
                if self.noise == 'sym':
                    add_symmetric_noise(list(np.unique(np.array(self.targets))), self)
                else:
                    raise NotImplementedError
            else:
                if noise == 'feature':
                    load_noisy_targets(dataset=self)

        self.initial_indexes = np.arange(len(self.data))

    @staticmethod
    def get_class_names():
        if not os.path.exists(base_path() + f'eurosat/DONE'):
            gdd.GoogleDriveDownloader.download_file_from_google_drive(file_id='1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o',
                                                                      dest_path=base_path() + 'eurosat/split.json')
        return pd.DataFrame(json.load(open(base_path() + 'eurosat/split.json', 'r'))['train'])[2].unique()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target, gt_targets = self.data[index], self.targets[index], self.gt_targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        #img = Image.fromarray(img)
        img = Image.open(self.root + '/' + img).convert('RGB')

        not_aug_img = self.totensor(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            if self.train:
                gt_targets = self.target_transform(gt_targets)

        if self.train:
            if hasattr(self, 'logits'):
                return img, target, not_aug_img, self.logits[index], gt_targets, self.initial_indexes[index]

            return img, target, not_aug_img, gt_targets, self.initial_indexes[index]
        else:
            return img, target


def my_collate_fn(batch):
    tmp = list(zip(*batch))
    imgs = torch.stack(tmp[0], dim=0)
    labels = torch.tensor(tmp[1])
    if len(tmp) == 2:
        return imgs, labels
    else:
        not_aug_imgs = tmp[2]
        not_aug_imgs = torch.stack(not_aug_imgs, dim=0)
        return imgs, labels, not_aug_imgs


class SequentialEuroSatRgb(ContinualDataset):

    NAME = 'seq-eurosat-rgb'
    SETTING = 'class-il'
    N_TASKS = 5
    N_CLASSES = 10
    N_CLASSES_PER_TASK = 2
    MEAN, STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    normalize = transforms.Normalize(mean=MEAN, std=STD)

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),  # from https://github.dev/KaiyangZhou/Dassl.pytorch defaults
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(224, interpolation=3),  # bicubic
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    STRONG_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),  # from https://github.dev/KaiyangZhou/Dassl.pytorch defaults
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        normalize,
    ])

    def get_class_names(self):
        try:
            classes = MyEuroSat.get_class_names()
        except BaseException:
            print("WARNING: dataset not loaded yet -- loading dataset...")
            MyEuroSat(base_path() + 'eurosat', train=True,
                                    transform=None)
            classes = MyEuroSat.get_class_names()
        if self.class_order is not None:
            classes = [classes[i] for i in self.class_order]
        return classes

    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def get_data_loaders(self):
        train_transform = self.TRANSFORM
        test_transform = transforms.Compose([
            transforms.Resize(224, interpolation=3),  # bicubic
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])

        train_dataset = MyEuroSat(base_path() + 'eurosat', split='train',
                                  noise=self.args.noise,
                                  noise_rate=self.args.noise_rate,
                                  transform=train_transform)
        if self.args.validation:
            test_dataset = MyEuroSat(base_path() + 'eurosat', split='val',
                                     transform=test_transform)
        else:
            test_dataset = MyEuroSat(base_path() + 'eurosat', split='test',
                                     transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialEuroSatRgb.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(hookme=False):
        num_classes = SequentialEuroSatRgb.N_CLASSES_PER_TASK * SequentialEuroSatRgb.N_TASKS
        backbone = vit_base_patch16_224(pretrained=True, num_classes=num_classes)
        return backbone

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return SequentialEuroSatRgb.normalize

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialEuroSatRgb.MEAN, SequentialEuroSatRgb.STD)
        return transform

    @staticmethod
    def get_epochs():
        return 5

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_virtual_bn_num():
        return 1

    @staticmethod
    def get_warmup_epochs():
        return 10

    @staticmethod
    def get_scheduler(model, args):
        return None


if __name__ == '__main__':
    d = MyEuroSat('../data/eurosat', train=True)
    d[0]
    pass