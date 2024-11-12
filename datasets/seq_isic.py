import timm

from backbone.vit import vit_base_patch16_224
assert timm.__version__ == '0.9.8', 'Please install timm version 0.9.8 (don\'t trust timm\'s defaults)'
from copy import deepcopy
import os
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils.autoaugment import ImageNetPolicy

from utils.conf import base_path
from PIL import Image
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from torch.utils.data import Dataset
import torch.nn as nn
import pickle
from torchvision.transforms.functional import InterpolationMode

from utils.noiser import add_symmetric_noise

ISIC_MEAN = [0.485, 0.456, 0.406]
ISIC_STD = [0.229, 0.224, 0.225]


class Isic(Dataset):
    N_CLASSES = 6

    """
    Overrides the ChestX dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, noise=None, noise_rate=None, download=False) -> None:

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.noise = noise
        self.noise_rate = noise_rate

        if download:
            if not os.path.exists(os.path.join(self.root, 'DONE')):
                from onedrivedownloader import download
                print('Downloading dataset...')
                ln = 'downloadpath'
                download(ln, filename=os.path.join(root, 'isic.tar.gz'), unzip=True, unzip_path=root, clean=True)
                os.system(f'mv {root}/isic/* {root}/')
                os.system(f'rmdir {root}/isic')
                # touch DONE
                open(os.path.join(self.root, 'DONE'), 'a').close()

        if train:
            filename_labels = f'{self.root}/train_labels.pkl'
            filename_images = f'{self.root}/train_images.pkl'
        else:
            filename_labels = f'{self.root}/test_labels.pkl'
            filename_images = f'{self.root}/test_images.pkl'

        self.not_aug_transform = transforms.Compose([
        ])

        with open(filename_images, 'rb') as f:
            self.data = pickle.load(f)

        with open(filename_labels, 'rb') as f:
            self.targets = pickle.load(f)

        if self.train:
            if noise_rate is not None:
                self.noisy_targets = deepcopy(self.targets)
                if self.noise == 'sym':
                    add_symmetric_noise(list(np.unique(np.array(self.targets))), self)
                else:
                    raise NotImplementedError

        self.initial_indexes = np.arange(len(self.data))

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

        original_img = transforms.ToTensor()(img.copy())

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            if self.train:
                gt_targets = self.target_transform(gt_targets)

        if not self.train:
            return img, target

        if self.train:
            if hasattr(self, 'logits'):
                return img, target, not_aug_img, self.logits[index], gt_targets, self.initial_indexes[index]

            return img, target, not_aug_img, gt_targets, self.initial_indexes[index]
        else:
            return img, target


class SequentialIsic(ContinualDataset):

    NAME = 'seq-isic'
    SETTING = 'class-il'
    N_TASKS = 3
    N_CLASSES_PER_TASK = 2
    N_CLASSES = 6
    normalize = transforms.Normalize(mean=ISIC_MEAN, std=ISIC_STD)

    TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        normalize,
    ])

    STRONG_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        ImageNetPolicy(),
        transforms.ToTensor(),
        normalize,
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.label_to_class_name = self.get_class_names()

    def get_data_loaders(self):
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize]
        )

        train_dataset = Isic(base_path() + 'isic', train=True,
                             noise=self.args.noise,
                             noise_rate=self.args.noise_rate,
                             download=True, transform=self.TRANSFORM)

        test_dataset = Isic(base_path() + 'isic', train=False, download=True,
                            transform=test_transform)

        class_order = None
        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    def get_class_names(self):
        return [
            'melanoma',
            'basal cell carcinoma',
            'actinic keratosis or intraepithelial carcinoma',
            'benign keratosis',
            'dermatofibroma',
            'vascular skin lesion',
        ]

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             SequentialIsic.TRANSFORM]
        )
        return transform

    @staticmethod
    def get_backbone(hookme=False):
        num_classes = SequentialIsic.N_CLASSES_PER_TASK * SequentialIsic.N_TASKS
        backbone = vit_base_patch16_224(pretrained=True, num_classes=num_classes)
        return backbone

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=ISIC_MEAN, std=ISIC_STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(mean=ISIC_MEAN, std=ISIC_STD)
        return transform

    def get_epochs(self):
        base_epochs = 30
        if self.args.model == 'star_prompt':
            return base_epochs + self.get_n_epochs_first_stage()
        return base_epochs

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_virtual_bn_num():
        return 1

    @staticmethod
    def get_n_epochs_first_stage():
        return 50

    @staticmethod
    def get_warmup_epochs():
        return 10


if __name__ == '__main__':
    d = Isic('../data/isic', train=False)
    d[0]