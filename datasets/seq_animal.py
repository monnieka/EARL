from typing import Tuple
import numpy as np

import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from utils.conf import base_path_dataset as base_path
from datasets.transforms.denormalization import DeNormalize
from torch.utils.data import Dataset
from datasets.utils.animal import ANIMAL
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val

class TANIMAL(ANIMAL):
    """Workaround to avoid printing the already downloaded messages."""
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TANIMAL, self).__init__(root, train, transform, target_transform, download=False)

class MyANIMAL(ANIMAL):
    """
    Overrides the ANIMAL dataset to change the getitem function.
    """
    
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, noise_type = None, noise_path = None,noise = None, noise_rate=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyANIMAL, self).__init__(root, train, transform, target_transform, noise_type, noise_path, noise, noise_rate,download=False)
        self.initial_indexes = np.arange(len(self.data))

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image, int, int]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target, gt_targets = self.data[index], self.targets[index], self.gt_targets[index]
        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            gt_targets = self.target_transform(gt_targets)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img, gt_targets, self.initial_indexes[index]


class SequentialANIMAL(ContinualDataset):

    NAME = 'seq-animal'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    MEAN,STD = (0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2615)
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(MEAN,STD)])
    
    STRONG_TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN,STD)])
    
    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyANIMAL(base_path() + 'ANIMAL', train=True,
                                transform=transform,
                                noise_type=self.args.noise_type,
                                noise_path=base_path() + 'CIFAR-N/CIFAR-10_human.pt',
                                noise = self.args.noise,
                                noise_rate = self.args.noise_rate,
                                download=True)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TANIMAL(base_path() + 'ANIMAL', train=False,
                                transform=test_transform,
                                download=True)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialANIMAL.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialANIMAL.N_CLASSES_PER_TASK
                        * SequentialANIMAL.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialANIMAL.MEAN,SequentialANIMAL.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialANIMAL.MEAN,SequentialANIMAL.STD)
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_warmup_epochs():
        return 0
        
    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialANIMAL.get_batch_size()