from typing import Tuple
import numpy as np

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from datasets.utils.cifar import CIFAR100
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.autoaugment import CIFAR10Policy
from utils.conf import base_path_dataset as base_path


class TCIFAR100(CIFAR100):
    """Workaround to avoid printing the already downloaded messages."""
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR100, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, noise_type=None, noise_path = None, noise = None,noise_rate=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, noise_type, noise_path, noise,noise_rate,not self._check_integrity())
        self.initial_indexes = np.arange(len(self.data))
        #super(MyCIFAR100, self).__init__(root, train, transform, target_transform, not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
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


class SequentialCIFAR100(ContinualDataset):

    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(MEAN, STD)])
    
    STRONG_TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN,STD)])
    
    def get_examples_number(self):
        train_dataset = MyCIFAR100(base_path() + 'CIFAR10', train=True,
                                  download=True)
        return len(train_dataset.data)

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True, transform=transform,
                                    noise_type=self.args.noise_type,
                                    noise_path=base_path() + 'CIFAR-N/CIFAR-100_human.pt',
                                    noise=self.args.noise,
                                    noise_rate=self.args.noise_rate,
                                    download=True)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TCIFAR100(base_path() + 'CIFAR100',train=False,
                                   download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR100.N_CLASSES_PER_TASK
                        * SequentialCIFAR100.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_epochs():
        return 50
    
    @staticmethod
    def get_warmup_epochs():
        return 15
    
    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR100.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler

