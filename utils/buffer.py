from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random

def icarl_replay(self, dataset, val_set_split=0):
    """
    Merge the replay buffer with the current task data.
    Optionally split the replay buffer into a validation set.

    :param self: the model instance
    :param dataset: the dataset
    :param val_set_split: the fraction of the replay buffer to be used as validation set
    """

    if self.task > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_train_mask = torch.zeros(len(dataset.train_loader.dataset.data)).bool()
        val_train_mask[torch.randperm(len(dataset.train_loader.dataset.data))[:buff_val_mask.sum()]] = True

        if val_set_split > 0:
            self.val_loader = deepcopy(dataset.train_loader)

        data_concatenate = torch.cat if isinstance(dataset.train_loader.dataset.data, torch.Tensor) else np.concatenate
        need_aug = hasattr(dataset.train_loader.dataset, 'not_aug_transform')
        if not need_aug:
            def refold_transform(x): return x.cpu()
        else:
            data_shape = len(dataset.train_loader.dataset.data[0].shape)
            if data_shape == 3:
                def refold_transform(x): return (x.cpu() * 255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)
            elif data_shape == 2:
                def refold_transform(x): return (x.cpu() * 255).squeeze(1).type(torch.uint8)

        # REDUCE AND MERGE TRAINING SET
        dataset.train_loader.dataset.targets = np.concatenate([
            dataset.train_loader.dataset.targets[~val_train_mask],
            self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
        ])
        dataset.train_loader.dataset.data = data_concatenate([
            dataset.train_loader.dataset.data[~val_train_mask],
            refold_transform((self.buffer.examples)[:len(self.buffer)][~buff_val_mask])
        ])

        if val_set_split > 0:
            # REDUCE AND MERGE VALIDATION SET
            self.val_loader.dataset.targets = np.concatenate([
                self.val_loader.dataset.targets[val_train_mask],
                self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
            ])
            self.val_loader.dataset.data = data_concatenate([
                self.val_loader.dataset.data[val_train_mask],
                refold_transform((self.buffer.examples)[:len(self.buffer)][buff_val_mask])
            ])


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir', rho=None):
        assert mode in ('ring', 'reservoir', 'prs')
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.current_task = 0
        self.mode = mode
        self.rho=rho
        self.abs=0

        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks

        self.attributes = ['examples', 'labels', 'logits','true_labels', 'task_labels', 'sample_indexes']
        # scores for abs
        self.importance_scores = torch.ones(self.buffer_size).to(self.device) * -float('inf')
        self.past_scores = torch.ones(self.buffer_size).to(self.device) * -float('inf')
        self.current_scores = torch.ones(self.buffer_size).to(self.device) * -float('inf')
        self.percentage = torch.ones(self.buffer_size).to(self.device) * -float('inf')

    def __getstate__(self):
        state = {k:i for k,i in self.__dict__.copy().items() if 'transform' not in k}
        return state

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self
    
    def is_full(self):
        return self.num_seen_examples >= self.buffer_size

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, true_labels: torch.Tensor, task_labels: torch.Tensor, sample_indexes: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith(('els','xes')) else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))
    
    def scale_scores(self):
   
        norm_importance = self.importance_scores
        
        self.past_indexes  = self.labels < self.current_task * self.dataset.N_CLASSES_PER_TASK
        
        past_importance = self.normalize_scores(norm_importance[self.past_indexes])
        current_importance = self.normalize_scores(norm_importance[~self.past_indexes])
        if past_importance is not None:
            past_importance = 1 - past_importance
            self.past_scores = past_importance / past_importance.sum()
        if current_importance is not None:
            self.current_scores = current_importance / current_importance.sum()

        self.past_percentage = self.past_indexes.sum()/self.buffer_size
        self.percentage[self.past_indexes] , self.percentage[~self.past_indexes] = self.past_percentage, 1 - self.past_percentage
        self.percentage = self.percentage/self.percentage.sum()
    
    def normalize_scores(self, values):
        if values.shape[0] > 0:
            if values.max() - values.min() != 0:
                values = (values - values.min()) / (values.max() - values.min())
            return values
        else:
            return None
        
    def update_scores(self, indexes, values):
        self.importance_scores[indexes.to(self.device)] = values.to(self.device)
 
    def functionalReservoir(self, N, m):
        if N < m:
            return N

        rn = np.random.randint(0, N)
        if rn < m:
            self.scale_scores()
            rp = np.random.choice((0,1), p=[self.past_percentage.cpu().numpy(), 1 -self.past_percentage.cpu().numpy()])
            if not rp:
                index = np.random.choice(np.arange(m)[self.past_indexes.cpu().numpy()], p=self.past_scores.cpu().numpy(), size=1)
            else:
                index = np.random.choice(np.arange(m)[~self.past_indexes.cpu().numpy()], p=self.current_scores.cpu().numpy(), size=1)
            return index
        else:
            return -1
        
    def update_proportions(self, label):
        label = label.detach().clone().to(self.device)
        if not hasattr(self, "unique_labels"):
            self.unique_labels, self.unique_counts = label, torch.tensor(1).to(self.device)
        else:
            if label in self.unique_labels:
                self.unique_counts[self.unique_labels == label] += 1
            else:
                self.unique_labels = torch.hstack((self.unique_labels, label.detach()))
                self.unique_counts = torch.hstack((self.unique_counts, torch.tensor(1).to(self.device)))
    
    def get_statistics(self):
        return self.unique_labels, self.unique_counts, self.unique_counts.pow(self.rho)/self.unique_counts.pow(self.rho).sum()
    
    def sample_in(self, clab):
        clab = clab.detach().clone().to(self.device)
        if clab not in self.unique_labels:
            return 1
        labels, counts, stats = self.get_statistics()
        mi = stats[labels==clab]*self.buffer_size
        ni = counts[labels==clab]
        w = torch.softmax(-counts.float(),dim=0)
        cq = mi/ni
        cq[mi>ni] = 1
        s = w*cq
        s = s.sum().cpu()
        return random.choices([1,-1],[s,1-s])
    
    def sample_out(self):
        """
        evict a sample from rsvr
        :returns: removed sample idx of rsvr
        """
        labs, _, stats = self.get_statistics()
        buf_labs, buf_counts = self.labels.unique(return_counts=True)

        deltas = torch.ones(len(labs)).float().to(self.device)
        buf_counts_order = torch.argsort(labs[torch.isin(labs, buf_labs)])
        deltas[torch.isin(labs, buf_labs)] = buf_counts[buf_counts_order] - stats[torch.isin(labs, buf_labs)]*len(self)

        probs = torch.softmax(deltas,dim=0)

        selected_key = random.choices(labs, weights=probs, k=1)[0]

        idxs = torch.where(self.labels == selected_key)[0]

        z = np.random.choice(idxs.cpu().numpy(), size=1)
        return z

    def add_data(self, examples, labels=None, logits=None, true_labels=None, task_labels=None, loss_scores=None, sample_indexes =None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, true_labels, task_labels, sample_indexes)

        for i in range(examples.shape[0]):
            if self.mode == 'reservoir':
                if loss_scores is not None and self.abs:
                    index = self.functionalReservoir(self.num_seen_examples, self.buffer_size)
                else:
                    index = reservoir(self.num_seen_examples, self.buffer_size)

            self.num_seen_examples += 1
            if index >= 0:
                if self.examples.device != self.device:
                    self.examples.to(self.device)
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    if self.labels.device != self.device:
                        self.labels.to(self.device)
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    if self.logits.device != self.device:
                        self.logits.to(self.device)
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    if self.task_labels.device != self.device:
                        self.task_labels.to(self.device)
                    self.task_labels[index] = task_labels[i].to(self.device)
                if true_labels is not None:
                    if self.true_labels.device != self.device:
                        self.true_labels.to(self.device)
                    self.true_labels[index] = true_labels[i].to(self.device)
                if sample_indexes is not None:
                    if self.sample_indexes.device != self.device:
                        self.sample_indexes.to(self.device)
                    self.sample_indexes[index] = sample_indexes[i].to(self.device).to(self.sample_indexes.dtype)
                self.importance_scores[index] = -float('inf') if loss_scores is None else loss_scores[i]

    def get_dataloader(self, batch_size, shuffle=False, drop_last=False, transform=None, sampler=None):
        self.dl_transform = transform
        self.dl_index = 0
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle if sampler is None else False, drop_last=drop_last, sampler=sampler)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.dl_index >= self.__len__():
            raise StopIteration
        return self.__getitem__(self.dl_index, transform=self.dl_transform)

    def __getitem__(self, index, transform=None):
        data = self._get_item_batched(torch.tensor([index]), transform=transform if self.dl_transform is None or transform is not None else self.dl_transform)

        return [d.squeeze(0) for d in data]

    @torch.no_grad()
    def _do_transform(self, data, transform):
        if transform is None:
            return data
        #if is kornia
        if isinstance(transform, nn.Module):
            data = transform(data)
            if len(data.shape) == 4 and data.shape[0]==1:
                data = data.squeeze(0)
            return data
        else:
            return torch.stack([transform(ee.cpu()) for ee in data])

    def _get_item_batched(self, indexes, transform=None, return_not_aug=False, device=None):
        device = self.device if device is None else device
        if transform is not None:
            trs = self.dl_transform if transform is None else transform
            ret_tuple = (self._do_transform(self.examples[indexes], trs).to(device),)
            if isinstance(trs, transforms.Compose) and return_not_aug:
                non_aug_tuple = (torch.stack([transform.transforms[1].transforms[-1](ee.cpu()) #selects the Nomalize transform
                                    for ee in self.examples[indexes]]).to(device),)
                ret_tuple = non_aug_tuple + ret_tuple
        else:
            ret_tuple = (self.examples[indexes.to(self.examples.device)].to(device),)

        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[indexes].to(device),)

        return ret_tuple

    
    def get_data(self, size: int, transform: nn.Module = None, return_index=False, return_not_aug=False, device=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        
        ret_tuple = self._get_item_batched(torch.from_numpy(choice), transform=transform, return_not_aug=return_not_aug, device=device)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple

    def get_data_by_index(self, indexes, transform: nn.Module = None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: nn.Module = None, device=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        device = self.device if device is None else device
        if transform is None:
            ret_tuple = (self.examples.to(device),)
        else:
            ret_tuple = (torch.stack([transform(ee.cpu())
                                    for ee in self.examples]).to(device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(device)
                ret_tuple += (attr,)
        return ret_tuple
    
    @torch.no_grad()
    def get_buffer_stats(self):
        """
        Returns the number of seen examples and the buffer size.
        """
        past_labels  = self.labels < self.current_task * self.dataset.N_CLASSES_PER_TASK
        current_task_clean = (self.labels[~past_labels] == self.true_labels[~past_labels]).sum().item()
        current_task_noisy = (self.labels[~past_labels] != self.true_labels[~past_labels]).sum().item()
        
        past_task_clean = (self.labels[past_labels] == self.true_labels[past_labels]).sum().item()
        past_task_noisy = (self.labels[past_labels] != self.true_labels[past_labels]).sum().item()
        
        return current_task_clean, current_task_noisy, past_task_clean, past_task_noisy
    
    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

    def log_cleaning_rate(self, transform: nn.Module = None):
        buf_i, buf_l , buf_tl= self.get_all_data(transform=transform)
        
        clean_set, noisy_set = buf_i[buf_l==buf_tl], buf_i[buf_l != buf_tl]
        clean_tot, noisy_tot = clean_set.shape[0], noisy_set.shape[0]

        assert( clean_tot + noisy_tot == self.buffer_size )
        
        cleaning_accuracy = (clean_tot / self.buffer_size) *100

        return cleaning_accuracy
