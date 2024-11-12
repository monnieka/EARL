
import torch
from datasets import get_dataset

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, add_noise_args, ArgumentParser
from utils.buffer import Buffer
from utils.no_bn import bn_track_stats
 
import tqdm
from utils.augmentations import cutmix_data
import numpy as np

def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_noise_args(parser)

    parser.add_argument('--enable_amnesia', default=1, type=int, help='flag to activate alternated replay')
    parser.add_argument('--bifold_sampling', default=1, type=int, choices=[0,1],help='flag for bifold_sampling sampling active or not')

    return parser

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, targets=None, true_targets=None, transform=None, device="cpu"):
        self.data = data.to(device)
        self.targets = targets.to(device) if targets is not None else None
        self.true_targets = targets.to(device) if targets is not None else None

        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if self.transform:
            data = self.transform(self.data[idx])
        else:
            data = self.data[idx]
        if self.targets is not None:
            return data, self.targets[idx], self.true_targets[idx]
        return data
    
class Earl(ContinualModel):
    NAME = 'earl'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Earl, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0
        self.buffer.dataset = get_dataset(args)
        self.buffer.bifold_sampling = self.args.bifold_sampling
        self.iteration = 0

    def end_task(self, dataset):
        self.task += 1
        self.buffer.current_task = self.task
        if self.args.buffer_fitting_epochs > 0:
            self.fit_buffer()
            
    def fit_buffer(self):
        
        inputs, labels, true_labels, _ = self.buffer.get_all_data()
        
        transform=self.transform

        train_dataset = Dataset(inputs, targets=labels, true_targets=true_labels, transform=transform, device=self.device)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
    
        opt = torch.optim.SGD(self.net.parameters(), lr=self.args.buffer_fitting_lr, momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2, eta_min=self.args.buffer_fitting_lr * 0.01
        )
        
        for e in tqdm.trange(self.args.buffer_fitting_epochs, desc ='Fitting on buffer'):
            for i, data in enumerate(train_loader):    
                inputs, labels, true_labels = data
                inputs,labels,true_labels = inputs.to(self.device), labels.to(self.device), true_labels.to(self.device)
                opt.zero_grad()
                if self.args.enable_cutmix and np.random.rand(0,1) < 0.5:  
                    inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels)

                    logits = self.net(inputs)

                    loss = lam * self.loss(logits, labels_a) + (1 - lam) * self.loss(logits, labels_b)
                else:
                    logits = self.net(inputs)

                    loss_ext = self.loss(logits, labels, reduction ='none')

                loss = loss_ext.mean()
                loss.backward()
                opt.step()

                self.iteration+=1
            scheduler.step()
                
    def observe(self, inputs, labels, not_aug_inputs, true_labels, indexes, epoch):
        
        labels = labels.long()
        true_labels = true_labels.long()

        present = labels.unique()

        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()
        
        logits = self.net(inputs) 
        with torch.no_grad():
            not_aug_logits = self.net(get_dataset(self.args).TRANSFORM.transforms[-1](not_aug_inputs))
        
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
        not_aug_logits = not_aug_logits.masked_fill(mask == 0, torch.finfo(not_aug_logits.dtype).min)
        
        loss_ext = self.loss(logits,labels, reduction='none')
        loss_not_aug_ext = self.loss(not_aug_logits,labels, reduction='none')
            
        loss = loss_ext.mean()

        loss_re_ext = torch.tensor(0.)
        loss_re = torch.tensor(0.)

        if not self.buffer.is_empty():
            # sample from buffer
            buf_indexes, not_aug_buf_inputs, buf_inputs, buf_labels, buf_true_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True, return_not_aug=True)

            buf_logits = self.net(buf_inputs)
            with torch.no_grad():
                not_aug_buf_logits = self.net(not_aug_buf_inputs)

            loss_re_ext = self.loss(buf_logits, buf_labels, reduction='none')
            with torch.no_grad():
                loss_not_aug_re_ext = self.loss(not_aug_buf_logits, buf_labels, reduction='none')
            loss_re =loss_re_ext.mean()

            if self.args.enable_amnesia and epoch % 2 == self.args.start_with_replay:
                loss_re= torch.tensor(0.)
                    
        loss += loss_re

        loss.backward()
                    
        self.opt.step()

        clean_mask = torch.ones_like(labels, dtype=torch.bool)
        _, clean_mask = torch.topk(loss_not_aug_ext, round((1-0.75)*inputs.shape[0]), largest=False)

        if self.args.bifold_sampling:
            loss_ext = loss_not_aug_ext
            if not self.buffer.is_empty():
                loss_re_ext = loss_not_aug_re_ext

        if not self.buffer.is_empty():
            self.buffer.update_scores(buf_indexes, loss_re_ext.detach().to(self.device))        
                  
        if self.args.enable_amnesia and epoch % 2 == self.args.start_with_replay:
            if self.args.bifold_sampling:
                self.buffer.add_data(examples=not_aug_inputs[clean_mask],
                                    labels=labels[clean_mask],
                                    true_labels=true_labels[clean_mask],
                                    loss_scores=loss_ext[clean_mask].detach(),
                                    sample_indexes=indexes[clean_mask])
            else:
                self.buffer.add_data(examples=not_aug_inputs[clean_mask],
                    labels=labels[clean_mask],
                    true_labels=true_labels[clean_mask],
                    sample_indexes=indexes[clean_mask])
        elif not self.args.enable_amnesia:
            self.buffer.add_data(examples=not_aug_inputs[clean_mask],
                                labels=labels[clean_mask],
                                true_labels=true_labels[clean_mask],
                                loss_scores=loss_ext[clean_mask].detach(),
                                sample_indexes=indexes[clean_mask])
        return loss.item()