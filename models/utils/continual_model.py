
import random
import PIL
from sklearn.mixture import GaussianMixture

import sys
from argparse import Namespace
from contextlib import suppress
from typing import List

import torch
import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm
from datasets import get_dataset
from utils import DoubleTransform, to_kornia_transform
from utils.augmentations import cutmix_data

from utils.conf import get_device
from utils.magic import persistent_locals

import os

import torch.nn.functional as F
from typing import List
import numpy as np
import math
from torchvision import transforms
from torch.distributions.beta import Beta
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets=None, extra=None, transform=None, device="cpu"):
        self.device = device
        self.extra = extra
        self.data = data
        if isinstance(data, torch.Tensor):
            self.data = self.data.to(self.device)
        self.targets = targets.to(device) if targets is not None else None
        self.transform = transform
        self.probs = (torch.ones(len(self.data)) / len(self.data)).to(device)
    def set_probs(self, probs):
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)
        self.probs = probs.to(self.data.device)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx): # data, labels, extra, not_aug_data, probs
        not_aug_data = self.data[idx]
        if isinstance(not_aug_data, str):
            not_aug_data = transforms.ToTensor()(PIL.Image.open(not_aug_data))
            data = not_aug_data.clone()
        else:
            data = not_aug_data.clone()
        if self.transform:
            data = self.transform(data)
        ret = (data,)
        if self.targets is not None:
            ret += (self.targets[idx],)
        if self.extra is not None:
            ret += (self.extra[idx],)
        ret += (not_aug_data,)
        return ret + (self.probs[idx],)

def coteaching_loss(y1, y2, label, r_t, cutmix=False, label_b=None, lam=None):
    criterion = nn.CrossEntropyLoss(reduction='none').to(label.device)
    if cutmix:
        assert label_b is not None and lam is not None
        loss_1 = lam * criterion(y1, label) + (1 - lam) * criterion(y1, label_b)
        loss_2 = lam * criterion(y2, label) + (1 - lam) * criterion(y2, label_b)
    else:
        loss_1 = criterion(y1, label)
        loss_2 = criterion(y2, label)
    num_to_use = math.ceil(r_t * len(label))
    ind_to_use_2 = torch.argsort(loss_1)[:num_to_use]
    ind_to_use_1 = torch.argsort(loss_2)[:num_to_use]
    loss = torch.mean(loss_1[ind_to_use_1]) + torch.mean(loss_2[ind_to_use_2])
    return loss

def linear_rampup(current, warm_up, lambda_u=25, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u * float(current)

def dividemix_loss(outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
    probs_u = torch.softmax(outputs_u, dim=1)

    Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
    Lu = torch.mean((probs_u - targets_u) ** 2)
    lambd = linear_rampup(epoch, warm_up)
    Lu = Lu * lambd
    return Lx, Lu

class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME: str
    COMPATIBILITY: List[str]
    OVERRIDE_SUPPORT_DISTRIBUTED: bool = False

    @torch.no_grad()
    def pseudo_label(self, net, not_aug_inputs, orig_labels, corr_probs, transform, T=0.5):
        N=self.args.mixmatch_naug_buffer_fitting
        # ------------------ PSEUDO LABEL ---------------------
        was_training = net.training
        net.eval()

        unsup_aug_inputs = transform(not_aug_inputs.repeat_interleave(N, 0))
        orig_labels = F.one_hot(orig_labels.repeat_interleave(N, 0), self.num_classes).float()
        corr_probs = corr_probs.repeat_interleave(N, 0)

        unsup_aug_outputs = self.net(unsup_aug_inputs).reshape(N, -1, self.N_HEADS).mean(0)
        unsup_sharp_outputs = unsup_aug_outputs ** (1 / T)
        unsup_norm_outputs = unsup_sharp_outputs / unsup_sharp_outputs.sum(1).unsqueeze(1)
        unsup_norm_outputs = unsup_norm_outputs.repeat(N, 1)

        pseudo_labels_u = corr_probs * orig_labels + (1 - corr_probs) * unsup_norm_outputs

        net.train(was_training)
        return pseudo_labels_u.float(), unsup_aug_inputs

    def consmatch(self, net, opt, loader):
        net.train()
        for data in loader: # data, labels, extra, not_aug_data, probs
            inputs, labels, has_label, not_aug_inputs, corr_probs = data
            inputs, labels, has_label, not_aug_inputs, corr_probs = inputs.to(self.device), labels.to(self.device), has_label.to(self.device), not_aug_inputs.to(self.device), corr_probs.to(self.device)
            has_label = has_label.bool()
            corr_probs = corr_probs[:,0].expand(-1, self.num_classes)

            N_SUP = len(labels[has_label])
            inputs_s = inputs[has_label]
            labels_s = F.one_hot(labels[has_label], self.N_HEADS).float()
            not_aug_inputs_u = not_aug_inputs[~has_label]

            if len(not_aug_inputs_u)>0:
                inputs_u = inputs[~has_label]
                hard_inputs_u = self.hard_transform(not_aug_inputs_u)
                if len(hard_inputs_u.shape)==3:
                    hard_inputs_u = hard_inputs_u.unsqueeze(0)

                all_inputs = torch.cat([inputs_s, inputs_u, hard_inputs_u], dim=0)
            else:
                all_inputs = inputs_s

            logits = net(all_inputs)

            logits_s = logits[:N_SUP]
            loss = F.cross_entropy(logits_s, labels_s, reduction='mean')
            if len(not_aug_inputs_u)>0:
                logits_u, logits_u_hard = logits[N_SUP:].chunk(2)
                loss_unsup = F.mse_loss(logits_u, logits_u_hard, reduction='mean')

                loss = len(logits_s)/len(inputs) * loss + len(logits_u)/len(inputs) * loss_unsup

            opt.zero_grad()
            loss.backward()
            opt.step()

    def mixmatch(self, net, opt, loader):
        net.train()
        for data in loader: # data, labels, extra, not_aug_data, probs
            inputs, labels, has_label, not_aug_inputs, corr_probs = data
            inputs, labels, has_label, not_aug_inputs, corr_probs = inputs.to(self.device), labels.to(self.device), has_label.to(self.device), not_aug_inputs.to(self.device), corr_probs.to(self.device)
            has_label = has_label.bool()
            corr_probs = corr_probs[:,0].expand(-1, self.num_classes)

            N_SUP = len(labels[has_label])
            inputs_s = inputs[has_label]
            labels_s = F.one_hot(labels[has_label], self.N_HEADS).float()
            not_aug_inputs_u = not_aug_inputs[~has_label]

            # mixmatch
            if len(not_aug_inputs_u)>0:
                pseudo_labels_u, inputs_u = self.pseudo_label(net, not_aug_inputs_u, labels[~has_label], corr_probs[~has_label], self.weak_transform)
                all_inputs = torch.cat([inputs_s, inputs_u], dim=0)
                all_targets = torch.cat([labels_s, pseudo_labels_u], dim=0)
            else:
                all_inputs = inputs_s
                all_targets = labels_s

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            lamda = Beta(self.args.mixmatch_alpha_buffer_fitting, self.args.mixmatch_alpha_buffer_fitting).rsample((len(all_inputs),)).to(self.device)
            lamda = torch.max(lamda, 1 - lamda)
            lamda_inputs = lamda.reshape([lamda.shape[0]] + [1] * (len(input_a.shape) - 1))
            lamda_targets = lamda.reshape([lamda.shape[0]] + [1] * (len(target_a.shape) - 1))

            mixed_input = lamda_inputs * input_a + (1 - lamda_inputs) * input_b
            mixed_target = lamda_targets * target_a + (1 - lamda_targets) * target_b

            logits = net(mixed_input)
            mixed_target = mixed_target.to(logits.dtype)
            logits_x = logits[:N_SUP]
            logits_u = logits[N_SUP:]
            
            loss_sup = F.cross_entropy(logits_x, mixed_target[:N_SUP], reduction='mean')
            loss_unsup = F.mse_loss(logits_u, mixed_target[N_SUP:], reduction='mean')

            loss = loss_sup + self.args.mixmatch_lambda_buffer_fitting * loss_unsup 
            # compute gradient and do SGD step
            opt.zero_grad()
            loss.backward()
            opt.step()

    def _dividemix(self, optimizer, net, net2, inputs_x, inputs_x2, labels_x, w_x, inputs_u, inputs_u2, epoch, batch_idx, num_iter, warm_up=10, alpha=4, T=0.5, use_unlabeled=True, device=None):
        device = self.device if device is None else device
        optimizer.zero_grad()
        batch_size = inputs_x.size(0) 
        if batch_size==0:
            return torch.tensor([0.]).to(device)
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, self.N_HEADS, device=device).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).float()
        use_unlabeled = inputs_u is not None and use_unlabeled and len(inputs_u) > 0
        # get_dividemix_targets
        with torch.no_grad():
            # label refinement of labeled samples
            outputs_x = net(inputs_x.to(net.device))
            outputs_x2 = net(inputs_x2.to(net.device))

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

            # label co-guessing of unlabeled samples
            if use_unlabeled:
                outputs_u11 = net(inputs_u.to(net.device))
                outputs_u12 = net(inputs_u2.to(net.device))
                outputs_u21 = net2(inputs_u.to(net2.device))
                outputs_u22 = net2(inputs_u2.to(net2.device))

                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(
                    outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
                ptu = pu ** (1 / T)  # temparature sharpening

                targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                targets_u = targets_u.detach()

            # mixmatch
            l = np.random.beta(alpha, alpha)
            l = max(l, 1 - l)

            if use_unlabeled:
                all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
                all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
            else:
                all_inputs = torch.cat([inputs_x, inputs_x2], dim=0)
                all_targets = torch.cat([targets_x, targets_x], dim=0)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

        logits = net(mixed_input.to(net.device))
        logits_x = logits[:batch_size * 2]
        e = epoch + batch_idx / num_iter
        if use_unlabeled:
            logits_u = logits[batch_size * 2:]
            Lx, Lu = dividemix_loss(logits_x, mixed_target[:batch_size * 2], logits_u,
                                            mixed_target[batch_size * 2:],
                                            e, warm_up)

        else:
            Lx, Lu = dividemix_loss(logits_x, mixed_target[:batch_size * 2], torch.Tensor([[0]]),
                                            torch.Tensor([[0]]),
                                            e, warm_up)

        # regularization
        prior = torch.ones(self.N_HEADS) / self.N_HEADS
        prior = prior.to(device)
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))
        (Lx + Lu + penalty).backward()

        # loss = Lx + lamb * Lu + penalty

        # compute gradient and do SGD step
        # loss.backward()
        optimizer.step()
        
        return Lx + Lu + penalty

    def dividemix(self, epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, warm_up=10, T=0.5,
                  alpha=4, device=None, desc=None):
        net.train()
        net2.eval()  # fix one network and train the other
        desc = '' if desc is None else desc

        num_iter = 0 if labeled_trainloader is None else len(labeled_trainloader)
        use_unlabeled = unlabeled_trainloader is not None

        if num_iter > 0:
            if use_unlabeled:
                unlabeled_train_iter = iter(unlabeled_trainloader)
            for batch_idx, data in tqdm(enumerate(labeled_trainloader), desc=f'Training with DivideMix {desc}', total=len(labeled_trainloader)):
                inputs = data[0].to(device)
                inputs_x2, inputs_x = inputs[:,0], inputs[:,1] # torch.split(inputs, len(inputs) // 2, dim=0)
                labels_x = data[1].to(device)
                if len(labels_x) <= 1:
                    continue
                w_x = data[-1][:, 0].to(device)
                inputs_u, inputs_u2 = None, None
                if use_unlabeled:
                    try:
                        data = next(unlabeled_train_iter)
                        inputs_u = data[0].to(device)
                    except:
                        unlabeled_train_iter = iter(unlabeled_trainloader)
                        data = next(unlabeled_train_iter)
                        inputs_u = data[0].to(device)
                        
                    inputs_u, inputs_u2 = inputs_u[:,0], inputs_u[:,1] # torch.split(inputs_u, len(inputs_u) // 2, dim=0)
                    if inputs_u.size(0) <= 1:
                        continue
                
                self._dividemix(optimizer, net, net2, inputs_x, inputs_x2, labels_x, w_x, inputs_u, inputs_u2, epoch, batch_idx, num_iter, warm_up, alpha, T, use_unlabeled, device)

    @torch.no_grad()
    def batch_split_data(self, inputs, labels, all_losses, model, return_indices=False):
        CE = nn.CrossEntropyLoss(reduction='none')
        was_training = model.training
        model.eval()
        
        c_loss = CE(model(inputs), labels)
        if len(all_losses)==0:
            all_losses = c_loss
        else:
            all_losses = torch.cat([all_losses, c_loss.cpu().detach()], dim=0)
        losses = (all_losses - all_losses.min()) / ((all_losses.max() - all_losses.min()) + torch.finfo(torch.float32).eps)
        input_loss = losses.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        mean_index = np.argsort(gmm.means_, axis=0)
        prob = prob[:, mean_index]
        pred = prob.argmax(axis=1)

        correct_idx = np.where(pred == 0)[0]
        amb_idx = np.where(pred == 1)[0]

        model.train(mode=was_training)

        return correct_idx, amb_idx, prob, losses

    @torch.no_grad()
    def split_data(self, dataset: CustomDataset, test_loader, model, return_indices=False, return_probs=False, device=None):
        CE = nn.CrossEntropyLoss(reduction='none')
        model.eval()
        device = self.device if device is None else device
        model = model.to(device)

        losses = torch.tensor([])
        for data in tqdm(test_loader, desc='Splitting data...'):
            inputs, targets = data[0], data[1]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = CE(outputs, targets)
            losses = torch.cat([losses, loss.detach().cpu()])
        losses = (losses - losses.min()) / ((losses.max() - losses.min()) + torch.finfo(torch.float32).eps)
        input_loss = losses.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        mean_index = np.argsort(gmm.means_, axis=0)
        prob = prob[:, mean_index]
        pred = prob.argmax(axis=1)

        correct_idx = np.where(pred == 0)[0]
        amb_idx = np.where(pred == 1)[0]

        if return_indices:
            return correct_idx, amb_idx, prob
        else:
            dataset.set_probs(prob.squeeze(axis=-1))
            correct_size = len(correct_idx)
            if correct_size == 0:
                return None, None

            amb_size = len(amb_idx)
            batch_size = int(amb_size / correct_size * test_loader.batch_size)
            if batch_size < 2:
                batch_size = 2
            batch_size = min(batch_size, test_loader.batch_size)

            dataloader_correct = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, correct_idx),
                                            shuffle=True,
                                            batch_size=test_loader.batch_size)
            if amb_size <= 2:
                dataloader_ambiguous = None
            else:
                dataloader_ambiguous = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, amb_idx),
                                                shuffle=True,
                                                batch_size=batch_size)

            return dataloader_correct, dataloader_ambiguous
            
    def fit_buffer(self):
        # get number of gpus
        print('Loading all data', file=sys.stderr)
        buf_data = self.buffer.get_all_data(device="cpu")
        inputs, labels = buf_data[0], buf_data[1]

        if self.args.lnl_mode == 'dividemix':
            transform = DoubleTransform(self.weak_transform, self.weak_transform)
        else:
            if self.args.use_hard_transform_buffer_fitting: 
                transform = self.hard_transform
            else:
                transform = self.weak_transform
        
        print('Building train dataset', file=sys.stderr)
        train_dataset = CustomDataset(inputs, labels, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.buffer_fitting_batch_size, shuffle=True)

        print('Building test dataset', file=sys.stderr)
        test_dataset = CustomDataset(inputs, labels, transform=self.fast_test_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.buffer_fitting_batch_size, shuffle=False)

        print('Building optimizers', file=sys.stderr)
        opt = self.get_opt([p for p in self.net.parameters() if p.requires_grad], lr=self.args.buffer_fitting_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2, eta_min=self.args.buffer_fitting_lr * 0.01
        )
        if self.args.lnl_mode in ["dividemix","coteaching"]:
            opt_co = self.get_opt([p for p in self.comodel.parameters() if p.requires_grad], lr=self.args.buffer_fitting_lr)
            scheduler_co = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_co, T_0=1, T_mult=2, eta_min=self.args.buffer_fitting_lr * 0.01
            )
            
        for e in range(self.args.buffer_fitting_epochs):
            print(f'Buffer fitting epoch {e}/{self.args.buffer_fitting_epochs+1}', file=sys.stderr)
            if self.args.lnl_mode == 'coteaching':
                self.coteaching(train_loader, e, opt, opt_co)
            elif self.args.lnl_mode == 'dividemix' and e>self.args.warmup_buffer_fitting_epochs:
                opt.zero_grad()
                opt_co.zero_grad()
                label_loader, unlabel_loader = self.split_data(dataset=train_dataset, test_loader=test_loader,
                                                               model=self.comodel, device=self.device)
                self.dividemix(e, self.net, self.comodel, opt, label_loader, unlabel_loader,
                               warm_up=self.args.warmup_buffer_fitting_epochs, device=self.device, desc=': net')
                
                opt.zero_grad()
                opt_co.zero_grad()
                label_loader, unlabel_loader = self.split_data(dataset=train_dataset, test_loader=test_loader,
                                                               model=self.net, device=self.device)
                self.dividemix(e, self.comodel, self.net, opt_co, label_loader, unlabel_loader,
                               warm_up=self.args.warmup_buffer_fitting_epochs, device=self.device, desc=': co-net')
                opt.zero_grad()
                opt_co.zero_grad()
            elif self.args.lnl_mode == 'mixmatch':
                _, amb_idxs, probs = self.split_data(dataset=train_loader.dataset, test_loader=test_loader, model=self.net, return_indices=True)
                amb_idxs, probs = torch.from_numpy(amb_idxs), torch.from_numpy(probs)
                corr_lab = torch.ones(len(train_dataset))
                corr_lab[amb_idxs] = 0
                train_loader.dataset.set_probs(probs.to(train_loader.dataset.device))
                train_loader.dataset.extra = corr_lab.to(train_loader.dataset.device)
                self.mixmatch(self.net, opt, train_loader)
            elif self.args.lnl_mode == 'consmatch':
                _, amb_idxs, probs = self.split_data(dataset=train_loader.dataset, test_loader=test_loader, model=self.net, return_indices=True)
                amb_idxs, probs = torch.from_numpy(amb_idxs), torch.from_numpy(probs)
                corr_lab = torch.ones(len(train_dataset))
                corr_lab[amb_idxs] = 0
                train_loader.dataset.set_probs(probs.to(train_loader.dataset.device))
                train_loader.dataset.extra = corr_lab.to(train_loader.dataset.device)
                self.consmatch(self.net, opt, train_loader)
            else:
                for i, data in enumerate(train_loader):    
                    inputs, labels = data[0], data[1]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    if self.args.lnl_mode == 'dividemix':
                        inputs = inputs[:,0]
                        opt_co.zero_grad()
                    opt.zero_grad()                        
                    if self.args.enable_cutmix and random.random() < self.args.cutmix_prob:
                        inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels)

                        logits = self.net(inputs)
                        loss = lam * self.loss(logits, labels_a) + (1 - lam) * self.loss(logits, labels_b)

                        if self.args.lnl_mode in ["dividemix","coteaching"]:
                            cologits = self.comodel(inputs)
                            loss += lam * self.loss(cologits, labels_a) + (1 - lam) * self.loss(cologits, labels_b)
                    else:
                        logits = self.net(inputs)
                        loss = self.loss(logits, labels)

                        if self.args.lnl_mode in ["dividemix","coteaching"]:
                            cologits = self.comodel(inputs)
                            loss += self.loss(cologits, labels)
                        
                    loss.backward()
                    opt.step()
                    if self.args.lnl_mode in ["dividemix"]:
                        opt_co.step()
            
            scheduler.step()
            if self.args.lnl_mode in ["dividemix","coteaching"]:
                scheduler_co.step()

    def _coteaching(self, x, y, opt, opt_co, epoch):
        if hasattr(self.args, 'noise_rate'):
            r_t = 1 - min([epoch / 10 * (self.args.noise_rate), self.args.noise_rate])
        else:
            r_t = 1 - min([epoch / 10 * (0.2), 0.2])

        opt.zero_grad()
        opt_co.zero_grad()

        do_cutmix = self.args.enable_cutmix and np.random.random(1) < self.args.cutmix_prob
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y)
            logit = self.net(x)
            logit_2 = self.comodel(x)

            loss = coteaching_loss(logit, logit_2, labels_a, r_t, cutmix=True,
                                    label_b=labels_b, lam=lam)
        else:
            logit = self.net(x)
            logit_2 = self.comodel(x)

            loss = coteaching_loss(logit, logit_2, y, r_t, cutmix=False)

        preds = torch.argmax(logit, dim=1)

        loss.backward()
        opt.step()
        opt_co.step()

        return loss, preds  

    def coteaching(self, dataset, epoch, opt, opt_co):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        self.net.train()
        self.comodel.train()
        for data in dataset:
            x = data[0]
            y = data[1]

            x = x.to(self.device)
            y = y.to(self.device)

            loss, preds = self._coteaching(x, y, opt, opt_co, epoch)

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        n_batches = len(dataset)

        return total_loss / n_batches, correct / num_data


    def init_comodel(self):
        self.comodel = get_dataset(self.args).get_backbone().to(self.device)
        self.reset_opt()

    def get_opt(self, params, lr=None):
        lr = self.args.lr if lr is None else lr

        return SGD(params, lr=lr,
                    weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)


    def reset_opt(self):
        self.opt = self.get_opt([p for p in self.net.parameters() if p.requires_grad])
        if self.args.lnl_mode is not None and hasattr(self, 'comodel'):
            self.opt_co = self.get_opt([p for p in self.comodel.parameters() if p.requires_grad])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x) #mascherare teste classi viste e non qui 

    def meta_observe(self, *args, **kwargs):

        ret = self.observe(*args, **kwargs)
        return ret

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, 
                not_aug_inputs: torch.Tensor, true_labels: torch.Tensor, epoch:int, idx:int) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        raise NotImplementedError

    def autolog_wandb(self, locals, **kwargs):
        """
        All variables starting with "_wandb_" or "loss" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        pass
            
    def log_losses(self, stream_losses, buffer_losses, logits, labels, true_labels, buf_logits, buf_labels, buf_true_labels):
        with torch.no_grad():
            loss_clean_stream = stream_losses[labels==true_labels].mean()
            loss_noisy_stream = stream_losses[labels!=true_labels].mean()
            if self.task > 0 and not self.buffer.is_empty():
                loss_clean = buffer_losses[buf_labels==buf_true_labels].mean()
                loss_noisy = buffer_losses[buf_labels!=buf_true_labels].mean()
            else:
                loss_clean, loss_noisy = torch.tensor(0.), torch.tensor(0.)
                #loss_clean = loss_clean / ([labels == true_labels].shape[0])
        stream_confidences = torch.tensor(-float('inf'))
        stream_clean_predictions = torch.topk(torch.softmax(logits[labels==true_labels],dim=1), k=2, dim=1)[0]
        stream_noisy_predictions = torch.topk(torch.softmax(logits[labels!=true_labels],dim=1), k=2, dim=1)[0] 
        stream_clean_confidences = stream_clean_predictions[:,0] - stream_clean_predictions[:,1]
        stream_noisy_confidences = stream_noisy_predictions[:,0] - stream_noisy_predictions[:,1]
        stream_confidences = torch.cat([stream_clean_confidences, stream_noisy_confidences])

        if self.task > 0 and not self.buffer.is_empty():
            buffer_confidences = torch.tensor(-float('inf'))
            buffer_clean_predictions = torch.topk(torch.softmax(buf_logits[buf_labels==buf_true_labels],dim=1), k=2, dim=1)[0]
            buffer_noisy_predictions = torch.topk(torch.softmax(buf_logits[buf_labels!=buf_true_labels],dim=1), k=2, dim=1)[0]
            buffer_clean_confidences = buffer_clean_predictions[:,0] - buffer_clean_predictions[:,1]
            buffer_noisy_confidences = buffer_noisy_predictions[:,0] - buffer_noisy_predictions[:,1]
            buffer_confidences = torch.cat([buffer_clean_confidences, buffer_noisy_confidences])
        

    def load_pretrained(self):
        assert os.path.isfile(self.args.checkpoint), f"File not found: {self.args.checkpoint}"
        self.load_from_checkpoint(self.args.checkpoint, ignore_classifier=False)
        print("Loaded!")

    def to(self, device):
        super().to(device)
        self.device = device
        for d in [x for x in self.__dir__() if hasattr(getattr(self, x), 'device')]:
            getattr(self, d).to(device)
        return self
    
    def load_from_checkpoint(self,cp_path, new_classes=None, ignore_classifier=False) -> None:
        """
        Load pretrain checkpoint, optionally ignores and rebuilds final classifier.

        :param cp_path: path to checkpoint
        :param new_classes: ignore and rebuild classifier with size `new_classes`
        :param moco: if True, allow load checkpoint for Moco pretraining
        """
        s = torch.load(cp_path, map_location=get_device())
        if 'state_dict' in s:  
            s = {k.replace('encoder_q.', ''): i for k,
                 i in s['state_dict'].items() if 'encoder_q' in k}

        if not ignore_classifier:
            if new_classes is not None:
                self.net.classifier = torch.nn.Linear(
                    self.net.classifier.in_features, self.num_aux_classes).to(get_device())
                for k in list(s):
                    if 'classifier' in k:
                        s.pop(k)
            else:
                cl_weights = [s[k] for k in list(s.keys()) if 'classifier' in k]
                if len(cl_weights) > 0:
                    cl_size = cl_weights[-1].shape[0]
                    self.net.classifier = torch.nn.Linear(
                        self.net.classifier.in_features, cl_size).to(get_device())
        else:
            for k in list(s):
                if 'classifier' in k:
                    s.pop(k)
                    
        for k in list(s):
            if 'net' in k:
                s[k[4:]] = s.pop(k)
        for k in list(s):
            if 'wrappee.' in k:
                s[k.replace('wrappee.', '')] = s.pop(k)
        for k in list(s):
            if '_features' in k:
                s.pop(k)

        _, unm = self.net.load_state_dict(s, strict=False)

        if new_classes is not None or ignore_classifier:
            assert all(['classifier' in k for k in unm]
                        ), f"Some of the keys not loaded where not classifier keys: {unm}"
        else:
            assert len(unm) == 0, f"Missing keys: {unm}"

    def base_path(self):
        return '/data/'


    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: nn.Module) -> None:
        super(ContinualModel, self).__init__()
        self.iteration = 0
        self.clean_loss, self.noisy_loss = {}, {}
        self.gmm_rate = None
        self.net = backbone
        self.dset = get_dataset(args)
        self.N_HEADS = self.dset.N_CLASSES_PER_TASK*self.dset.N_TASKS if isinstance(self.dset.N_CLASSES_PER_TASK, int) else sum(self.dset.N_CLASSES_PER_TASK)
        self.loss = loss
        self.args = args
        self.args.lip_compute_mode = self.args.lip_compute_mode if hasattr(args, "lip_compute_mode") else "different_layer"
        self.args.lip_difference_mode = self.args.lip_difference_mode if hasattr(args, "lip_difference_mode") else "sample"
        self.transform = transform
        if self.args.enable_amnesia:
            assert self.args.n_epochs%2==0 or self.args.start_with_replay==1, "If replay_on_off, n_epochs must be even or start_with_replay must be 1 (or last epoch will not have replay)"

        dset = get_dataset(args)
        if 'seq-ntu60' not in args.dataset:
            if not hasattr(self.dset, 'TEST_TRANSFORM'):
                self.test_transform = transforms.Compose(transform.transforms[-1].transforms[-2:])
                self.fast_test_transform = to_kornia_transform(transform.transforms[-1].transforms[-2:])
            else:
                self.test_transform = dset.TEST_TRANSFORM
                self.fast_test_transform = to_kornia_transform(dset.TEST_TRANSFORM)
            self.normalize_transform = transform.transforms[-1].transforms[-1]
            self.weak_transform = to_kornia_transform(transform.transforms[-1].transforms)
            self.hard_transform = to_kornia_transform(transforms.Compose([transforms.ToPILImage()]+dset.STRONG_TRANSFORM.transforms))
        else:
            if not hasattr(self.dset, 'TEST_TRANSFORM'):
                self.test_transform = transforms.Compose([ntu_to_tensor(), dset.get_normalization_transform()])
                self.fast_test_transform = self.test_transform
            else:
                self.test_transform = dset.TEST_TRANSFORM
                self.fast_test_transform = dset.TEST_TRANSFORM
            self.normalize_transform = dset.get_normalization_transform()
            self.weak_transform = transform
            self.hard_transform = dset.STRONG_TRANSFORM
            # self.hard_transform = lambda: (_ for _ in ()).throw(NotImplementedError("Strong transform not implemented for NTU60")) # god forgive me

        self.reset_opt()
        self.device = get_device()
        if self.args.checkpoint is not None:
            self.load_pretrained()

            if not hasattr(self.net, 'classifier'):
                raise NotImplementedError('Backbone does not support checkpoint loading. Please implement a classifier.')
            self.net.classifier = torch.nn.Linear(self.net.classifier.in_features, self.N_HEADS).to(get_device())

        if not self.NAME or not self.COMPATIBILITY:
            raise NotImplementedError('Please specify the name and the compatibility of the model.')

        if self.args.lnl_mode in ["dividemix","coteaching"]:
            print("Using LNL mode -- setting up CO-MODEL for", self.args.lnl_mode)
            self.init_comodel()

    @torch.no_grad()
    def compute_persample_losses(self, inputs: torch.Tensor, labels: torch.Tensor, true_labels: torch.Tensor,group = 'stream') -> float:
        clean_set, noisy_set = inputs[labels==true_labels], inputs[labels != true_labels]
        clean_loss, noisy_loss =None, None

        if clean_set.shape[0]>0:
            clean_outs = self.net(clean_set)
            clean_loss = self.loss(clean_outs, labels[labels == true_labels].long(), reduction='none').detach()
        if noisy_set.shape[0]>0:
            noisy_outs = self.net(noisy_set)
            noisy_loss = self.loss(noisy_outs, labels[labels != true_labels].long(), reduction='none').detach()

        if clean_loss is not None:
            if not f'{group}' in self.clean_loss.keys():
                self.clean_loss[f'{group}'] = clean_loss
            else:
                self.clean_loss[f'{group}'] = torch.cat( (self.clean_loss[f'{group}'], clean_loss))

        if noisy_loss is not None:
            if not f'{group}' in self.noisy_loss.keys():
                self.noisy_loss[f'{group}']  = noisy_loss
            else:
                self.noisy_loss[f'{group}'] = torch.cat( (self.noisy_loss[f'{group}'], noisy_loss))    
    
    @torch.no_grad()
    def compute_lip_values(self, inputs,group: str):
        _, partial_features = self.net(inputs, returnt='full')

        lip_inputs = [inputs] + partial_features[:-1]

        lip_values = self.get_feature_lip_coeffs(lip_inputs)
        lip_values = torch.stack(lip_values, dim=1)
        
        if not f'{group}' in self.lip_values.keys():
            self.lip_values[f'{group}'] = lip_values
        else:
            self.lip_values[f'{group}'] = torch.cat( (self.lip_values[f'{group}'],lip_values) )        


    @torch.no_grad()
    def get_clean_mask(self, logits, labels, true_labels):                        

        logits, labels, true_labels = logits.to(self.device), labels.to(self.device), true_labels.to(self.device)
       
        #clean = None
        loss_each = self.loss(logits, labels, reduction='none')
        loss_each = (loss_each-loss_each.min())/((loss_each.max()-loss_each.min())+torch.finfo(torch.float32).eps)    
        loss_each = loss_each.reshape(-1,1).cpu()
        if loss_each.shape[0] < 2:
            return None
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(loss_each)
        prob = gmm.predict_proba(loss_each)
        mean_index = np.argsort(gmm.means_, axis=0)
        prob = prob[:, mean_index]
        pred = prob.argmax(axis=1)

        clean = pred == 0
        #true_tot  = (labels == true_labels).sum()
        #clean_tot = (labels[clean] == true_labels[clean]).sum()
        #gmm_rate = (clean_tot / true_tot) *100
        #if gmm_rate is not None:
        #    if self.gmm_rate == None:
        #        self.gmm_rate = gmm_rate
        #    else:
        #        self.gmm_rate = torch.cat((self.gmm_rate.view(-1),gmm_rate.view(-1)))
        return clean
        
    

 
