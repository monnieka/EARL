import time
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from torch import nn
import tqdm
 
from datasets import get_dataset
from sklearn.metrics.pairwise import cosine_similarity

from models.utils.continual_model import ContinualModel, CustomDataset
from utils import DoubleTransform, to_kornia_transform
from utils.cutmix import cutmix
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, add_noise_args, ArgumentParser
from utils.buffer import Buffer
import torch.nn.functional as F
from torchvision import transforms

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='PuriDivER')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_noise_args(parser)

    parser.add_argument('--use_bn_classifier', type=int, default=1, choices=[0,1])
    parser.add_argument('--disable_train_aug', type=int, default=0, choices=[0,1])
    parser.add_argument('--initial_alpha', type=float, default=0.5)
    
    return parser

# https://github.com/pytorch/pytorch/issues/11959
def soft_cross_entropy_loss(input, target, reduction='mean'):
    """
    :param input: (batch, *)
    :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
    """
    logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
    batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')

class PuriDivER(ContinualModel):
    NAME = 'puridiver'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, "cpu")
        self.buffer.dataset = get_dataset(args)
        self.seen_so_far = torch.tensor([]).long().to(self.device)

        dset = get_dataset(args)
        self.cpt = dset.N_CLASSES_PER_TASK
        self.num_classes = dset.N_TASKS * self.cpt
        self.task = 0

        self.ii = 0
        self._past_it_t = time.time()
        self._avg_it_t = 0
        self.past_loss = 0
        self.eye = torch.eye(self.num_classes).to(self.device)

        self.hard_transform = to_kornia_transform(transforms.Compose([transforms.ToPILImage()]+dset.STRONG_TRANSFORM.transforms))
        self.fast_test_transform = to_kornia_transform(transform.transforms[-1].transforms[-2:])
        self.test_transform = transforms.Compose(transform.transforms[-1].transforms[-2:])
        self.weak_transform = to_kornia_transform(transform.transforms[-1].transforms)

        if self.args.minibatch_size is None:
            self.args.minibatch_size = self.args.batch_size


    def reset_opt(self):
        self.opt = torch.optim.SGD(
            self.net.parameters(), lr=self.args.buffer_fitting_lr, momentum=self.args.optim_mom, weight_decay=self.args.optim_wd
        )

    def forward(self, inputs):
        if self.net.device != inputs.device:
            self.net.to(inputs.device)
        return self.net(inputs)
    
    def get_subset_dl_from_idxs(self, idxs, batch_size, probs=None, transform=None):
        if idxs is None:
            return None
        assert batch_size is not None
        
        examples, labels, true_labels = self.buffer.examples[idxs], self.buffer.labels[idxs], self.buffer.true_labels[idxs]

        tmp_buffer = Buffer(self.args.buffer_size, self.device)
        if probs is not None:
            probs=torch.from_numpy(probs)
        tmp_buffer.add_data(examples=examples, labels=labels, true_labels=true_labels, logits=probs)
        return tmp_buffer.get_dataloader(batch_size=batch_size, shuffle=True, transform=transform)


    @torch.no_grad()
    def split_data_puridiver(self, n=2):
        self.net.eval()

        losses = []
        uncertainties = []
        for batch_idx, batch in enumerate(self.buffer.get_dataloader(batch_size=64, shuffle=False)):
            x, y, y_true = batch[0], batch[1], batch[-1]
            x, y, y_true = x.to(self.device), y.to(self.device), y_true.to(self.device)
            x = self.fast_test_transform(x)
            out = self.net(x)
            probs = F.softmax(out, dim=1)
            uncerts = 1 - torch.max(probs, 1)[0]

            losses.append(F.cross_entropy(out, y, reduction='none'))
            uncertainties.append(uncerts)
        
        losses = torch.cat(losses, dim=0).cpu()
        uncertainties = torch.cat(uncertainties, dim=0).cpu().reshape(-1, 1)
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        losses = losses.unsqueeze(1)

        # GMM for correct vs others samples
        gmm_loss = GaussianMixture(n_components=n, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm_loss.fit(losses)
        gmm_loss_means = gmm_loss.means_
        if gmm_loss_means[0] <= gmm_loss_means[1]:
            small_loss_idx = 0
            large_loss_idx = 1
        else:
            small_loss_idx = 1
            large_loss_idx = 0

        loss_prob = gmm_loss.predict_proba(losses)
        pred = loss_prob.argmax(axis=1)

        corr_idxs = np.where(pred == small_loss_idx)[0]
        if len(corr_idxs) == 0:
            return None, None, None

        # 2nd GMM using large loss datasets
        high_loss_idxs = np.where(pred == large_loss_idx)[0]

        ambiguous_idxs, incorrect_idxs = None, None
        if len(high_loss_idxs) > 2:
            # GMM for uncertain vs incorrect samples
            gmm_uncert = GaussianMixture(n_components=n, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm_uncert.fit(uncertainties[high_loss_idxs])
            prob_uncert = gmm_uncert.predict_proba(uncertainties[high_loss_idxs])
            pred_uncert = prob_uncert.argmax(axis=1)
            if gmm_uncert.means_[0] <= gmm_uncert.means_[1]:
                small_loss_idx = 0
                large_loss_idx = 1
            else:
                small_loss_idx = 1
                large_loss_idx = 0
            
            idx_uncert = np.where(pred_uncert == small_loss_idx)[0]
            amb_size = len(idx_uncert)
            ambiguous_batch_size = max(2,int(amb_size / len(corr_idxs) * self.args.batch_size))
            if amb_size <= 2:
                ambiguous_idxs = None
            else:
                ambiguous_idxs = high_loss_idxs[idx_uncert]

            idx_uncert = np.where(pred_uncert == large_loss_idx)[0]
            incorrect_size = len(idx_uncert)
            incorrect_batch_size = max(2,int(incorrect_size / len(corr_idxs) * self.args.batch_size))
            if incorrect_size<=2:
                incorrect_idxs = None
            else:
                incorrect_idxs = high_loss_idxs[idx_uncert]
        
        correct_dl = self.get_subset_dl_from_idxs(corr_idxs, self.args.batch_size, transform=self.hard_transform)

        if ambiguous_idxs is not None:
            ambiguous_dl = self.get_subset_dl_from_idxs(ambiguous_idxs, ambiguous_batch_size, transform=DoubleTransform(self.weak_transform, self.hard_transform))
        else:
            ambiguous_dl = None

        if incorrect_idxs is not None:
            incorrect_dl = self.get_subset_dl_from_idxs(incorrect_idxs, incorrect_batch_size, probs=loss_prob[incorrect_idxs], transform=DoubleTransform(self.weak_transform, self.hard_transform))
        else:
            incorrect_dl = None

        return correct_dl, ambiguous_dl, incorrect_dl
    
    def train_with_mixmatch(self, loader_L, loader_U, loader_R):
        criterion_U = nn.MSELoss()
        criterion_L = nn.CrossEntropyLoss()

        iter_U = iter(loader_U)
        iter_R = iter(loader_R)
        avg_loss = 0

        # R: weak, hard
        # L: hard
        # U: weak, hard
        self.net.train()
        for batch in loader_L:
            self.opt.zero_grad()
            inputs_L, labels_L = batch[0], batch[1]
            if len(inputs_L)==1:
                continue
            try:
                inputs_U = next(iter_U)[0]
            except:
                iter_U = iter(loader_U)
                inputs_U = next(iter_U)[0]
            try:
                batch_R = next(iter_R)
                inputs_R, labels_R, probs_R = batch_R[0], batch_R[1], batch_R[2]
            except:
                iter_R = iter(loader_R)
                batch_R = next(iter_R)
                inputs_R, labels_R, probs_R = batch_R[0], batch_R[1], batch_R[2]

            inputs_L, labels_L = inputs_L.to(self.device), labels_L.to(self.device)
            inputs_U, inputs_R = inputs_U.to(self.device), inputs_R.to(self.device)
            labels_R, probs_R = labels_R.to(self.device), probs_R.to(self.device)
            labels_R = F.one_hot(labels_R, self.num_classes)
            corr_prob = probs_R[:,0].unsqueeze(1).expand(-1, self.num_classes)

            inputs_U = torch.cat([inputs_U[:,0], inputs_U[:,1]], dim=0)
            inputs_R = torch.cat([inputs_R[:,0], inputs_R[:,1]], dim=0)

            do_cutmix = self.args.enable_cutmix and np.random.random(1) < self.args.cutmix_prob
            if do_cutmix:
                inputs_L, labels_L_a, labels_L_b, lam = cutmix(inputs_L, labels_L)

                all_inputs = torch.cat([inputs_R, inputs_U, inputs_L], dim=0)
                all_outputs = self.net(all_inputs)
                outputs_R, outputs_U, outputs_L = torch.split(all_outputs, [inputs_R.size(0), inputs_U.size(0), inputs_L.size(0)])
            
                loss_L = lam * self.loss(outputs_L, labels_L_a) + (1 - lam) * criterion_L(outputs_L, labels_L_b)
            else:
                all_inputs = torch.cat([inputs_R, inputs_U, inputs_L], dim=0)
                all_outputs = self.net(all_inputs)
                outputs_R, outputs_U, outputs_L = torch.split(all_outputs, [inputs_R.size(0), inputs_U.size(0), inputs_L.size(0)])
                outputs_L = self.net(inputs_L)
        
                loss_L = self.loss(outputs_L, labels_L)

            outputs_U_weak, outputs_U_strong = torch.split(outputs_U, outputs_U.size(0) // 2)
            outputs_R_pseudo, outputs_R = torch.split(outputs_R, outputs_R.size(0) // 2) # weak, strong

            probs_R_pseudo = torch.softmax(outputs_R_pseudo, dim=1)
            soft_pseudo_labels = corr_prob * labels_R + (1 - corr_prob) * probs_R_pseudo.detach()

            loss_R = soft_cross_entropy_loss(outputs_R, soft_pseudo_labels)
            loss_U = criterion_U(outputs_U_weak, outputs_U_strong)
            
            coeff_L = (len(labels_L) / (len(labels_L) + len(labels_R) + len(outputs_U_weak)))
            coeff_R = (len(labels_R) / (len(labels_R) + len(labels_L) + len(outputs_U_weak)))
            coeff_U = (len(outputs_U_weak) / (len(labels_R) + len(labels_L) + len(outputs_U_weak)))
            loss = coeff_L * loss_L + coeff_U * loss_U + coeff_R * loss_R

            assert not torch.isnan(loss).any()
            # backward
            loss.backward()
            self.opt.step()

            avg_loss += loss.item()
        return avg_loss / len(loader_L)

    def base_fit_buffer(self, loader=None):
        self.net.train()
        avg_loss = 0
        loader = self.buffer.get_dataloader(batch_size=self.args.batch_size, shuffle=True, transform=self.hard_transform) if loader is None else loader
        for batch in loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            if len(x)==1:
                continue
            
            self.opt.zero_grad()

            do_cutmix = self.args.enable_cutmix and np.random.rand(1) < self.args.cutmix_prob
            if do_cutmix:
                x, y_a, y_b, lam = cutmix(x, y)

                out = self.net(x)

                loss = lam * self.loss(out, y_a) + (1 - lam) * self.loss(out, y_b)
            else:
                out = self.net(x)

                loss = self.loss(out, y)

            assert not torch.isnan(loss).any()
            loss.backward()
            self.opt.step()

            avg_loss += loss.item()
        return avg_loss / len(loader)
        

    def fit_buffer(self):
        for param_group in self.opt.param_groups:
            param_group["lr"] = self.args.buffer_fitting_lr

        for epoch in range(self.args.buffer_fitting_epochs):
            if self.args.debug_mode and epoch > self.args.warmup_buffer_fitting_epochs + 50:
                break
            if epoch < self.args.warmup_buffer_fitting_epochs:
                print(' - WARM-UP buffer epoch {}/{}'.format(epoch+1, self.args.buffer_fitting_epochs), end='')
                loss = self.base_fit_buffer()
            else:
                correct_dl, ambiguous_dl, incorrect_dl = self.split_data_puridiver()

                if ambiguous_dl is not None and incorrect_dl is not None:
                    print(' - train with puridiver {}/{}'.format(epoch+1, self.args.buffer_fitting_epochs), end='')
                    loss = self.train_with_mixmatch(correct_dl, ambiguous_dl, incorrect_dl)
                else:
                    print(' - normal train on buffer {}/{}'.format(epoch+1, self.args.buffer_fitting_epochs), end='')
                    loss = self.base_fit_buffer()

            buf_data = self.buffer.get_all_data()
            buf_not_aug_inputs, buf_labels, buf_true_labels = buf_data[0], buf_data[1], buf_data[2]
            _, _, buf_acc, true_buf_acc = self._non_observe_data(self.fast_test_transform(buf_not_aug_inputs), buf_labels, buf_true_labels)

            perc_clean = (self.buffer.labels==self.buffer.true_labels).float().mean().item()
            print(f' == loss: {loss:.2f} | buf_acc: {buf_acc:.2f} | true_buf_acc: {true_buf_acc:.2f} | perc_clean: {perc_clean:.2f} | LR: {self.opt.param_groups[0]["lr"]:.4f}')
            self.scheduler.step()
                    
    def end_task(self, dataset):
        # fit classifier on P
        self.task += 1

        if self.args.buffer_fitting_epochs>0:
            self.fit_buffer()

    def get_classifier_weights(self):
        if isinstance(self.net.classifier, nn.Sequential):
            return self.net.classifier[0].weight.detach()
        return self.net.classifier.weight.detach()

    def get_sim_score(self, feats, targets):
        # relevant representation
        cl_weights = self.get_classifier_weights()

        relevant_idx = cl_weights[targets[0],:]>cl_weights.mean(dim=0)

        cls_features = feats[:, relevant_idx]
        sim_score = torch.from_numpy(cosine_similarity(cls_features.cpu())).mean(1).to(self.device)#, cls_features, dim=1)

        return (sim_score - sim_score.mean()) / sim_score.std()
    
    def get_current_alpha_sim_score(self, loss):
        return self.args.initial_alpha*min(1,1/loss)
    
    def begin_task(self, dataset):
        self.total_its = len(dataset.train_loader) * self.args.n_epochs
        self.ii = 0

        if self.task==0: #BLE
            if self.args.use_bn_classifier:
                self.net.classifier = nn.Sequential(nn.Linear(self.net.classifier.in_features, self.net.classifier.out_features, bias=False), 
                                                    nn.BatchNorm1d(self.net.classifier.out_features, affine=True, eps=1e-6).to(self.device)).to(self.device)

                for m in self.net.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                        m.eps = 1e-6

        self.reset_opt()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt, T_0=1, T_mult=2, eta_min=self.args.buffer_fitting_lr * 0.01
        )
        for param_group in self.opt.param_groups:
            param_group["lr"] = self.args.lr
    
        if self.args.disable_train_aug:
            dataset.train_loader.dataset.transform = self.test_transform

    @torch.no_grad()
    def _non_observe_data(self, inputs, labels, true_labels=None):
        was_training = self.net.training
        self.net.eval()

        dset = CustomDataset(inputs, labels, extra=true_labels, device=self.device)
        dl = torch.utils.data.DataLoader(dset, batch_size=min(len(dset), 256), shuffle=False, num_workers=0)
        
        feats = []
        losses = []
        true_accs, accs = [], []
        for batch in tqdm.tqdm(dl, leave=False, desc=' - recomputing buffer losses and features', disable=self.args.non_verbose):
            inputs, labels, true_labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            out, feat = self.net(inputs, returnt='both')
            acc = (out.argmax(dim=1) == labels).float().mean().item()
            tacc = (out.argmax(dim=1) == true_labels).float().mean().item()
            feats.append(feat)
            losses.append(F.cross_entropy(out, labels, reduction='none'))
            accs.append(acc)
            true_accs.append(tacc)

        feats = torch.cat(feats, dim=0)
        losses = torch.cat(losses, dim=0)
        acc = np.mean(accs)
        true_acc = np.mean(true_accs)

        self.net.train(was_training)

        return feats, losses, acc, true_acc
    
    def puridiver_update_buffer(self, stream_not_aug_inputs: torch.Tensor, stream_labels: torch.Tensor, stream_true_labels: torch.Tensor, alpha_sim_score):
        if len(self.buffer)<self.args.buffer_size:
            self.buffer.add_data(examples=stream_not_aug_inputs, labels=stream_labels, true_labels=stream_true_labels)
            return -1, -1
        buf_data = self.buffer.get_all_data()
        buf_not_aug_inputs, buf_labels, buf_true_labels = buf_data[0], buf_data[1], buf_data[2]
        buf_not_aug_inputs, buf_labels, buf_true_labels = buf_not_aug_inputs.to(self.device), buf_labels.to(self.device), buf_true_labels.to(self.device)
        not_aug_inputs = torch.cat([buf_not_aug_inputs, stream_not_aug_inputs], dim=0)
        labels = torch.cat([buf_labels, stream_labels], dim=0)
        true_labels = torch.cat([buf_true_labels, stream_true_labels], dim=0)

        cur_idxs = torch.arange(len(not_aug_inputs)).to(self.device)
        feats, losses, buf_acc, true_buf_acc = self._non_observe_data(self.fast_test_transform(not_aug_inputs), labels, true_labels=true_labels)
        lbs = labels[cur_idxs]
        while len(lbs)>self.args.buffer_size:
            fts = feats[cur_idxs]
            lss = losses[cur_idxs]

            clss, cls_cnt = lbs.unique(return_counts=True)
            # argmax w/ random tie-breaking
            cls_to_drop = clss[cls_cnt == cls_cnt.max()]
            cls_to_drop = cls_to_drop[torch.randperm(len(cls_to_drop))][0]
            mask = lbs == cls_to_drop
            
            sim_score = self.get_sim_score(fts[mask], lbs[mask])
            div_score = (1 - alpha_sim_score) * lss[mask] + alpha_sim_score * sim_score

            drop_cls_idx = div_score.argmax()
            drop_idx = cur_idxs[mask][drop_cls_idx]
            cur_idxs = cur_idxs[cur_idxs != drop_idx]

            lbs = labels[cur_idxs]

        self.buffer.empty()
        self.buffer.add_data(examples=not_aug_inputs[cur_idxs], labels=labels[cur_idxs], true_labels=true_labels[cur_idxs])
        return buf_acc, true_buf_acc
    
    def observe(self, inputs, labels, not_aug_inputs, true_labels, indexes, epoch):
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        self.net.train()

        self.ii+=1
        labels = labels.long()
        true_labels = true_labels.long()

        B = len(inputs)

        self.opt.zero_grad()

        if self.task>0: # starting from second task
            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.hard_transform, device=self.device)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        do_cutmix = self.args.enable_cutmix and np.random.rand(1) < self.args.cutmix_prob
        if do_cutmix:
            inputs, labels_a, labels_b, lam = cutmix(inputs, labels)

            outputs = self.net(inputs)

            loss = lam * self.loss(outputs, labels_a) + (1 - lam) * self.loss(outputs, labels_b)
        else:
            outputs = self.net(inputs)

            loss = self.loss(outputs, labels)

        assert not torch.isnan(loss).any()
        loss.backward()
        self.opt.step()

        alpha_sim_score = self.get_current_alpha_sim_score(loss)

        buf_acc, true_buf_acc = self.puridiver_update_buffer(not_aug_inputs[:B], labels[:B], true_labels[:B], alpha_sim_score)
        
        ctime = time.time()
        perc_clean = (self.buffer.labels==self.buffer.true_labels).float().mean().item()
        self._avg_it_t = (self._avg_it_t+(ctime-self._past_it_t))/(self.ii)
        remaing_time = (self.total_its-self.ii)*self._avg_it_t

        print(f" - Epoch it.: {self.ii}/{self.total_its}; loss: {loss.item():.2f}"
            f" (it/sec: {self._avg_it_t:.2f} / {remaing_time:.2f}s left)"
            f" [buffer acc: {round(buf_acc*100, 2)} | true buffer acc: {round(true_buf_acc*100, 2)} | perc clean: {perc_clean:.2f} | alpha: {alpha_sim_score:.2f} | loss: {loss.item():.2f}]")
        self._past_it_t = ctime

        return loss.item()
    