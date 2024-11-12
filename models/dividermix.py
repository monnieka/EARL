
import sys
import torch
from datasets import get_dataset

from models.utils.continual_model import ContinualModel, CustomDataset
from utils import DoubleTransform
from utils.args import add_management_args, add_experiment_args, add_noise_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.no_bn import bn_track_stats
 
from torchvision import transforms

def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_noise_args(parser)
    return parser

class DividERmix(ContinualModel):
    NAME = 'dividermix'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.buffer.dataset = get_dataset(args)

        assert self.args.lnl_mode is not None

        if self.args.lnl_mode == 'dividemix':
            transform = DoubleTransform(self.weak_transform, self.weak_transform)
        else: 
            transform = self.weak_transform

        self.transform = transform
        self.task = 0

    def end_task(self, dataset):
        with torch.no_grad():                
            self.task += 1

    def begin_task(self, dataset):
        self.past_epoch = -1
        self.batch_idx = 0
        self.all_iters = len(dataset.train_loader)
        self.cur_task_indexes = torch.from_numpy(dataset.train_loader.dataset.task_indexes).to(self.device)

        if self.args.lnl_mode=='dividemix':
            all_inputs, all_labels = dataset.train_loader.dataset.data, dataset.train_loader.dataset.targets
            all_inputs, all_labels = torch.from_numpy(all_inputs)/255, torch.from_numpy(all_labels)/255
            all_inputs, all_labels = all_inputs.permute(0,3,1,2).float(), all_labels.long()
            self.custom_dset_test = CustomDataset(all_inputs, all_labels, transform=self.fast_test_transform, device='cpu')

            dataset.train_loader.dataset.transform = transforms.ToTensor()

        self.reset_opt()
        self.comodel.train()
        self.net.train()

    def observe(self, inputs, labels, not_aug_inputs, true_labels, indexes, epoch):
        labels = labels.long()
        true_labels = true_labels.long()
        
        if epoch!=self.past_epoch:
            self.past_epoch = epoch
            self.batch_idx = 0
            if epoch>=self.args.warmup_buffer_fitting_epochs:
                all_data = self.custom_dset_test.data
                all_targets = self.custom_dset_test.targets
                all_indexes = self.cur_task_indexes
                if not self.buffer.is_empty():
                    all_data = torch.cat((all_data, self.buffer.examples[:len(self.buffer)].cpu()), dim=0)
                    all_targets = torch.cat((all_targets, self.buffer.labels[:len(self.buffer)].cpu()), dim=0)
                    all_indexes = torch.cat((all_indexes, self.buffer.sample_indexes[:len(self.buffer)]), dim=0)
        
                custom_dset_test = CustomDataset(all_data, all_targets, transform=self.fast_test_transform, device='cpu')
                eval_train_dataset = torch.utils.data.DataLoader(custom_dset_test, shuffle=False, batch_size=64)

                self.correct_idxs_net, self.amb_idxs_net, self.probs_net = self.split_data(None, eval_train_dataset, self.net, return_indices=True, return_probs=True)
                self.correct_idxs_co, self.amb_idxs_co, self.probs_co = self.split_data(None, eval_train_dataset, self.comodel, return_indices=True, return_probs=True)
                self.correct_idxs_net, self.amb_idxs_net, self.probs_net = torch.from_numpy(self.correct_idxs_net), torch.from_numpy(self.amb_idxs_net), torch.from_numpy(self.probs_net)
                self.correct_idxs_co, self.amb_idxs_co, self.probs_co = torch.from_numpy(self.correct_idxs_co), torch.from_numpy(self.amb_idxs_co), torch.from_numpy(self.probs_co)
                self.correct_idxs_net, self.amb_idxs_net, self.probs_net = all_indexes[self.correct_idxs_net].to(self.device), all_indexes[self.amb_idxs_net].to(self.device), self.probs_net.to(self.device)
                self.correct_idxs_co, self.amb_idxs_co, self.probs_co = all_indexes[self.correct_idxs_co].to(self.device), all_indexes[self.amb_idxs_co].to(self.device), self.probs_co.to(self.device)
            
                self.buffer_correct_idxs_net, self.buffer_amb_idxs_net = self.correct_idxs_net[torch.isin(self.correct_idxs_net, self.buffer.sample_indexes)], self.amb_idxs_net[torch.isin(self.amb_idxs_net, self.buffer.sample_indexes)]
                self.buffer_correct_idxs_co, self.buffer_amb_idxs_co = self.correct_idxs_co[torch.isin(self.correct_idxs_co, self.buffer.sample_indexes)], self.amb_idxs_co[torch.isin(self.amb_idxs_co, self.buffer.sample_indexes)]
                self.correct_idxs_net, self.amb_idxs_net = self.correct_idxs_net[torch.isin(self.correct_idxs_net, self.cur_task_indexes)], self.amb_idxs_net[torch.isin(self.amb_idxs_net, self.cur_task_indexes)]
                self.correct_idxs_co, self.amb_idxs_co = self.correct_idxs_co[torch.isin(self.correct_idxs_co, self.cur_task_indexes)], self.amb_idxs_co[torch.isin(self.amb_idxs_co, self.cur_task_indexes)]
                self.buffer_probs_net, self.buffer_probs_co = self.probs_net[-len(self.buffer):], self.probs_co[-len(self.buffer):]
                self.probs_net, self.probs_co = self.probs_net[:-len(self.buffer)], self.probs_co[:-len(self.buffer)]
               
                self.comodel.train()
                self.net.train()

        self.batch_idx += 1
        real_batch_size = inputs.shape[0]

        if not self.buffer.is_empty():
            buf_orig_indexes, buf_inputs, buf_labels, buf_tl, buf_sample_indexes= self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform if self.args.lnl_mode!='dividemix' else None, return_index=True, device=self.device)
            buf_not_aug_inputs = self.buffer.examples[buf_orig_indexes]
            inputs = torch.cat((inputs, buf_inputs), dim=0)
            labels = torch.cat((labels, buf_labels), dim=0)
            not_aug_inputs = torch.cat((not_aug_inputs, buf_not_aug_inputs), dim=0)
            true_labels = torch.cat((true_labels, buf_tl), dim=0)
            indexes = torch.cat((indexes, buf_sample_indexes), dim=0)
        
        self.opt.zero_grad()
        self.opt_co.zero_grad()
        
        if self.args.lnl_mode == 'coteaching':
            loss = self._coteaching(inputs, labels, self.opt, self.opt_co, epoch)
        elif self.args.lnl_mode == 'dividemix' and epoch>=self.args.warmup_buffer_fitting_epochs:
            assert not self.buffer.is_empty()
            with torch.no_grad():
                inputs = self.transform(inputs)
                stream_inputs, buf_inputs = inputs[:,:real_batch_size], inputs[:,real_batch_size:]
                stream_labels, buf_labels = labels[:real_batch_size], labels[real_batch_size:]
                stream_indexes, buf_indexes = indexes[:real_batch_size], indexes[real_batch_size:]

                # ----- net loss
                correct_mask = torch.isin(stream_indexes, self.correct_idxs_net)
                amb_mask = torch.isin(stream_indexes, self.amb_idxs_net)

                buf_correct_mask = torch.isin(buf_indexes, self.buffer_correct_idxs_net)
                buf_amb_mask = torch.isin(buf_indexes, self.buffer_amb_idxs_net)

                str_probs_x = self.probs_net[stream_indexes[correct_mask]].squeeze(-1)[:, 0]
                str_inputs_x, str_inputs_x2 = stream_inputs[0,correct_mask], stream_inputs[1,correct_mask]
                str_inputs_u, str_inputs_u2 = stream_inputs[0,amb_mask], stream_inputs[1,amb_mask]
                str_labels_x = stream_labels[correct_mask]

                buf_idxs = self.buffer.sample_indexes[torch.isin(self.buffer.sample_indexes, buf_indexes[buf_correct_mask])]
                buf_probs_x = self.buffer_probs_net[torch.isin(self.buffer.sample_indexes, buf_indexes[buf_correct_mask])].squeeze(-1)[:, 0]
                buf_mask = ~((buf_idxs.unsqueeze(1)==buf_idxs).cumsum(1)>1).any(1)
                buf_drop_duplicate = torch.isin(buf_indexes[buf_correct_mask], buf_idxs[buf_mask])
                
                buf_probs_x = buf_probs_x[buf_mask]
                buf_inputs_x, buf_inputs_x2 = buf_inputs[0,buf_correct_mask][buf_drop_duplicate], buf_inputs[1,buf_correct_mask][buf_drop_duplicate]
                buf_inputs_u, buf_inputs_u2 = buf_inputs[0,buf_amb_mask], buf_inputs[1,buf_amb_mask]
                buf_labels_x = buf_labels[buf_correct_mask][buf_drop_duplicate]

                inputs_x = torch.cat((str_inputs_x, buf_inputs_x), dim=0)
                inputs_x2 = torch.cat((str_inputs_x2, buf_inputs_x2), dim=0)
                inputs_u = torch.cat((str_inputs_u, buf_inputs_u), dim=0)
                inputs_u2 = torch.cat((str_inputs_u2, buf_inputs_u2), dim=0)
                labels_x = torch.cat((str_labels_x, buf_labels_x), dim=0)
                probs_x = torch.cat((str_probs_x, buf_probs_x), dim=0)

            self.opt.zero_grad()
            self.opt_co.zero_grad()
            loss = self._dividemix(self.opt, self.net, self.comodel, inputs_x, inputs_x2, labels_x, probs_x, inputs_u, inputs_u2, epoch, self.batch_idx,
                                   self.all_iters, self.args.warmup_buffer_fitting_epochs)

            with torch.no_grad():
                # ----- co-net loss
                correct_mask = torch.isin(stream_indexes, self.correct_idxs_co)
                amb_mask = torch.isin(stream_indexes, self.amb_idxs_co)

                buf_correct_mask = torch.isin(buf_indexes, self.buffer_correct_idxs_co)
                buf_amb_mask = torch.isin(buf_indexes, self.buffer_amb_idxs_co)

                str_probs_x = self.probs_co[stream_indexes[correct_mask]].squeeze(-1)[:, 0]
                
                str_inputs_x, str_inputs_x2 = stream_inputs[0,correct_mask], stream_inputs[1,correct_mask]
                str_inputs_u, str_inputs_u2 = stream_inputs[0,amb_mask], stream_inputs[1,amb_mask]
                str_labels_x = stream_labels[correct_mask]

                # drop duplicates
                buf_idxs = self.buffer.sample_indexes[torch.isin(self.buffer.sample_indexes, buf_indexes[buf_correct_mask])]
                buf_probs_x = self.buffer_probs_co[torch.isin(self.buffer.sample_indexes, buf_indexes[buf_correct_mask])].squeeze(-1)[:, 0]
                buf_mask = ~((buf_idxs.unsqueeze(1)==buf_idxs).cumsum(1)>1).any(1)
                buf_drop_duplicate = torch.isin(buf_indexes[buf_correct_mask], buf_idxs[buf_mask])
                
                buf_probs_x = buf_probs_x[buf_mask]
                buf_inputs_x, buf_inputs_x2 = buf_inputs[0,buf_correct_mask][buf_drop_duplicate], buf_inputs[1,buf_correct_mask][buf_drop_duplicate]
                buf_inputs_u, buf_inputs_u2 = buf_inputs[0,buf_amb_mask], buf_inputs[1,buf_amb_mask]
                buf_labels_x = buf_labels[buf_correct_mask][buf_drop_duplicate]

                inputs_x = torch.cat((str_inputs_x, buf_inputs_x), dim=0)
                inputs_x2 = torch.cat((str_inputs_x2, buf_inputs_x2), dim=0)
                inputs_u = torch.cat((str_inputs_u, buf_inputs_u), dim=0)
                inputs_u2 = torch.cat((str_inputs_u2, buf_inputs_u2), dim=0)
                labels_x = torch.cat((str_labels_x, buf_labels_x), dim=0)
                probs_x = torch.cat((str_probs_x, buf_probs_x), dim=0)

            self.opt.zero_grad()
            self.opt_co.zero_grad()
            loss_co = self._dividemix(self.opt_co, self.comodel, self.net, inputs_x, inputs_x2, labels_x, probs_x, inputs_u, inputs_u2, epoch, self.batch_idx,
                                      self.all_iters, self.args.warmup_buffer_fitting_epochs)
            
            loss = loss + loss_co
        else:
            if self.args.lnl_mode == 'dividemix':
                inputs = self.weak_transform(inputs)
            loss = self.loss(self.net(inputs), labels)
            loss_co = self.loss(self.comodel(inputs), labels)
            loss = loss + loss_co

            loss.backward()
            self.opt.step()
            self.opt_co.step()

        self.buffer.add_data(examples=not_aug_inputs[:real_batch_size],
                             labels=labels[:real_batch_size],
                             true_labels=true_labels[:real_batch_size],
                             sample_indexes=indexes[:real_batch_size])

        return loss.item()