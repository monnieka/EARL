"""
CODA-Prompt: COntinual Decomposed Attention-based Prompting

Note:
    CODA-Prompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import timm
from utils.args import *
from models.utils.continual_model import ContinualModel
import torch
from datasets import get_dataset
from models.coda_prompt_utils.model import Model
from utils.schedulers import CosineSchedule
from utils.args import add_management_args, add_experiment_args, add_noise_args, ArgumentParser
from torchvision import transforms

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                            ' CODA-Prompt: COntinual Decomposed Attention-based Prompting')

    add_management_args(parser)
    add_experiment_args(parser)
    add_noise_args(parser)

    # Custom rehearsal args
    parser.add_argument('--buffer_size', type=int, default=0,
                    help='The size of the memory buffer.')
    parser.add_argument('--use_abs', type=int, choices=[0, 1], default=0)


    parser.add_argument('--mu', type=float, default=0.0, help='weight of prompt loss')
    parser.add_argument('--pool_size', type=int, default=100, help='pool size')
    parser.add_argument('--prompt_len', type=int, default=8, help='prompt length')
    parser.add_argument('--virtual_bs_iterations', type=int, default=1, help="virtual batch size iterations")
    return parser


class CodaPrompt(ContinualModel):
    NAME = 'coda_prompt'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        del backbone
        print("-" * 20)
        print(f"WARNING: CODA-Prompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        if args.use_abs:
            assert args.buffer_size > 0, "Buffer size must be greater than 0 when using ABS."

        self.dataset = get_dataset(args)
        self.n_classes = self.dataset.N_CLASSES
        self.n_tasks = self.dataset.N_TASKS
        backbone = Model(num_classes=self.n_classes, pt=True, prompt_param=[self.n_tasks, [args.pool_size, args.prompt_len, 0]])
        super().__init__(backbone, loss, args, transform)
        self.net.task_id = 0
        self.opt = self.get_optimizer()
        
        self.dataset = get_dataset(args)
        if self.args.buffer_size>0:
            from utils.buffer import Buffer

            self.buffer = Buffer(self.args.buffer_size)
            self.buffer.lossoir = 2 if self.args.use_abs else 0
            self.buffer.dataset = get_dataset(args)

        self.current_task=0

    def end_task(self, dataset):
        self.current_task += 1

    def get_optimizer(self):
        params_to_opt = list(self.net.prompt.parameters()) + list(self.net.last.parameters())
        optimizer_arg = {'params': params_to_opt,
                         'lr': self.args.lr,
                         'weight_decay': self.args.optim_wd}
        if self.args.optimizer == 'sgd':
            opt = torch.optim.SGD(**optimizer_arg)
        elif self.args.optimizer == 'adam':
            opt = torch.optim.Adam(**optimizer_arg)
        else:
            raise ValueError('Optimizer not supported for this method')
        return opt

    def begin_task(self, dataset):
        self.offset_1, self.offset_2 = self._compute_offsets(self.current_task)

        if self.current_task != 0:
            self.net.task_id = self.current_task
            self.net.prompt.process_task_count()
            self.opt = self.get_optimizer()

        self.scheduler = CosineSchedule(self.opt, K=self.args.n_epochs)
        self.old_epoch = 0
        self.task_iteration = 0

    def _compute_offsets(self, task):
        cpt = self.dataset.N_CLASSES_PER_TASK
        if isinstance(cpt, int):
            offset1 = task * cpt
            offset2 = (task + 1) * cpt
        else:
            offset1 = sum(cpt[:task])
            offset2 = sum(cpt[:task + 1])
        return offset1, offset2
    
    def observe(self, inputs, labels, not_aug_inputs, true_labels, indexes, epoch=None):
        labels=labels.long()
        B=len(inputs)
        buf_inputs = None
        if self.args.buffer_size > 0 and len(self.buffer) > 0:
            if self.args.replay_on_off==0 or epoch % 2 != self.args.start_with_replay:
                buf_inputs, buf_labels = self.buffer.get_data(self.args.batch_size, transform=self.transform, device=self.device)
                inputs = torch.cat([inputs, buf_inputs])

        if self.scheduler and self.old_epoch != epoch:
            self.scheduler.step()
            self.old_epoch = epoch
            self.iteration = 0
        self.opt.zero_grad()
        logits, loss_prompt = self.net(inputs, train=True)
        loss_prompt = loss_prompt.sum()
        logits = logits[:, :self.offset_2]
        
        loss_buf = torch.tensor(0., device=self.device)
        if buf_inputs is not None:
            buf_logits = logits[B:]
            loss_buf = self.loss(buf_logits, buf_labels)
            logits = logits[:B]

        logits[:, :self.offset_1] = -float('inf')
        loss_ce = self.loss(logits, labels)
        loss = loss_ce + self.args.mu * loss_prompt + loss_buf
        if self.task_iteration == 0:
            self.opt.zero_grad()

        torch.cuda.empty_cache()
        loss.backward()
        if self.task_iteration > 0 and self.task_iteration % self.args.virtual_bs_iterations == 0:
            self.opt.step()
            self.opt.zero_grad()

        if self.args.use_abs:
            with torch.no_grad():
                if isinstance(self.weak_transform, transforms.Compose):
                    not_aug_logits = self.net(self.weak_transform.transforms[-1](not_aug_inputs))
                else: # kornia
                    not_aug_logits = self.net(self.weak_transform[-1](not_aug_inputs))

                not_aug_logits = not_aug_logits
                loss_not_aug_ext = self.loss(not_aug_logits[:, self.offset_1:self.offset_2],labels-self.offset_1, reduction='none')

        if self.args.buffer_size > 0:
            if self.args.replay_on_off == 0 or epoch % 2 == self.args.start_with_replay:
                # Insert selection
                if self.args.use_abs:
                    clean_mask = torch.ones_like(labels, dtype=torch.bool)
                    if self.args.discard_noisy == 'topk':
                        _, clean_mask = torch.topk(loss_not_aug_ext, round((1-self.args.top_k)*inputs.shape[0]), largest=False)
                    if clean_mask.sum() > 0:
                        loss_scores = loss_not_aug_ext[clean_mask].detach().to(dtype=not_aug_inputs.dtype)
                        not_aug_inputs = not_aug_inputs[clean_mask]
                        labels = labels[clean_mask]
                        
                        self.buffer.add_data(examples=not_aug_inputs, labels=labels, loss_scores=loss_scores)
                else:
                    self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        self.task_iteration += 1
        return loss.item()

    def forward(self, x):
        return self.net(x)[:, :self.offset_2]