"""
L2P: Learning to Prompt for Continual Learning

Note:
    L2P USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import torch

from models.utils.continual_model import ContinualModel
from timm import create_model  # noqa
from models.l2p_utils.l2p_model import L2PModel
from datasets import get_dataset
from utils.args import add_management_args, add_experiment_args, add_noise_args, ArgumentParser
from torchvision import transforms

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Learning to Prompt (L2P)')
    add_management_args(parser)
    add_experiment_args(parser)
    add_noise_args(parser)

    # Custom rehearsal args
    parser.add_argument('--buffer_size', type=int, default=0,
                    help='The size of the memory buffer.')
    parser.add_argument('--use_abs', type=int, choices=[0, 1], default=0)

    # Prompt parameters
    parser.add_argument('--prompt_pool', default=True, type=bool,)
    parser.add_argument('--pool_size_l2p', default=10, type=int, help='number of prompts (M in paper)')
    parser.add_argument('--length', default=5, type=int, help='length of prompt (L_p in paper)')
    parser.add_argument('--l2p_top_k', default=5, type=int, help='top k prompts to use (N in paper)')
    parser.add_argument('--prompt_key', default=True, type=bool, help='Use learnable prompt key')
    parser.add_argument('--prompt_key_init', default='uniform', type=str, help='initialization type for key\'s prompts')
    parser.add_argument('--use_prompt_mask', default=False, type=bool)
    parser.add_argument('--batchwise_prompt', default=True, type=bool)
    parser.add_argument('--embedding_key', default='cls', type=str)
    parser.add_argument('--predefined_key', default='', type=str)
    parser.add_argument('--pull_constraint', default=True)
    parser.add_argument('--pull_constraint_coeff', default=0.1, type=float)

    parser.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
    parser.add_argument('--head_type', default='prompt', choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
    parser.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], nargs='*', type=list, help='freeze part in backbone model')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    parser.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')

    parser.add_argument('--clip_grad', type=float, default=1, help='Clip gradient norm')
    return parser

class L2P(ContinualModel):
    NAME = 'l2p'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        """
        L2P re-defines the backbone model to include the prompt parameters. This is done *before* calling the super constructor, so that the backbone is already initialized when the super constructor is called.
        """
        del backbone
        print("-" * 20)
        print(f"WARNING: L2P USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        if args.use_abs:
            assert args.buffer_size > 0, "Buffer size must be greater than 0 when using bi-fold loss-aware."

        args.lr = args.lr * args.batch_size / 256.0
        backbone = L2PModel(args)

        super().__init__(backbone, loss, args, transform)
        self.opt = self.get_opt([p for p in self.net.model.parameters() if p.requires_grad])
        self.dataset = get_dataset(args)
        if self.args.buffer_size>0:
            from utils.buffer import Buffer

            self.buffer = Buffer(self.args.buffer_size)
            self.buffer.lossoir = 2 if self.args.use_abs else 0
            self.buffer.dataset = get_dataset(args)

        self.current_task=0

    def end_task(self, dataset):
        self.current_task += 1

        if self.args.buffer_size > 0 and len(self.buffer) > 0:
            with torch.no_grad():
                is_true_labels = self.buffer.true_labels.cpu().float()
                avg_buf_clean = is_true_labels.mean()

                print('Clean buffer perc: ', avg_buf_clean.item())
                print('N buffer clean: ', is_true_labels.sum().item())
                print('N buffer noisy: ', (1-is_true_labels).sum().item())


        if self.args.buffer_fitting_epochs > 0:
            required_grad_params_name = [n for n,p in self.net.named_parameters() if p.requires_grad]
            for n,p in self.net.named_parameters():
                p.requires_grad=False
            self.net.model.head = torch.nn.Linear(self.net.model.head.in_features, self.net.model.head.in_features).to(self.device)
            for p in self.net.model.head.parameters():
                p.requires_grad = True
            self.fit_buffer()
            for n,p in self.net.named_parameters():
                if n in required_grad_params_name:
                    p.requires_grad=True

    def _compute_offsets(self, task):
        cpt = self.dataset.N_CLASSES_PER_TASK
        if isinstance(cpt, int):
            offset1 = task * cpt
            offset2 = (task + 1) * cpt
        else:
            offset1 = sum(cpt[:task])
            offset2 = sum(cpt[:task + 1])
        return offset1, offset2

    def begin_task(self, dataset):
        self.net.original_model.eval()

        if hasattr(self, 'opt'):
            self.opt.zero_grad(set_to_none=True)
            del self.opt
        self.opt = self.get_opt([p for p in self.net.model.parameters() if p.requires_grad])

    def observe(self, inputs, labels, not_aug_inputs, is_true_labels, indexes, epoch=None):
        labels=labels.long()
        B=len(inputs)
        buf_inputs = None
        if self.args.buffer_size > 0 and len(self.buffer) > 0:
            if self.args.replay_on_off==0 or epoch % 2 != self.args.start_with_replay:
                buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.batch_size, transform=self.transform, device=self.device)
                inputs = torch.cat([inputs, buf_inputs])

        outputs = self.net(inputs, return_outputs=True)
        logits = outputs['logits']

        offset_1, offset_2 = self._compute_offsets(self.current_task)

        loss_buf = torch.tensor(0., device=self.device)
        if buf_inputs is not None:
            buf_logits = logits[B:]
            loss_buf = self.loss(buf_logits[:, :offset_2], buf_labels)
            logits = logits[:B]
        logits[:, :offset_1] = -float('inf')

        loss_stream = self.loss(logits[:, :offset_2], labels)
        loss = loss_stream + loss_buf
        if self.args.pull_constraint and 'reduce_sim' in outputs:
            loss = loss - self.args.pull_constraint_coeff * outputs['reduce_sim']

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        self.opt.step()

        if self.args.use_abs:
            with torch.no_grad():
                if isinstance(self.weak_transform, transforms.Compose):
                    not_aug_logits = self.net(self.weak_transform.transforms[-1](not_aug_inputs))
                else: # kornia
                    not_aug_logits = self.net(self.weak_transform[-1](not_aug_inputs))

                not_aug_logits = not_aug_logits
                loss_not_aug_ext = self.loss(not_aug_logits[:, offset_1:offset_2],labels-offset_1, reduction='none')

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
                        
                        self.buffer.add_data(examples=not_aug_inputs, labels=labels, true_labels=is_true_labels, loss_scores=loss_scores)
                else:
                    self.buffer.add_data(examples=not_aug_inputs, labels=labels, true_labels=is_true_labels)

        return loss.item()

    def get_parameters(self):
        return [p for n, p in self.net.model.named_parameters() if 'prompt' in n or 'head' in n]

    def forward(self, x):
        if self.current_task > 0:
            _, offset_2 = self._compute_offsets(self.current_task - 1)
        else:
            offset_2 = self.N_CLASSES
        return self.net(x)[:, :offset_2]