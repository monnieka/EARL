import time
import numpy # needed (don't change it)
import importlib
import os

import socket
import sys

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')

from utils import is_rehearsal
import datetime
import uuid
from argparse import ArgumentParser

import torch
from datasets import ContinualDataset, get_dataset
from models import get_all_models, get_model

from utils.args import add_management_args
from utils.conf import set_random_seed
from utils.continual_training import train as ctrain
from utils.distributed import make_dp
from utils.training import train

def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)
    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)
    
    noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
    if args.noise_type is not None:
        args.noise_type = noise_type_map[args.noise_type]

    if args.savecheck:
        now = time.strftime("%Y%m%d-%H%M%S")
        args.ckpt_name = f"{args.model}_{args.dataset}_{args.buffer_size if hasattr(args, 'buffer_size') else 0}_{args.n_epochs}_{args.noise if args.noise is not None else 'clean'}_{int(args.noise_rate*100) if args.noise_rate is not None and args.noise_rate>0 else '0'}_{str(now)}"
        args.ckpt_name_replace = f"{args.model}_{args.dataset}_{'{}'}_{args.buffer_size if hasattr(args, 'buffer_size') else 0}__{args.n_epochs}_{args.noise if args.noise is not None else 'clean'}_{int(args.noise_rate*100) if args.noise_rate is not None and args.noise_rate>0 else '0'}_{str(now)}"

    return args


def main(args=None):
    lecun_fix()
    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)

    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
    if is_rehearsal(args) and args.minibatch_size is None:
        args.minibatch_size = dataset.get_minibatch_size()
    if args.n_warmup_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_warmup_epochs = dataset.get_warmup_epochs()

    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    if args.distributed == 'dp':
        model.net = make_dp(model.net)
        model.to('cuda:0')
        args.conf_ngpus = torch.cuda.device_count()
    elif 'ddp' in args.distributed:
        # DDP breaks the buffer, it has to be synchronized.
        assert not is_rehearsal(args) or model.OVERRIDE_SUPPORT_DISTRIBUTED, 'Distributed Data Parallel not supported yet.'

    if not hasattr(args, "minibatch_size") or args.minibatch_size is None:
        args.minibatch_size = args.batch_size

    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args)  #200 teste, rimettere a 100
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)

if __name__ == '__main__':
    main()
