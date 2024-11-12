import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.buffer import Buffer

def setup(rank, world_size):
    host = os.environ['SLURM_NODELIST'].split(',')[0]
    ephemeral_port_range = 65535 - 32768
    port = 32768 + int(os.environ['SLURM_JOBID']) % ephemeral_port_range

    os.environ['MASTER_ADDR'] = host
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    print(f"Running basic DDP example on rank {rank}/{world_size} (host {host}, node {os.environ['SLURMD_NODENAME']} port {port}).")
    sys.stdout.flush()
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print("Inited")
    sys.stdout.flush()

def is_distributed():
    return dist.is_initialized()

def wait_for_master(verbose=False):
    if is_distributed() and 'RANK' in os.environ:
        if verbose:
            print("Waiting for master", os.environ["RANK"], flush=True)

        dist.barrier()

def is_master():
    return not is_distributed() or int(os.environ['RANK']) == 0

def sync_ddp_buffer_from_master(buffer: Buffer, verbose=False):
    wait_for_master(verbose=verbose)
    if is_distributed():
        for attr in buffer.attributes:
            if verbose:
                print("Syncing", attr, os.environ["RANK"], flush=True)
            if hasattr(buffer, attr):
                dist.broadcast(getattr(buffer, attr), src=0)
    wait_for_master(verbose=verbose)

class CustomDDP(DDP):
    intercept_names = ['args', 'classifier', 'num_classes', 'set_return_prerelu', 'projector', 'NAME']

    def __init__(self, module, *args, **kwargs):
        prev_names = [name for name in dir(module) if not name.startswith('_')]
        super().__init__(module, *args, **kwargs)

        # append attributes from module
        self.intercept_names += [n for n in prev_names if n not in dir(self)]

    def __getattr__(self, name: str):
        if name in self.intercept_names:
            return getattr(self.module, name)
        else:
            return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        if name in self.intercept_names:
            setattr(self.module, name, value)
        else:
            super().__setattr__(name, value)

def make_ddp(model):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["SLURM_GPUS_ON_NODE"]) * int(os.environ["SLURM_NNODES"])
    rank = int(os.environ["LOCAL_RANK"]) + (int(os.environ["SLURM_NODEID"]) * int(os.environ["SLURM_GPUS_ON_NODE"]))
    os.environ["RANK"] = str(rank)

    print("INIT PROCESS GROUP::", "RANK", rank, "|", "WORLD SIZE", world_size, "LOCAL RANK", 
          local_rank, "MASTER ADDR", os.environ['MASTER_ADDR'], "MASTER PORT", os.environ['MASTER_PORT'], flush=True)
    print(torch.cuda.device_count())
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    model.to(local_rank)
    model.device = f"cuda:{local_rank}"
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print("INIT DDP::", "RANK", rank, "|", "WORLD SIZE", world_size)
    ddp_model = CustomDDP(model, device_ids=[local_rank])
    return ddp_model


class CustomDP(DataParallel):

    intercept_names = ['classifier', 'num_classes', 'set_return_prerelu', 'projector']

    def __getattr__(self, name: str):
        if name in self.intercept_names:
            return getattr(self.module, name)
        else:
            return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        if name in self.intercept_names:
            setattr(self.module, name, value)
        else:
            super().__setattr__(name, value)


def make_dp(model):
    return CustomDP(model, device_ids=range(torch.cuda.device_count()))#.to('cuda:0')
