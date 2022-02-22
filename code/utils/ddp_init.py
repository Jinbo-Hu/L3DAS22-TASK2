import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os


def run(demo_fn, args, cfg, dataset):
    mp.spawn(demo_fn,
            args=(args, cfg, dataset),
            nprocs=torch.cuda.device_count(),
            join=True)

def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size, args=None):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:  # single GPU
        return value
    if type(value) != torch.Tensor:
        value = torch.as_tensor(value).to(dist.get_rank())
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value.item()    

   