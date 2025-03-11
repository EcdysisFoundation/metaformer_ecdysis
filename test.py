import datetime
import os
import torch
import torch.distributed as dist
import numpy as np
import torch.backends.cudnn as cudnn

from tqdm import tqdm

from main import parse_option
from data import build_loader

"""
# example, version = 3, folder = morphospecies
python test.py --cfg configs/ecdysis_test.yaml  --data-path "datasets/bugbox_model_3/" --tag morphospecies --version 3 --pretrain "output/ecdysis/morphospecies/3/best.pth"
"""

def setup_distributed(config):
    rank = int(config.LOCAL_RANK)
    world_size = 2
    torch.cuda.set_device(int(config.LOCAL_RANK))
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout = datetime.timedelta(seconds=900),
        world_size=world_size,
        rank=rank)
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

def test_dataloader(config):

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    print(len(data_loader_val))

    pbar_desc = f'Testing | Rank {dist.get_rank()}'

    with tqdm(desc=pbar_desc, total=len(data_loader_val), unit='batch') as pbar:
        for idx, data in enumerate(data_loader_val):
            images, target = data
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

            pbar.update()
            pbar.set_postfix_str(f'Memory {memory_used:.0f}MB')

if __name__ == '__main__':
    args, config = parse_option()

    print(os.environ.__dict__)
    exit()

    setup_distributed(config)
    config.defrost()
    config.freeze()

    test_dataloader(config)
    dist.barrier()
    dist.destroy_process_group()

