import os
import torch
import torch.distributed as dist

from tqdm import tqdm

from main import parse_option, setup_distributed
from data import build_loader

"""
# example, version = 3, folder = morphospecies
conda activate pytroch
/home/ecdysis/miniconda3/envs/pytorch/bin/torchrun --nproc_per_node 2 test.py --cfg configs/ecdysis_test.yaml  --data-path "datasets/bugbox_model_3/" --tag morphospecies --version 3 --pretrain "output/ecdysis/morphospecies/3/best.pth"
"""

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

    setup_distributed(config)
    config.defrost()
    config.freeze()

    test_dataloader(config)
    dist.barrier()
    dist.destroy_process_group()

