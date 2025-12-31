# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
#
# this file has been modified from original
#
# --------------------------------------------------------
import math
import numpy as np

import torch
import torch.distributed as dist


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Modified from original to to make it evenly divisible for torch.distributed.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.epoch = 0
        self.num_replicas = num_replicas
        self.rank = rank
        self.indices = np.arange(rank, len(self.dataset), num_replicas)
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # add extra samples if needed to make it evenly divisible
        diff_s_i = self.num_samples - len(self.indices)
        if diff_s_i > 0:
            self.indices = np.insert(self.indices, self.indices[:diff_s_i], 0)
        assert len(self.indices) == self.num_samples

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedWeightedSampler(torch.utils.data.Sampler):
    """
    Distributed weighted sampler that samples more frequently the classes with lower support. Useful for imbalanced
    datasets.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.shuffle = shuffle

    def calculate_weights(self, targets):
        """
        Calculate the weights for each class in the dataset as the inverse of its support.
        Args:
            targets: Vector of targets of shape (N,)

        Returns: Vector of weights of shape (N,)

        """

        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample indices
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.targets
        # calculate weights on the complete targets
        weights = self.calculate_weights(torch.tensor(targets))
        # do the weighted sampling
        subsample_balanced_indices = torch.multinomial(weights, self.total_size, self.replacement)
        # subsample the balanced indices
        subsample_balanced_indices = subsample_balanced_indices[indices]

        return iter(subsample_balanced_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
