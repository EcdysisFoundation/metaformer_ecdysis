# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
#
# this file has been modified from orginal
#
# --------------------------------------------------------

from pathlib import Path

import torch
import torch.distributed as dist
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.mixup import Mixup
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import InterpolationMode

from logger import create_logger
from .samplers import SubsetRandomSampler, DistributedWeightedSampler


def build_loader(config):

    config.defrost()
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=__name__,
                           local_rank=config.LOCAL_RANK)

    if config.EVAL_MODE:
        dataset_test, _ = build_dataset(is_train=False, config=config, logger=logger)
        data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                       batch_size=config.DATA.BATCH_SIZE,
                                                       shuffle=False,
                                                       num_workers=config.DATA.NUM_WORKERS,
                                                       pin_memory=config.DATA.PIN_MEMORY,
                                                       drop_last=False)
        config.DATA.TEST_SAMPLES = len(dataset_test)
        config.freeze()
        return dataset_test, data_loader_test

    else:
        dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config, logger=logger)
        config.DATA.TRAIN_SAMPLES = len(dataset_train)
        config.freeze()
        logger.info(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")

        dataset_val, _ = build_dataset(is_train=False, config=config, logger=logger)
        logger.info(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
           sampler_train = SubsetRandomSampler(dataset_train)
        else:
            if config.TRAIN.SAMPLER == 'weighted':
                sampler_train = DistributedWeightedSampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
            else:
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )

        sampler_val = SubsetRandomSampler(dataset_val)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )

        # setup mixup / cutmix
        mixup_fn = None
        mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
                label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

        return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config, logger):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET.startswith('bugbox'):
        dataset, nb_classes = load_insect_data(config, is_train, transform, logger)
        if is_train:
            config.DATA.CLASS_NAMES = dataset.classes
    else:
        raise NotImplementedError("Dataset not supported.")

    return dataset, nb_classes


def load_insect_data(config, is_train, transform, logger) -> (datasets.ImageFolder, int):
    """
    Loads insect images from a folder. The data must be inside the datasets directory and must follow IMAGENET's
    directory structure, i.e.:
    datasets/insectfam/
    ├── meta.yaml
    ├── test
    │ ├── Coleoptera_Staphylinidae
    │ ├── Diptera_Chloropidae
    │ └── Diptera_Sphaeroceridae
    ├── train
    │ ├── Coleoptera_Staphylinidae
    │ ├── Diptera_Chloropidae
    │ └── Diptera_Sphaeroceridae
    └── val
        ├── Coleoptera_Staphylinidae
        ├── Diptera_Chloropidae
        └── Diptera_Sphaeroceridae

    Returns: Dataset and number of classes

    """
    root = Path(config.DATA.DATA_PATH)
    if not root.is_dir():
        raise ValueError(f'Dataset root directory not found in {root}')
    if config.EVAL_MODE:
        prefix = 'test'
    else:
        prefix = 'train' if is_train else 'val'
    dataset = datasets.ImageFolder(str(root / prefix), transform=transform)
    classes, class_to_index = dataset.find_classes(dataset.root)
    nb_classes = len(classes)

    logger.info(f'Found {len(dataset)} images and {nb_classes} classes in {prefix} split of {config.DATA.DATASET}')

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation='bilinear',
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=InterpolationMode.BILINEAR)
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
