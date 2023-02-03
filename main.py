import logging
import os
import time
import argparse
import datetime
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchmetrics
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, StatScores
from tqdm import tqdm

from config import get_config
from data.build import build_dataset
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, load_pretained
from torch.utils.tensorboard import SummaryWriter
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('MetaFG training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path',default='./imagenet', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')  # Disabled by default due to apex library installation issues
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    
    parser.add_argument('--num-workers', type=int, 
                        help="num of workers on dataloader ")
    
    parser.add_argument('--lr', type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float,
                        help='weight decay (default: 0.05 for adamw)')
    
    parser.add_argument('--min-lr', type=float,
                        help='learning rate')
    parser.add_argument('--warmup-lr', type=float,
                        help='warmup learning rate')
    parser.add_argument('--epochs', type=int,
                        help="epochs")
    parser.add_argument('--warmup-epochs', type=int,
                        help="epochs")
    
    parser.add_argument('--dataset', type=str,
                        help='dataset')
    parser.add_argument('--lr-scheduler-name', type=str,
                        help='lr scheduler name,cosin linear,step')
    
    parser.add_argument('--pretrain', type=str,
                        help='pretrain')
    
    parser.add_argument('--tensorboard', action='store_true', help='using tensorboard')

    parser.add_argument('--ignore-user-warnings', action='store_true', default=False,
                        help='Disable logging of UserWarnings')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    metrics = [
        Accuracy(num_classes=config.MODEL.NUM_CLASSES, average='micro'),
        Precision(num_classes=config.MODEL.NUM_CLASSES, average='macro'),
        Recall(num_classes=config.MODEL.NUM_CLASSES, average='macro'),
        F1Score(num_classes=config.MODEL.NUM_CLASSES, average='macro')
    ]

    if config.EVAL_MODE:
        metrics.append(StatScores(num_classes=config.MODEL.NUM_CLASSES, reduce='macro'))
        metrics.append(ConfusionMatrix(num_classes=config.MODEL.NUM_CLASSES))

    metrics = torchmetrics.MetricCollection(metrics)
    model.metric = metrics  # This need to be here to avoid problems with distributed training

    model.cuda()
    logger.debug(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {n_parameters}")

    if config.MODEL.PRETRAINED:
        load_pretained(config, model_without_ddp, logger)
        if config.EVAL_MODE:
            # Create test set loader
            dataset_test, _ = build_dataset(is_train=False, config=config, logger=logger)
            data_loader_test = DataLoader(
                dataset_test,
                batch_size=config.DATA.BATCH_SIZE,
                shuffle=False,
                num_workers=config.DATA.NUM_WORKERS,
                pin_memory=config.DATA.PIN_MEMORY,
                drop_last=False
            )
            # Eval and return
            _ = validate(config, data_loader_test, model, 0, metrics)
            return

    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9}")
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    max_accuracy = 0.0
    if config.MODEL.RESUME:
        logger.info(f"**********normal test***********")
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model, config.TRAIN.START_EPOCH, metrics)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} validation images: {acc1:.1f}%")
        if config.DATA.ADD_META:
            logger.info(f"**********mask meta test***********")
            acc1, acc5, loss = validate(config, data_loader_val, model, config.TRAIN.START_EPOCH, metrics, mask_meta=True)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    # min_loss = float('inf')  # Used to save the best model

    # TB logger
    tb_dir = Path(config.OUTPUT) / 'tb'
    tb_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    start_time = time.time()
    with tqdm(desc=f'Training | Rank {dist.get_rank()}', total=config.TRAIN.EPOCHS, unit='epoch') as pbar:
        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
            # Train
            data_loader_train.sampler.set_epoch(epoch)
            train_one_epoch_local_data(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn,
                                       lr_scheduler, writer)
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0):
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

            # Validate
            acc1, acc5, loss = validate(config, data_loader_val, model, epoch, metrics, tb_logger=writer)
            # logger.info(f"Accuracy of the network on the {len(dataset_val)} validation images: {acc1:.1f}%")

            if acc1 > max_accuracy:
                max_accuracy = acc1
                pbar.set_postfix_str(f'Maximum accuracy on validation so far: {max_accuracy:.3f}%')
                if dist.get_rank() == 0:
                    save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger,
                                    best=True)

            # if loss < min_loss:
            #     # Best validation loss so far
            #     min_loss = loss
            #     logger.info(f'Minimum validation loss: {min_loss:.3f}')

            if config.DATA.ADD_META:
                logger.info(f"**********mask meta test***********")
                acc1, acc5, loss = validate(config, data_loader_val, model, epoch, metrics, mask_meta=True, tb_logger=writer)
                logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

            pbar.update()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch_local_data(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler,tb_logger=None):
    """
    Train for one epoch
    Args:
        config: Configuration object
        model: Model to train, usually a subclass or `torch.nn.Module`
        criterion: Loss function
        data_loader: Train data loader
        optimizer: Optimizer object
        epoch: Epoch index
        mixup_fn: Mixup function, only needed when using AMP
        lr_scheduler: Learning rate scheduler object
        tb_logger: Tensorboard `SummaryWriter` object for logging
    """
    model.train()
    if hasattr(model.module, 'cur_epoch'):
        model.module.cur_epoch = epoch
        model.module.total_epoch = config.TRAIN.EPOCHS
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    with tqdm(desc=f'Training | Rank {dist.get_rank()} | Epoch [{epoch}/{config.TRAIN.EPOCHS}]',
              total=len(data_loader), unit='batch') as pbar:
        for idx, data in enumerate(data_loader):
            if config.DATA.ADD_META:
                samples, targets,meta = data
                meta = [m.float() for m in meta]
                meta = torch.stack(meta,dim=0)
                meta = meta.cuda(non_blocking=True)
            else:
                samples, targets= data
                meta = None

            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
            if config.DATA.ADD_META:
                outputs = model(samples,meta)
            else:
                outputs = model(samples)

            if config.TRAIN.ACCUMULATION_STEPS > 1:
                loss = criterion(outputs, targets)
                loss = loss / config.TRAIN.ACCUMULATION_STEPS
                if config.AMP_OPT_LEVEL != "O0":
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(amp.master_params(optimizer))
                else:
                    loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(model.parameters())
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                if config.AMP_OPT_LEVEL != "O0":
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(amp.master_params(optimizer))
                else:
                    loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(model.parameters())
                optimizer.step()
                lr_scheduler.step_update(epoch * num_steps + idx)

            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

            pbar.update()
            pbar.set_postfix_str(f'Memory {memory_used:.0f}MB')

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if tb_logger is not None:
            step = epoch * num_steps + idx
            tb_logger.add_scalar('train/loss', loss_meter.avg, global_step=step)
            tb_logger.add_scalar('train/grad_norm', norm_meter.avg, global_step=step)
            lr = optimizer.param_groups[0]['lr']
            tb_logger.add_scalar('train/lr', lr, global_step=step)

    epoch_time = time.time() - start


@torch.no_grad()
def validate(config, data_loader, model, epoch, metric, mask_meta=False, tb_logger=None):
    """
    Compute metrics on validation or test sets
    Args:
        config: Configuration object
        data_loader: Validation or test data loader
        model: Model to train, usually a subclass or `torch.nn.Module`
        epoch: Epoch index
        metric: Metric object from `torchmetrics`
        mask_meta: Set it to True to use metadata for classification
        tb_logger: Tensorboard writer object for logging

    Returns: Accuracy and loss metrics
    """
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    with tqdm(desc=f'Validating | Rank {dist.get_rank()} | Epoch [{epoch}/{config.TRAIN.EPOCHS}]',
              total=len(data_loader), unit='batch') as pbar:
        for idx, data in enumerate(data_loader):
            if config.DATA.ADD_META:
                images,target,meta = data
                meta = [m.float() for m in meta]
                meta = torch.stack(meta,dim=0)
                if mask_meta:
                    meta = torch.zeros_like(meta)
                meta = meta.cuda(non_blocking=True)
            else:
                images, target = data
                meta = None

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            if config.DATA.ADD_META:
                output = model(images, meta)
            else:
                output = model(images)

            # measure accuracy and record loss
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, config.MODEL.NUM_CLASSES)))

            metric.update(output, target)

            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

            pbar.update()
            pbar.set_postfix_str(f'Memory {memory_used:.0f}MB')

    epoch_metric = metric.compute()  # TODO fix issue where each sample is evaluated twice due to distribution in 2 gpus (only in eval mode)

    if tb_logger is not None:
        step = epoch
        tb_logger.add_scalar('val/loss', loss_meter.avg, global_step=step)
        tb_logger.add_scalars('val/metrics', epoch_metric, global_step=step)

    if config.EVAL_MODE:
        class_names = data_loader.dataset.classes
        stats = get_stats(metric, class_names, config.OUTPUT, save=True)
        log_metrics(logger, epoch_metric, 'test')
        logger.info(f"Statistics per class:\n{stats}")
        plot_confusion_matrix(metric, class_names, config.OUTPUT, save=True)

    metric.reset()  # Do not accumulate over epochs

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


def get_stats(metrics, class_names, output_dir, save=True):
    """
    Get and save per class statistics
    Args:
        metrics: torchmetrics collection, has to have a `StatScores` metric
        class_names: List of class names
        output_dir: Output directory path
        save: Whether to save as a csv file

    Returns: Statistics data frame
    """
    stats = metrics['StatScores']
    tp, fp, tn, fn = stats.tp.cpu().numpy(), stats.fp.cpu().numpy(), stats.tn.cpu().numpy(), stats.fn.cpu().numpy(),

    stats_data = {'TP': tp,
                  'FP': fp,
                  'TN': tn,
                  'FN': fn,
                  'Precision': tp / (tp + fp),
                  'Recall': tp / (tp + fn),
                  'F1': 2*tp / (2*tp + fp + fn),
                  'Total samples': tp + fn}
    stats = pd.DataFrame(data=stats_data, index=class_names).fillna(0)

    if save:
        csv = Path(output_dir)/'eval_stats.csv'
        stats.to_csv(csv)

    return stats


def plot_confusion_matrix(metrics, class_names, output_dir, save=True):
    """
    Render and save confusion matrix
    Args:
        metrics: torchmetrics collection, has to have a `ConfusionMatrix` metric
        output_dir: Output directory path
        save: save: Whether to save as a csv file
    """
    matrix = metrics['ConfusionMatrix'].confmat.cpu().numpy()
    matrix_display = ConfusionMatrixDisplay(matrix, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(40, 40))
    matrix_display.plot(xticks_rotation='vertical', colorbar=False, ax=ax)

    if save:
        confusion_matrix = Path(output_dir) / 'confusion_matrix.png'
        plt.savefig(confusion_matrix)

    plt.show()


def log_metrics(logger: logging.Logger, metrics: torchmetrics.Metric, aggregation: str):
    """
    Simple function to log classification metrics at batch or epoch level
    Args:
        logger:
        metrics:
        aggregation:

    Returns:

    """
    metrics_string = ' | '.join(f'{m} = {v}' for m, v in metrics.items() if m not in ['StatScores', 'ConfusionMatrix'])
    logger.info(f'{aggregation.title()} metrics:\n\t{metrics_string}')


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if args.ignore_user_warnings:
        warnings.filterwarnings('ignore', category=UserWarning)

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}",
                           local_rank=config.LOCAL_RANK)

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
