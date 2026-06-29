import logging
import os
import time
import argparse
import contextlib
import datetime
from pathlib import Path

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter, accuracy

from callbacks import EarlyStopper
from config import get_config
from metrics import get_model_metrics, get_stats, log_metrics, dump_summary
from models.build import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, load_pretained
from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save_csv memory")
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

    parser.add_argument('--sampler', type=str, default=None, choices=('weighted',), help='Type of training sampler')

    parser.add_argument('--dataset', type=str,
                        help='dataset', default='bugbox')
    parser.add_argument('--lr-scheduler-name', type=str,
                        help='lr scheduler name,cosin linear,step')

    parser.add_argument('--pretrain', type=str,
                        help='pretrain')

    parser.add_argument('--version', type=str, help='Version to tag trained model')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    if config.EVAL_MODE:
        logger.info("Running in eval mode")
        if config.MODEL.PRETRAINED:
            logger.info(f"Loading pretrained model from {config.MODEL.PRETRAINED}")
        else:
            raise ValueError("Pretrained model path needs to be specified when running in eval mode")
        dataset_test, data_loader_test = build_loader(config)
        logger.info("Test class mapping:", dataset_test.class_to_idx)
    else:
        # The data needs to be loaded before the model is created to fill the num_classes field
        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model: {config.MODEL.TYPE}-{config.MODEL.NAME}/{config.TAG}/{config.VERSION}")
    model = build_model(config)
    model = model.to(memory_format=torch.channels_last)

    model.cuda()

    optimizer = build_optimizer(config, model)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[int(os.environ["LOCAL_RANK"])], broadcast_buffers=False)

    model_without_ddp = model.module

    model = torch.compile(model, dynamic=False)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Number of trainable parameters: {n_parameters}")

    if config.EVAL_MODE:
        load_pretained(config, model_without_ddp, logger)
        test(config, data_loader_test, model)
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
        acc1, acc5, loss = validate(config, data_loader_val, model, config.TRAIN.START_EPOCH)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} validation images: {acc1:.1f}%")
        if config.DATA.ADD_META:
            logger.info("**********mask meta test***********")
            acc1, acc5, loss = validate(config, data_loader_val, model, config.TRAIN.START_EPOCH, mask_meta=True)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    # Only initialize TensorBoard SummaryWriter on Rank 0
    if dist.get_rank() == 0:
        tb_dir = Path(config.OUTPUT) / 'tensorboard'
        tb_dir.mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(log_dir=tb_dir)
    else:
        writer = None

    # Early stopping
    stopper = EarlyStopper(patience=config.TRAIN.EARLY_STOP.PATIENCE, min_delta=config.TRAIN.EARLY_STOP.MIN_DELTA)

    start_time = time.time()

    range_end = config.TRAIN.START_EPOCH + config.TRAIN.EPOCHS
    logger.info(f"Starting training loop from epoch {config.TRAIN.START_EPOCH} to {range_end}")

    for epoch in range(config.TRAIN.START_EPOCH, range_end):
        data_loader_train.sampler.set_epoch(epoch)

        # Train
        train_one_epoch_local_data(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn,
                                   lr_scheduler, writer)

        # Validate
        if config.DATA.ADD_META:
            acc1, acc5, loss = validate(config, data_loader_val, model, epoch, mask_meta=True, tb_logger=writer)
        else:
            acc1, acc5, loss = validate(config, data_loader_val, model, epoch, tb_logger=writer)

        if acc1 > max_accuracy:
            max_accuracy = acc1
            logger.info(f"--> New maximum validation accuracy achieved: {max_accuracy:.3f}%")
            # Save best checkpoint
            if dist.get_rank() == 0:
                logger.info("Saving checkpoint (best)...")
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, 'best')

        if dist.get_rank() == 0:
            # Save periodic checkpoint
            if epoch % config.SAVE_FREQ == 0:
                logger.info(f"Saving periodic checkpoint at epoch {epoch}...")
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, f'epoch_{epoch}')
            # Save latest checkpoint
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, 'latest')

        if epoch > config.TRAIN.EARLY_STOP.MIN_EPOCHS and stopper.early_stop(acc1):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Total Training Time: {}'.format(total_time_str))


def train_one_epoch_local_data(
        config, model, criterion, data_loader, optimizer,
        epoch, mixup_fn, lr_scheduler, tb_logger=None, log_freq=500):
    model.train()
    if hasattr(model.module, 'cur_epoch'):
        model.module.cur_epoch = epoch
        model.module.total_epoch = config.TRAIN.EPOCHS
    optimizer.zero_grad()

    num_steps = len(data_loader)
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start_time = time.time()

    # Calculate the base step for the LR scheduler
    steps_per_epoch = num_steps // config.TRAIN.ACCUMULATION_STEPS
    if num_steps % config.TRAIN.ACCUMULATION_STEPS != 0:
        steps_per_epoch += 1
    optimizer_step = epoch * steps_per_epoch

    for idx, data in enumerate(data_loader):
        if idx == 1 and dist.get_rank() == 0:
            logger.info(f"Time to load the first batch: {time.time() - start_time:.4f} seconds")

        if config.DATA.ADD_META:
            samples, targets, meta = data
            meta = [m.to(dtype=torch.bfloat16) for m in meta]
            meta = torch.stack(meta, dim=0)
            meta = meta.cuda(non_blocking=True)
        else:
            samples, targets = data
            meta = None

        samples = samples.cuda(non_blocking=True)
        samples = samples.to(memory_format=torch.channels_last)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        is_accum_boundary = (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0
        is_final_batch = (idx + 1) == num_steps
        should_step = is_accum_boundary or is_final_batch

        # Prevent DDP gradient synchronization unless we are stepping the optimizer
        require_sync = should_step or not isinstance(model, torch.nn.parallel.DistributedDataParallel)
        sync_context = contextlib.nullcontext() if require_sync else model.no_sync()

        with sync_context:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if config.DATA.ADD_META:
                    outputs = model(samples, meta)
                else:
                    outputs = model(samples)
                loss = criterion(outputs, targets)

            if config.TRAIN.ACCUMULATION_STEPS > 1:
                loss = loss / config.TRAIN.ACCUMULATION_STEPS

            loss.backward()

        if should_step:
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step_update(optimizer_step)

            if dist.get_rank() == 0 and tb_logger is not None:
                tb_logger.add_scalar('train/batch_grad_norm', grad_norm, optimizer_step)

            norm_meter.update(grad_norm)
            optimizer_step += 1
        else:
            grad_norm = torch.tensor(0.0)

        torch.cuda.synchronize()

        loss_val = loss.item() * config.TRAIN.ACCUMULATION_STEPS if config.TRAIN.ACCUMULATION_STEPS > 1 else loss.item()
        loss_meter.update(loss_val, targets.size(0))

        if idx % log_freq == 0 and dist.get_rank() == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Train Epoch: [{epoch}][{idx}/{num_steps}]  "
                f"Batch Loss: {loss_val:.4f} (Avg: {loss_meter.avg:.4f})  "
                f"Grad Norm (Avg): {norm_meter.avg:.4f}  "
                f"LR: {lr:.6f}  "
                f"Mem: {memory_used:.0f}MB"
            )

            if tb_logger is not None:
                # Use the actual optimizer step for Tensorboard logging for consistency
                tb_logger.add_scalar('train/batch_loss', loss_val, optimizer_step)

    # Epoch-level Summary for TensorBoard
    if dist.get_rank() == 0 and tb_logger is not None:
        tb_logger.add_scalar('train/epoch_loss', loss_meter.avg, global_step=epoch)
        tb_logger.add_scalar('train/epoch_grad_norm', norm_meter.avg, global_step=epoch)
        tb_logger.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step=epoch)


@torch.no_grad()
def validate(config, data_loader, model, epoch, mask_meta=False, tb_logger=None):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    for idx, data in enumerate(data_loader):
        if config.DATA.ADD_META:
            images, target, meta = data
            meta = [m.to(dtype=torch.bfloat16) for m in meta]
            torch.stack(meta, dim=0)
            if mask_meta:
                meta = torch.zeros_like(meta)
            meta = meta.cuda(non_blocking=True)
        else:
            images, target = data
            meta = None

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if config.DATA.ADD_META:
                output = model(images, meta)
            else:
                output = model(images)

            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, config.MODEL.NUM_CLASSES)))

            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

    if dist.get_rank() == 0:
        logger.info(
            f"=== Validation Epoch [{epoch}] Summary ===\n"
            f"Loss: {loss_meter.avg:.4f} | Acc@1: {acc1_meter.avg:.3f}% | Acc@5: {acc5_meter.avg:.3f}%"
        )
        if tb_logger is not None:
            tb_logger.add_scalar('val/acc1', acc1_meter.avg, global_step=epoch)
            tb_logger.add_scalar('val/acc5', acc5_meter.avg, global_step=epoch)
            tb_logger.add_scalar('val/loss', loss_meter.avg, global_step=epoch)

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def test(config, data_loader, model):
    model.eval()
    metrics = get_model_metrics(config)
    metrics = metrics.to(torch.device(int(os.environ["LOCAL_RANK"])))

    for idx, data in enumerate(data_loader):
        if config.DATA.ADD_META:
            images, target, meta = data
            meta = [m.to(dtype=torch.bfloat16) for m in meta]
            meta = torch.stack(meta, dim=0)
            meta = meta.cuda(non_blocking=True)
        else:
            images, target = data
            meta = None

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if config.DATA.ADD_META:
                output = model(images, meta)
            else:
                output = model(images)

            metrics.update(output, target)

    epoch_metric = metrics.compute()

    display = 3
    logger.info(f"First class entries are:\n{list(config.DATA.CLASS_NAMES)[:display]}")
    stats = get_stats(metrics, list(config.DATA.CLASS_NAMES), Path(config.OUTPUT), config.VERSION, save_csv=True)
    print('stats' + str(len(stats)))
    log_metrics(logger, epoch_metric, 'test')
    logger.info(f"Statistics per class:\n{stats}")
    dump_summary(epoch_metric, config, dump=True)

    metrics.reset()
    return


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


def setup_distributed(config):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    if rank == 0:
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout=datetime.timedelta(seconds=900),
        world_size=world_size,
        rank=rank)
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True


if __name__ == '__main__':
    args, config = parse_option()
    logging.basicConfig(level=logging.INFO)
    torch._dynamo.config.cache_size_limit = 64
    setup_distributed(config)

    if dist.get_rank() == 0:
        print(f"Output directory path: {config.OUTPUT}")

    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0

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
                           local_rank=int(os.environ["LOCAL_RANK"]))

    main(config)

    if dist.get_rank() == 0 and not config.EVAL_MODE:
        path = os.path.join(config.OUTPUT, "config.yaml")
        with open(path, "w") as f:
            f.write(config.dump())

    dist.barrier()
    dist.destroy_process_group()
