import argparse
import datetime
import numpy as np
import random
import time
import torch
import torch.backends.cudnn as cudnn
import json
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, ApexScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate, prune_with_Taylor, finetune_one_epoch
from losses import DistillationLoss
from samplers import RASampler
import models
import utils

from torch.utils.tensorboard import SummaryWriter

from pruning_core.pruning_utils import prepare_logging, get_lr, lr_cosine_policy, initialize_pruning_engine
from model_pruning import create_pruning_structure_vit, enable_pruning

from cskd.cskd import CSKDLoss
from cskd.config import ConfigBase
from configs.cskd_config import Config

#from apex.contrib.sparsity import ASP
from einops import rearrange

#try:
#    from apex import amp
#    from apex.parallel import DistributedDataParallel as ApexDDP
#    from apex.parallel import convert_syncbn_model

#    has_apex = True
#except ImportError:
has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

def str2bool(v):
    # from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='use pretrained model')
    parser.add_argument('--scratch', action='store_true', default=False,
                        help='use pretrained model')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10', 'CIFAR100', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--asp', action='store_true', default=False,
                        help='Apply ampere sparsity before finetuning')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument(
        "--hidden_loss_coeff",
        type=float,
        default=3e-2,
        required=False,
        help=
        "loss for cross feature map regularization on layernorms"
    )

    parser.add_argument(
        "--original_loss_coeff",
        type=float,
        default=0.0,
        required=False,
        help=
        "original QA loss for pruning"
    )

    parser.add_argument(
        "--kl_loss_coeff",
        type=float,
        default=1000.0,
        required=False,
        help=
        "KL loss for pruning"
    )

    parser.add_argument(
    '--student-path',
    default='',
    type=str,
    help='path to local weights'
    )
    
    return parser

def ce_train_one_epoch(model, loader, criterion, optimizer, device, scaler, mixup_fn=None, clip_grad=None,
set_training_mode=True):
    model.train(set_training_mode)
    if not set_training_mode:
        model.head.train()
        model.head_dist.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        clean_images = images.clone()
        clean_targets = targets.clone()

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            raw_out = model(images)
            outputs = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out
            loss = criterion(outputs, targets)
        if scaler is not None:
            scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                parameters=model.parameters()
            )
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)

        with torch.no_grad():
            raw_out_clean = model(clean_images)
            outputs_clean = raw_out_clean[0] if isinstance(raw_out_clean, (tuple, list)) else raw_out_clean
            preds = outputs_clean.argmax(dim=1)
            correct += preds.eq(clean_targets).sum().item()
            total += images.size(0)
    return {"loss": running_loss / total, "acc1": 100.0 * correct / total}


def main(args):
    # 1. Initialize distributed training (if enabled)
    utils.init_distributed_mode(args)
    print(args)

    # 2. Checkpoint / output directory setup
    if args.finetune:
        args.checkpoint = Path(args.finetune) / "ft_checkpoint.pth"
    else:
        raise ValueError("--finetune <run_dir> is required")

    tag = "scratch" if args.scratch else "ft_dense"
    args.output_dir = Path(args.finetune) / f"{tag}_{args.epochs}_lr_{args.lr}_a_{args.distillation_alpha}_T_{args.distillation_tau}"
    if utils.is_main_process():
        writer = SummaryWriter(str(args.output_dir))

    # 3. Device & seed
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    # 4. Datasets & samplers
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Eval dataset not divisible by number of processes; results may vary.')
            sampler_val = torch.utils.data.RandomSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size), num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    args.iters_per_epoch = len(data_loader_train)
    args.train_iters = args.epochs * args.iters_per_epoch

    # 5. Mixup / CutMix
    mixup_active = args.mixup > 0 or args.cutmix > 0 or args.cutmix_minmax is not None
    mixup_fn = Mixup(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        label_smoothing=args.smoothing, num_classes=args.nb_classes
    ) if mixup_active else None
    if not mixup_active:
        print('No mixup/cutmix')

    # 6. Load dense checkpoint & rebuild model
    assert args.checkpoint, "Pass --checkpoint /path/to/ft_checkpoint.pth"
    if utils.is_main_process():
        print(f"Loading dense checkpoint: {args.checkpoint}")
    ckpt = torch.load(str(args.checkpoint), map_location='cpu')
    dim_dict = ckpt['dim']

    dense_model = create_model(
        args.model + '_small', pretrained=False, num_classes=args.nb_classes,
        EMB=dim_dict['EMB'], QK=dim_dict['QK'], V=dim_dict['V'],
        MLP=dim_dict['MLP'], head=dim_dict['head'],
        drop_rate=args.drop, drop_path_rate=args.drop_path,
    )
    state_dict = ckpt['model']
    for key in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if key in state_dict and state_dict[key].shape[0] != args.nb_classes:
            del state_dict[key]
    missing, unexpected = dense_model.load_state_dict(state_dict, strict=False)
    if utils.is_main_process():
        print('state-dict loaded | missing:', missing, '| unexpected:', unexpected)

    # Re-initialize classifier heads
    in_dim = dense_model.head.in_features
    dense_model.head = torch.nn.Linear(in_dim, args.nb_classes)
    dense_model.head_dist = torch.nn.Linear(in_dim, args.nb_classes)
    torch.nn.init.trunc_normal_(dense_model.head.weight, std=0.02)
    torch.nn.init.trunc_normal_(dense_model.head_dist.weight, std=0.02)

    dense_model.to(device)
    if args.distributed:
        dense_model = NativeDDP(dense_model, device_ids=[args.gpu])
    dense_model_without_ddp = dense_model.module if args.distributed else dense_model

    dense_model_without_ddp.eval()
    dense_model_without_ddp.head.train()
    dense_model_without_ddp.head_dist.train()
    # # after you rebuild dense_model and replace its heads…
    # for name, p in dense_model_without_ddp.named_parameters():
    #     if "head" not in name:
    #         p.requires_grad = False


    # Placeholder for EMA model (optional)
    dense_model_ema = None

    # 7. AMP / optimizer / scheduler / criterion
    use_amp = None
    if args.amp:
        if has_apex:
            args.apex_amp = True
        elif has_native_amp:
            args.native_amp = True

    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        print('Neither APEX nor native Torch AMP found – training in fp32.')

    args.lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    optimizer = create_optimizer(args, dense_model)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.mixup > 0:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print('No label smoothing')

    scaler = None
    if use_amp == 'native':
        scaler = NativeScaler()
    elif use_amp == 'apex':
        dense_model, optimizer = amp.initialize(dense_model, optimizer, opt_level='O1')
        scaler = ApexScaler()

    # 8. Count parameters
    n_parameters = sum(p.numel() for p in dense_model.parameters() if p.requires_grad)
    if utils.is_main_process():
        print(f'#params (trainable): {n_parameters/1e6:.2f} M')

    # 9. Resume from checkpoint (optional)
    output_dir = Path(args.output_dir)
    if args.resume:
        resume_ckpt = torch.load(str(args.resume), map_location='cpu')
        dense_model_without_ddp.load_state_dict(resume_ckpt['model'])
        if not args.eval and all(k in resume_ckpt for k in ['optimizer', 'lr_scheduler', 'epoch']):
            optimizer.load_state_dict(resume_ckpt['optimizer'])
            lr_scheduler.load_state_dict(resume_ckpt['lr_scheduler'])
            total_epoch = resume_ckpt['epoch'] + 1
            if args.model_ema and dense_model_ema is not None:
                utils._load_checkpoint_for_ema(dense_model_ema, resume_ckpt['model_ema'])
            if 'scaler' in resume_ckpt and scaler is not None:
                scaler.load_state_dict(resume_ckpt['scaler'])
        else:
            total_epoch = 0
    else:
        total_epoch = 0

    # 10. Eval-only mode
    if args.eval:
        test_stats = evaluate(data_loader_val, dense_model_without_ddp, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    # 11. Initial validation
    max_accuracy = 0.0
    test_stats = evaluate(data_loader_val, dense_model_without_ddp, device)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    max_accuracy = max(max_accuracy, test_stats['acc1'])
    print(f"Max accuracy: {max_accuracy:.2f}%")

    if utils.is_main_process():
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': total_epoch, 'n_parameters': n_parameters}
        for key, val in log_stats.items():
            if 'epoch' not in key:
                writer.add_scalar(key, val, total_epoch)

    # 12. Main training loop
    print(f"Start training for {args.epochs} epochs")
    set_training_mode = False
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        lr_scheduler.step(epoch)

        train_stats = ce_train_one_epoch(
            dense_model_without_ddp,
            data_loader_train,
            criterion,
            optimizer,
            device,
            scaler,
            mixup_fn,
            args.clip_grad,
            set_training_mode=set_training_mode
        )
        total_epoch += 1

        print(
            f"Train: [{epoch:3d}/{args.epochs}]  "
            f"loss: {train_stats['loss']:.4f}  "
            f"acc:  {train_stats['acc1']:.2f}%"
        )

        # checkpoint
        if args.output_dir:
            ckpt_dict = {
                'model': dense_model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': total_epoch,
                'model_ema': utils.get_state_dict(dense_model_ema) if dense_model_ema is not None else None,
                'scaler': scaler.state_dict() if scaler is not None else None,
                'args': args,
                'dim': dim_dict,
            }
            utils.save_on_master(ckpt_dict, output_dir / 'ckpt_last.pth')

        # validation
        test_stats = evaluate(data_loader_val, dense_model_without_ddp, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats['acc1'])
        print(f"Max accuracy: {max_accuracy:.2f}%")

        # logging
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch+1, 'n_parameters': n_parameters}
        if utils.is_main_process():
            for key, val in log_stats.items():
                if 'epoch' not in key:
                    writer.add_scalar(key, val, total_epoch)
            with (output_dir / 'log.txt').open('a') as f:
                f.write(json.dumps(log_stats) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
