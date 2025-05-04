# Copyright (c) 2015-present, Facebook, Inc. All rights reserved.
import os, subprocess, time
from datetime import datetime, timedelta
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from filelock import FileLock, Timeout


def extract_tar_system(tar_path, target_dir):
    """Extract tar file safely using file lock to avoid concurrent extractions."""
    flag_file = os.path.join(target_dir, "extraction_complete.flag")
    lock_file = os.path.join(target_dir, "extraction.lock")

    if os.path.exists(flag_file):
        return target_dir

    os.makedirs(target_dir, exist_ok=True)
    lock = FileLock(lock_file, timeout=900)
    try:
        with lock:
            if os.path.exists(flag_file):
                return target_dir
            subprocess.run([
                "tar", "--transform", "s/\\\\/\\//g", "-xf", tar_path, "-C", target_dir
            ], check=True)
            with open(flag_file, "w") as f:
                f.write("done")
    except Timeout:
        raise Exception("Timeout acquiring extraction lock.")
    return target_dir


def extract_if_needed(tar_path, final_dir, args):
    """Ensure only local_rank==0 extracts tar file; others wait until done."""
    flag_file = os.path.join(final_dir, "extraction_complete.flag")
    if args.local_rank == 0:
        if not os.path.exists(flag_file):
            extract_tar_system(tar_path, final_dir)
    else:
        while not os.path.exists(flag_file):
            time.sleep(5)
    return final_dir


def build_dataset(is_train, args):
    """Build dataset from torchvision or local ImageNet folders/tars."""
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR10':
        download_flag = os.path.join(args.data_path, "cifar_download_complete.flag")
        if args.local_rank == 0:
            datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
            with open(download_flag, "w") as f:
                f.write("done")
        else:
            while not os.path.exists(download_flag):
                time.sleep(1)
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=False)
        nb_classes = 10

    elif args.data_set == 'CIFAR100':
        download_flag = os.path.join(args.data_path, "cifar_download_complete.flag")
        if args.local_rank == 0:
            datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
            with open(download_flag, "w") as f:
                f.write("done")
        else:
            while not os.path.exists(download_flag):
                time.sleep(1)
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=False)
        nb_classes = 100

    elif args.data_set == 'IMNET':
        base_data_path = args.data_path
        os.makedirs(base_data_path, exist_ok=True)
        prefix = 'train' if is_train else 'val'
        tar_path = os.path.join(base_data_path, f"{prefix}.tar")
        final_dir = os.path.join(base_data_path, prefix)

        if os.path.exists(tar_path):
            extract_if_needed(tar_path, final_dir, args)
            dataset = ImageFolder(final_dir, transform=transform)
        elif os.path.exists(final_dir):
            dataset = ImageFolder(final_dir, transform=transform)
        else:
            raise FileNotFoundError(f"Neither {tar_path} nor {final_dir} found.")

        nb_classes = 1000

    return dataset, nb_classes


def build_transform(is_train, args):
    """Create standard train/test transforms including resize and normalization."""
    resize_im = args.input_size > 32
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(transforms.Resize(size, interpolation=3))
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
