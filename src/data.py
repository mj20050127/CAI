"""固定分层划分；只用训练图片计算归一化；下载由 prepare_data 显式触发。"""
import hashlib
import random
from pathlib import Path

import numpy as np

from src.common import digest, read_json, write_json


def stratified_split(targets, validation_per_class=500, monitor_per_class=500,
                     split_seed=2026, monitor_seed=2027):
    targets = np.asarray(targets)
    if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("targets 必须是一维整数标签")
    if validation_per_class < 1 or monitor_per_class < 1:
        raise ValueError("每类抽取数必须为正")
    train, validation, monitor = [], [], []
    split_rng, monitor_rng = np.random.default_rng(split_seed), np.random.default_rng(monitor_seed)
    for label in np.unique(targets):
        indices = split_rng.permutation(np.flatnonzero(targets == label))
        if len(indices) < validation_per_class + monitor_per_class:
            raise ValueError("该类别没有足够的训练/验证样本")
        validation.extend(indices[:validation_per_class].tolist())
        selected = indices[validation_per_class:]
        train.extend(selected.tolist())
        monitor.extend(monitor_rng.permutation(selected)[:monitor_per_class].tolist())
    return {"train": sorted(train), "validation": sorted(validation), "monitor": sorted(monitor)}


def channel_statistics(images, indices):
    """分块累计像素，避免把整个训练集一次性转成 float64。"""
    if len(indices) == 0:
        raise ValueError("统计样本不能为空")
    total, squares = np.zeros(3), np.zeros(3)
    count = 0
    for start in range(0, len(indices), 512):
        pixels = images[indices[start:start + 512]].astype(np.float64) / 255.0
        total += pixels.sum(axis=(0, 1, 2))
        squares += np.square(pixels).sum(axis=(0, 1, 2))
        count += pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
    if np.any(std <= 0):
        raise ValueError("存在零方差通道，无法归一化")
    return mean.tolist(), std.tolist()


def dataset_identity(dataset):
    hasher = hashlib.sha256()
    hasher.update(dataset.data.tobytes())
    hasher.update(np.asarray(dataset.targets, dtype=np.int64).tobytes())
    return hasher.hexdigest()


def cifar10(data_dir, train=True, download=False, transform=None):
    # 延迟导入：数据划分的数学测试不需要安装 torchvision。
    from torchvision.datasets import CIFAR10
    return CIFAR10(root=data_dir, train=train, download=download, transform=transform)


def prepare(config, download=False):
    raw = cifar10(config["data_dir"], download=download)
    counts = np.bincount(raw.targets, minlength=10)
    if len(raw) != 50000 or counts.tolist() != [5000] * 10:
        raise ValueError("预期官方 CIFAR-10 训练集，每类 5,000 张")
    parts = stratified_split(raw.targets, split_seed=config["split_seed"], monitor_seed=config["monitor_seed"])
    mean, std = channel_statistics(raw.data, parts["train"])
    manifest = {"version": 1, "dataset_sha256": dataset_identity(raw),
                "split_seed": config["split_seed"], "monitor_seed": config["monitor_seed"],
                "classes": raw.classes, "mean": mean, "std": std, "indices": parts}
    path = Path(config["data_dir"]) / "split_manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("已有划分与本配置不一致，请使用新的 data_dir，避免覆盖已有实验划分。")
    write_json(path, manifest)
    return manifest


def load_manifest(config):
    path = Path(config["data_dir"]) / "split_manifest.json"
    if not path.exists():
        raise FileNotFoundError("请先运行 prepare_data.py --download。")
    manifest = read_json(path)
    if (manifest["split_seed"], manifest["monitor_seed"]) != (config["split_seed"], config["monitor_seed"]):
        raise ValueError("当前划分种子与已经保存的数据划分不一致")
    raw = cifar10(config["data_dir"])
    if dataset_identity(raw) != manifest["dataset_sha256"]:
        raise ValueError("数据内容与已保存的数据清单不一致")
    expected = stratified_split(raw.targets, split_seed=config["split_seed"], monitor_seed=config["monitor_seed"])
    if manifest["indices"] != expected:
        raise ValueError("数据索引清单被修改，请核查数据准备步骤")
    return manifest


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_RESIZE_SIZE = 256
IMAGENET_CROP_SIZE = 224


def normalization_for(manifest, preprocessing="cifar32"):
    if preprocessing == "cifar32":
        return {k: manifest[k] for k in ("mean", "std", "classes")}
    if preprocessing == "imagenet224":
        return {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "classes": list(manifest["classes"]),
            "preprocessing": "imagenet224",
            "resize_size": IMAGENET_RESIZE_SIZE,
            "crop_size": IMAGENET_CROP_SIZE,
            "interpolation": "bilinear",
        }
    raise ValueError(f"不支持的 preprocessing：{preprocessing}")


def transforms_for(manifest, augment, preprocessing="cifar32"):
    from torchvision import transforms
    if preprocessing == "cifar32":
        steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else []
        return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(manifest["mean"], manifest["std"])])
    if preprocessing == "imagenet224":
        from torchvision.transforms import InterpolationMode
        if augment:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((IMAGENET_CROP_SIZE, IMAGENET_CROP_SIZE),
                                  interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        return transforms.Compose([
            transforms.Resize(IMAGENET_RESIZE_SIZE, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(IMAGENET_CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    raise ValueError(f"不支持的 preprocessing：{preprocessing}")


def seed_worker(worker_id):
    import torch
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def take_per_class(indices, targets, count):
    targets = np.asarray(targets)
    return sorted(i for label in range(10) for i in [j for j in indices if targets[j] == label][:count])


def make_loaders(config, manifest, smoke=False, use_cuda=False):
    import torch
    from torch.utils.data import DataLoader, Subset
    preprocessing = config.get("preprocessing", "cifar32")
    training = cifar10(config["data_dir"],
                       transform=transforms_for(manifest, config.get("train_augmentation", True), preprocessing))
    clean = cifar10(config["data_dir"], transform=transforms_for(manifest, False, preprocessing))
    indices = manifest["indices"].copy()
    if smoke:
        indices["train"] = take_per_class(indices["train"], clean.targets, 100)
        indices["validation"] = take_per_class(indices["validation"], clean.targets, 50)
        indices["monitor"] = take_per_class(indices["train"], clean.targets, 50)
    generators = {name: torch.Generator().manual_seed(config["seed"] + offset)
                  for offset, name in enumerate(indices, 1)}
    loaders = {}
    for name, selected in indices.items():
        dataset = training if name == "train" else clean
        loaders[name] = DataLoader(Subset(dataset, selected), batch_size=config["batch_size"],
                                   shuffle=name == "train", num_workers=config["num_workers"],
                                   pin_memory=use_cuda, worker_init_fn=seed_worker,
                                   generator=generators[name], persistent_workers=False)
    return loaders, generators


def manifest_hash(manifest):
    return digest(manifest)
