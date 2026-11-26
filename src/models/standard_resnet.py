"""E9/E10 共用标准 ResNet-18 结构。ImageNet 权重只在 E10 新训练时加载。"""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet, ResNet18_Weights, resnet18

from src.common import RESNET18_IMAGENET1K_V1_SHA256


def build_standard_resnet18(num_classes: int = 10, pretrained: bool = False) -> ResNet:
    """保留标准 7×7 conv1 与 3×3 maxpool 入口，仅替换 fc 为十分类线性层。"""
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
        raise ValueError("num_classes 必须是正整数。")
    if type(pretrained) is not bool:
        raise ValueError("pretrained 必须是 true 或 false。")
    if pretrained:
        raise ValueError("请通过 E10 训练入口选择 ImageNet 权重和配对分类头。")
    # 保护全局 CPU 随机流位置，使后续 E10 复用分类头初始值时与 E9 拥有一致的增强随机序列。
    with torch.random.fork_rng(devices=[]):
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_imagenet_resnet18(data_dir):
    weights = ResNet18_Weights.IMAGENET1K_V1
    path = (Path(data_dir) / "weights" / weights.url.rsplit("/", 1)[-1]).resolve()
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"正在取回 ResNet-18 ImageNet 权重：{weights.url}", flush=True)
        torch.hub.download_url_to_file(weights.url, path, hash_prefix=RESNET18_IMAGENET1K_V1_SHA256)
    with path.open("rb") as file:
        checksum = hashlib.file_digest(file, "sha256").hexdigest()
        if checksum != RESNET18_IMAGENET1K_V1_SHA256:
            raise ValueError(f"ResNet-18 权重校验失败：{path}")
        file.seek(0)
        state = torch.load(file, map_location="cpu", weights_only=True)
    return state, {"enum": f"ResNet18_Weights.{weights.name}", "url": weights.url,
                   "sha256": checksum, "path": str(path)}
