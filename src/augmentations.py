"""训练 batch 的 MixUp/CutMix；标签权重与实际图片变换一致。"""
import math

import numpy as np
import torch


def mixup_batch(images, targets, alpha):
    """每批抽取 Beta(alpha, alpha)；返回混图、两份标签和实际 lambda。"""
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("MixUp alpha 必须是有限正数")
    if images.shape[0] == 0 or targets.ndim != 1 or images.shape[0] != targets.shape[0]:
        raise ValueError("MixUp 需要非空、样本数相同的图片与一维类别标签")
    # NumPy 状态已纳入 checkpoint，避免额外消耗裁剪/翻转和 Dropout 的 Torch 随机流。
    lam = float(np.random.beta(alpha, alpha))
    index = torch.as_tensor(np.random.permutation(images.shape[0]), device=images.device, dtype=torch.long)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, targets, targets[index.to(targets.device)], lam


def cutmix_batch(images, targets, alpha):
    """每批共享矩形；按边界裁剪后的真实替换面积修正标签权重。"""
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("CutMix alpha 必须是有限正数")
    if (images.ndim != 4 or images.shape[0] == 0 or min(images.shape[-2:]) < 1
            or targets.ndim != 1 or images.shape[0] != targets.shape[0]):
        raise ValueError("CutMix 需要非空 NCHW 图片与相同样本数的一维类别标签")
    height, width = images.shape[-2:]
    # 顺序固定并由 checkpoint 的 NumPy 状态恢复；不额外消耗 Torch 随机流。
    raw = float(np.random.beta(alpha, alpha))
    index = torch.as_tensor(np.random.permutation(images.shape[0]), device=images.device, dtype=torch.long)
    cx, cy = int(np.random.randint(width)), int(np.random.randint(height))
    ratio = math.sqrt(1.0 - raw)
    cut_w, cut_h = int(width * ratio), int(height * ratio)
    left, top = cx - cut_w // 2, cy - cut_h // 2
    x1, x2 = max(0, min(width, left)), max(0, min(width, left + cut_w))
    y1, y2 = max(0, min(height, top)), max(0, min(height, top + cut_h))
    area = (x2 - x1) * (y2 - y1)
    effective = 1.0 - area / (width * height)
    mixed = images.clone()
    # RHS 来自未修改的原 batch，避免配对来源在复制过程中被污染。
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    meta = {"lambda_raw": raw, "lambda_eff": effective,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "actual_area": area}
    return mixed, targets, targets[index.to(targets.device)], meta


def mixed_target_cross_entropy(logits, targets_a, targets_b, lam):
    """与混合 one-hot 软标签交叉熵等价；logits 不预先做 softmax。"""
    if not math.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("混合标签 lambda 必须位于 0–1")
    return (lam * torch.nn.functional.cross_entropy(logits, targets_a)
            + (1.0 - lam) * torch.nn.functional.cross_entropy(logits, targets_b))


def mixup_cross_entropy(logits, targets_a, targets_b, lam):
    """保留 E3 旧接口；同一计算也供 CutMix 的实际面积标签使用。"""
    return mixed_target_cross_entropy(logits, targets_a, targets_b, lam)
