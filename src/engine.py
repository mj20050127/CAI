"""一次训练 / 验证：每个 batch 的平均损失按实际样本数加权。"""
import math
import os
from pathlib import Path

import torch
from torch import nn
import numpy as np

from src.augmentations import cutmix_batch, mixed_target_cross_entropy, mixup_batch, mixup_cross_entropy
from src.common import normalize_label_smoothing


def train_one_epoch(model, loader, optimizer, scaler, device, amp=False, log_interval=50,
                    mixup_alpha=0.0, mixup_probability=0.0,
                    cutmix_alpha=0.0, cutmix_probability=0.0, label_smoothing=0.0):
    for value in (mixup_alpha, mixup_probability):
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("MixUp alpha/probability 必须为有限非负数")
    if mixup_probability > 1 or (mixup_alpha == 0) != (mixup_probability == 0):
        raise ValueError("MixUp alpha/probability 设置不一致")
    use_mixup = mixup_alpha > 0
    for value in (cutmix_alpha, cutmix_probability):
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("CutMix alpha/probability 必须为有限非负数")
    if cutmix_probability > 1 or (cutmix_alpha == 0) != (cutmix_probability == 0):
        raise ValueError("CutMix alpha/probability 设置不一致")
    use_cutmix = cutmix_alpha > 0
    if use_mixup and use_cutmix:
        raise ValueError("MixUp 与 CutMix 必须互斥")
    label_smoothing = normalize_label_smoothing(label_smoothing)
    if label_smoothing and (use_mixup or use_cutmix):
        raise ValueError("标签平滑单项实验不能与 MixUp 或 CutMix 同时启用")
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss, total, correct = 0.0, 0, 0
    trace = []
    cutmix_trace = []
    for step, (images, targets) in enumerate(loader, 1):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        applied = use_mixup and (mixup_probability == 1 or np.random.random() < mixup_probability)
        cutmix_applied = use_cutmix and (cutmix_probability == 1 or np.random.random() < cutmix_probability)
        targets_a, targets_b, lam = targets, targets, 1.0
        if applied:
            images, targets_a, targets_b, lam = mixup_batch(images, targets, mixup_alpha)
        if use_mixup:
            trace.append({"batch": step, "lambda": lam, "applied": bool(applied)})
        if use_cutmix:
            meta = {"lambda_raw": 1.0, "lambda_eff": 1.0,
                    "x1": 0, "y1": 0, "x2": 0, "y2": 0, "actual_area": 0}
            if cutmix_applied:
                images, targets_a, targets_b, meta = cutmix_batch(images, targets, cutmix_alpha)
                lam = meta["lambda_eff"]
            cutmix_trace.append({"batch": step, **meta, "applied": bool(cutmix_applied)})
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            if cutmix_applied:
                loss = mixed_target_cross_entropy(logits, targets_a, targets_b, lam)
            else:
                loss = (mixup_cross_entropy(logits, targets_a, targets_b, lam)
                        if applied else criterion(logits, targets))
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError("训练损失非有限值，请检查输入与学习率。")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += value * targets.size(0)
        total += targets.size(0)
        predicted = logits.argmax(1)
        if applied or cutmix_applied:
            correct += (lam * (predicted == targets_a).sum().item()
                        + (1.0 - lam) * (predicted == targets_b).sum().item())
        else:
            correct += (predicted == targets).sum().item()
        if step % log_interval == 0 or step == len(loader):
            print(f"  batch {step}/{len(loader)} objective={total_loss / total:.4f}", flush=True)
    if total == 0:
        raise ValueError("训练集不能为空")
    result = {"loss": total_loss / total, "accuracy": correct / total}
    if use_mixup:
        values = [row["lambda"] for row in trace if row["applied"]]
        result.update(mixup_trace=trace, mixup_batches=len(values), mixup_total_batches=len(trace),
                      mixup_lambda_mean=float(np.mean(values)) if values else 0.0,
                      mixup_lambda_min=min(values) if values else 0.0,
                      mixup_lambda_max=max(values) if values else 0.0)
    if use_cutmix:
        values = [row["lambda_eff"] for row in cutmix_trace if row["applied"]]
        result.update(cutmix_trace=cutmix_trace, cutmix_batches=len(values),
                      cutmix_nonzero_batches=sum(row["actual_area"] > 0 for row in cutmix_trace if row["applied"]),
                      cutmix_total_batches=len(cutmix_trace),
                      cutmix_lambda_mean=float(np.mean(values)) if values else 0.0,
                      cutmix_lambda_min=min(values) if values else 0.0,
                      cutmix_lambda_max=max(values) if values else 0.0)
    return result


@torch.inference_mode()
def evaluate_model(model, loader, device, predictions=False):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total, correct = 0.0, 0, 0
    truth, predicted, confidence = [], [], []
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, targets).item()
        labels = logits.argmax(1)
        total += targets.size(0)
        correct += (labels == targets).sum().item()
        if predictions:
            truth.extend(targets.cpu().tolist())
            predicted.extend(labels.cpu().tolist())
            confidence.extend(logits.softmax(1).max(1).values.cpu().tolist())
    if total == 0:
        raise ValueError("评估集不能为空")
    result = {"loss": total_loss / total, "accuracy": correct / total, "sample_count": total}
    if predictions:
        result.update(targets=truth, predictions=predicted, confidences=confidence)
    return result


def save_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)
