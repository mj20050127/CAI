"""比较两次已完成、已做验证集评估的运行；不训练，不读取测试集。"""
import argparse
import csv
import hashlib
import math
from itertools import zip_longest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.common import (RESNET18_IMAGENET1K_V1_SHA256, ROOT, digest, normalize_label_smoothing,
                        read_json, write_json)
from src.early_stopping import replay_early_stopping, validate_early_stopping
from src.metrics import classification_metrics


# 这些字段必须相同；消融因素单独校验，只接受明确设计的实验对。
PROTOCOL_FIELDS = (
    "epochs", "batch_size", "learning_rate", "momentum", "weight_decay",
    "min_learning_rate", "dropout", "amp", "seed", "split_seed",
    "monitor_seed", "num_workers", "device",
)
HISTORY_FIELDS = (
    "epoch", "lr", "train_objective", "train_clean_loss", "train_clean_accuracy",
    "val_loss", "val_accuracy", "epoch_seconds",
)
CBAM_MODEL = "cifar_resnet18_cbam"
CBAM_CONFIG = {"cbam_reduction": 16, "cbam_spatial_kernel_size": 7}
CBAM_EXPERIMENTS = {"E5", "E11_stem_cbam"}
TRANSFER_EXPERIMENTS = {"E9", "E10"}
TRANSFER_PROTOCOL = {
    "model": "standard_resnet18", "input_size": 224, "preprocessing": "imagenet224", "optimizer": "adamw",
}
RESIDUAL_TRANSFER_EXPERIMENTS = {"E4", "E11"}
STEM_TRANSFER_BASELINES = {"E4", "E11"}
STEM_CBAM_EXPERIMENTS = {"E11_stem", "E11_stem_cbam"}
CIFAR_TRANSFER_EXPERIMENTS = RESIDUAL_TRANSFER_EXPERIMENTS | STEM_CBAM_EXPERIMENTS
RESIDUAL_TRANSFER_PROTOCOL = {
    "model": "cifar_resnet18", "input_size": 32, "preprocessing": "cifar32", "optimizer": "sgd",
}
CIFAR_TRANSFER_DEFAULTS = {
    "input_size": 32, "preprocessing": "cifar32", "optimizer": "sgd", "pretrained": False,
    "scheduler": "cosine", "label_smoothing": 0.0,
}
TRANSFER_WEIGHTS = "IMAGENET1K_V1"
TRANSFER_WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
STEM_TRANSFER_SCOPE = "stem-and-layer1-layer4"
STEM_KERNEL_ADAPTATION = "center_crop_7_to_3"


def _training_augmentation(config):
    """历史 E0/E1 默认基础增强，避免改写原有配置和 checkpoint。"""
    value = config.get("train_augmentation", True)
    if type(value) is not bool:
        raise ValueError("train_augmentation 必须是 true 或 false")
    required = {"E0": True, "E1": True, "E2": False, "E3": True, "E4": True,
                "E5": True, "E6": True, "E7": True, "E8": True,
                "E9": True, "E10": True, "E11": True, "E11_stem": True,
                "E11_stem_cbam": True}.get(config["experiment"])
    if required is not None and value is not required:
        raise ValueError(f"{config['experiment']} 的 train_augmentation 必须为 {required}")
    return value


def _mixup_protocol(config):
    """旧运行未使用 MixUp；E3 必须每个 batch 应用，避免混入其它实验变量。"""
    result = {}
    for field, default in (("mixup_alpha", 0.0), ("mixup_probability", 0.0)):
        value = config.get(field, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} 必须是有限非负数")
        result[field] = float(value)
    alpha, probability = result["mixup_alpha"], result["mixup_probability"]
    if probability > 1 or (alpha == 0) != (probability == 0):
        raise ValueError("mixup_alpha / mixup_probability 必须同时关闭，启用概率须位于 (0, 1]")
    experiment = config["experiment"]
    if (experiment in {"E0", "E1", "E2", "E5", "E6", "E7", "E8", "E9", "E10"} | CIFAR_TRANSFER_EXPERIMENTS
            and (alpha != 0 or probability != 0)):
        raise ValueError(f"{experiment} 基线禁止启用 MixUp")
    if experiment == "E3" and (alpha <= 0 or probability != 1):
        raise ValueError("E3 必须设置 mixup_alpha>0、mixup_probability=1")
    return result


def _cutmix_protocol(config):
    """旧运行默认关闭 CutMix；E4 按方案每个 batch 应用。"""
    result = {}
    for field in ("cutmix_alpha", "cutmix_probability"):
        value = config.get(field, 0.0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} 必须是有限非负数")
        result[field] = float(value)
    alpha, probability = result["cutmix_alpha"], result["cutmix_probability"]
    if probability > 1 or (alpha == 0) != (probability == 0):
        raise ValueError("cutmix_alpha / cutmix_probability 必须同时关闭，启用概率须位于 (0, 1]")
    experiment = config["experiment"]
    if experiment in {"E0", "E1", "E2", "E3", "E5", "E6", "E7", "E8", "E9", "E10"} and (alpha != 0 or probability != 0):
        raise ValueError(f"{experiment} 基线禁止启用 CutMix")
    if experiment == "E4" and (alpha <= 0 or probability != 1):
        raise ValueError("E4 必须设置 cutmix_alpha>0、cutmix_probability=1")
    if experiment in CIFAR_TRANSFER_EXPERIMENTS - {"E4"} and (alpha != 1 or probability != 1):
        raise ValueError(f"{experiment} 必须沿用 E4 的 cutmix_alpha=1、cutmix_probability=1")
    if config["mixup_alpha"] > 0 and alpha > 0:
        raise ValueError("MixUp 与 CutMix 必须互斥，不能同时启用")
    return result


def _cbam_protocol(config):
    values = {field: config.get(field) for field in CBAM_CONFIG}
    experiment = config["experiment"]
    if experiment in CBAM_EXPERIMENTS:
        if config.get("model") != CBAM_MODEL:
            raise ValueError(f"{experiment} 必须使用 {CBAM_MODEL} 模型")
        for field, expected in CBAM_CONFIG.items():
            if type(values[field]) is not int or values[field] != expected:
                raise ValueError(f"{experiment} 必须设置 {field}={expected}")
    elif config.get("model") == CBAM_MODEL or any(value is not None for value in values.values()):
        raise ValueError("只允许 E5 和 E11_stem_cbam 配置启用 CBAM")
    return values


def _scheduler_protocol(config):
    experiment = config["experiment"]
    scheduler = config.get("scheduler", "cosine")
    required = "fixed" if experiment == "E6" else "cosine"
    if scheduler != required:
        raise ValueError(f"{experiment} 必须设置 scheduler={required}")
    return scheduler


def _mixing_method(config):
    return "mixup" if config["mixup_alpha"] > 0 else (
        "cutmix" if config["cutmix_alpha"] > 0 else "none")


def _training_objective(config):
    smoothing = config["label_smoothing"]
    if smoothing:
        return "label_smoothed_cross_entropy", f"smoothed-label CE (epsilon={smoothing:g})"
    return {
        "none": ("cross_entropy", "hard-label CE"),
        "mixup": ("mixup_cross_entropy", "MixUp CE"),
        "cutmix": ("cutmix_cross_entropy", "CutMix CE"),
    }[_mixing_method(config)]


def _mixed_augmentation(config):
    return {"method": _mixing_method(config), **{
        key: config[key] for key in ("mixup_alpha", "mixup_probability", "cutmix_alpha", "cutmix_probability")}}


def _require_common_model_and_initial_state(left, right, description):
    models = [run["config"].get("model", "simple_cnn") for run in (left, right)]
    if models != ["cifar_resnet18", "cifar_resnet18"]:
        raise ValueError(f"{description}必须使用相同的 cifar_resnet18 模型")
    if left["summary"]["parameter_count"] != right["summary"]["parameter_count"]:
        raise ValueError(f"{description}的参数量必须相同")
    initial_a, initial_b = (run["summary"].get("initial_state_sha256") for run in (left, right))
    if not isinstance(initial_a, str) or not initial_a.strip() or initial_a != initial_b:
        raise ValueError(f"{description}必须具有相同的非空 initial_state_sha256")


def _require_e1_initial_source(run, baseline):
    source = run["summary"].get("initial_checkpoint_source")
    expected = baseline["summary"]["initial_state_sha256"]
    if (not isinstance(source, dict) or source.get("experiment") != "E1"
            or type(source.get("epoch")) is not int or source["epoch"] != 0
            or source.get("seed") != baseline["config"]["seed"]
            or source.get("state_sha256") != expected):
        experiment = run["config"]["experiment"]
        raise ValueError(f"{experiment} initial_checkpoint_source 必须匹配 E1 第 0 轮初始状态和种子")


def _compare_cbam(left, right):
    runs = {run["config"]["experiment"]: run for run in (left, right)}
    if set(runs) != {"E1", "E5"}:
        raise ValueError("CBAM 单项对照只允许 E1/E5")
    baseline, attention = runs["E1"], runs["E5"]
    if baseline["config"].get("model") != "cifar_resnet18":
        raise ValueError("CBAM 对照的 E1 必须使用 cifar_resnet18 模型")
    baseline_summary, attention_summary = baseline["summary"], attention["summary"]
    if attention_summary["parameter_count"] <= baseline_summary["parameter_count"]:
        raise ValueError("E5 启用 CBAM 后的参数量必须大于 E1")

    expected = baseline_summary.get("initial_state_sha256")
    if not isinstance(expected, str) or not expected.strip():
        raise ValueError("E1 缺少 initial_state_sha256")
    if attention_summary.get("initial_shared_state_sha256") != expected:
        raise ValueError("E5 initial_shared_state_sha256 与 E1 初始状态不一致")
    _require_e1_initial_source(attention, baseline)
    full = attention_summary.get("initial_state_sha256")
    if not isinstance(full, str) or not full.strip() or full == expected:
        raise ValueError("E5 initial_state_sha256 必须包含新增的 CBAM 参数")

    return "cbam", {"field": "cbam", **{
        side: {"model": run["config"]["model"], **{field: run["config"][field] for field in CBAM_CONFIG}}
        for side, run in (("left", left), ("right", right))}}


def _compare_lr_schedule(left, right):
    runs = {run["config"]["experiment"]: run for run in (left, right)}
    if set(runs) != {"E1", "E6"}:
        raise ValueError("学习率调度单项对照只允许 E1/E6")
    _require_common_model_and_initial_state(left, right, "E1/E6 学习率调度对照")
    _require_e1_initial_source(runs["E6"], runs["E1"])
    return "lr_schedule", {"field": "scheduler", "left": left["config"]["scheduler"],
                           "right": right["config"]["scheduler"]}


def _compare_early_stopping(left, right):
    runs = {run["config"]["experiment"]: run for run in (left, right)}
    if set(runs) != {"E1", "E7"}:
        raise ValueError("验证集早停单项对照只允许 E1/E7")
    _require_common_model_and_initial_state(left, right, "E1/E7 早停对照")
    _require_e1_initial_source(runs["E7"], runs["E1"])
    return "early_stopping", {"field": "early_stopping", "left": left["config"]["early_stopping"],
                              "right": right["config"]["early_stopping"]}


def _compare_label_smoothing(left, right):
    runs = {run["config"]["experiment"]: run for run in (left, right)}
    if set(runs) != {"E1", "E8"}:
        raise ValueError("标签平滑单项对照只允许 E1/E8")
    _require_common_model_and_initial_state(left, right, "E1/E8 标签平滑对照")
    _require_e1_initial_source(runs["E8"], runs["E1"])
    baseline = runs["E1"]["summary"]
    baseline_smoothing = baseline.get("label_smoothing", 0.0)
    if (baseline.get("train_objective_kind", "cross_entropy") != "cross_entropy"
            or type(baseline_smoothing) not in (int, float) or baseline_smoothing != 0.0):
        raise ValueError("E1 summary 必须使用普通交叉熵")
    return "label_smoothing", {"field": "label_smoothing", "left": left["config"]["label_smoothing"],
                               "right": right["config"]["label_smoothing"]}


def _check_transfer_protocol(config, summary, metrics, manifest):
    experiment = config["experiment"]
    cifar = experiment in CIFAR_TRANSFER_EXPERIMENTS
    protocol = RESIDUAL_TRANSFER_PROTOCOL if cifar else TRANSFER_PROTOCOL
    if experiment == "E11_stem_cbam":
        protocol = {**protocol, "model": CBAM_MODEL}
    defaults = CIFAR_TRANSFER_DEFAULTS if experiment == "E4" else {}
    pretrained = experiment in {"E10", "E11"} | STEM_CBAM_EXPERIMENTS
    for key, value in {**protocol, "pretrained": pretrained}.items():
        configured = config.get(key, defaults.get(key))
        if type(configured) is not type(value) or configured != value:
            raise ValueError(f"{experiment} 必须设置 {key}={value}")
        summary_key = "model_name" if key == "model" else key
        recorded = summary.get(summary_key, defaults.get(key))
        if type(recorded) is not type(value) or recorded != value:
            raise ValueError(f"{experiment} summary 的 {summary_key} 与配置不一致")
    checkpoint = config.get("initial_checkpoint")
    expected_weights = TRANSFER_WEIGHTS if pretrained else None
    if (config.get("pretrained_weights") != expected_weights
            or (pretrained and (not isinstance(checkpoint, str) or not checkpoint.strip()))
            or (experiment == "E9" and checkpoint is not None)):
        raise ValueError(f"{experiment} 的 pretrained_weights 或 initial_checkpoint 与初始化方案不一致")
    if cifar:
        if (any(config[key] != 1 or summary.get(key) != 1 for key in ("cutmix_alpha", "cutmix_probability"))
                or summary.get("train_objective_kind") != "cutmix_cross_entropy"
                or summary.get("train_accuracy_semantics") != "cutmix_weighted"):
            raise ValueError(f"{experiment} 需要 E4 的 CutMix 设置及对应训练指标")
    elif (config["dropout"] != 0 or config["momentum"] != 0
          or config.get("loss", "cross_entropy") != "cross_entropy"
          or summary.get("train_objective_kind") != "cross_entropy"
          or summary.get("train_accuracy_semantics") != "hard_label"):
        raise ValueError(f"{experiment} 要求标准分类头、AdamW 和普通交叉熵")
    if (metrics.get("model_name") != protocol["model"]
            or metrics.get("preprocessing", defaults.get("preprocessing")) != protocol["preprocessing"]
            or metrics.get("input_size", defaults.get("input_size")) != protocol["input_size"]
            or metrics.get("manifest_hash") != digest(manifest)):
        raise ValueError(f"{experiment} 的验证配置、输入尺寸或数据清单与训练不一致")


def _load_transfer_initial(run, protocol_fields=TRANSFER_PROTOCOL, *, defaults=None):
    import numpy as np
    import torch
    from src.models import state_dict_sha256

    path = run["path"] / "initial.pt"
    if not path.is_file():
        raise FileNotFoundError(f"预训练对照需要第 0 轮权重：{path}")
    # 允许读取项目写入的 NumPy RNG 状态，其余内容仍由 weights_only 限制。
    numpy_rng_types = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]
    with torch.serialization.safe_globals(numpy_rng_types):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config, summary = run["config"], run["summary"]
    snapshot = checkpoint.get("config", {})
    defaults = defaults or {}
    fields = (*PROTOCOL_FIELDS, *protocol_fields, "experiment", "pretrained", "pretrained_weights",
              "initial_checkpoint", "scheduler", "train_augmentation", "smoke")
    state = checkpoint.get("model")
    if (type(checkpoint.get("epoch")) is not int or checkpoint["epoch"] != 0
            or checkpoint.get("model_name") != config["model"]
            or checkpoint.get("manifest_hash") != digest(run["manifest"])
            or not isinstance(snapshot, dict)
            or any(snapshot.get(key, defaults.get(key)) != config.get(key, defaults.get(key)) for key in fields)
            or not isinstance(state, dict) or not all(isinstance(tensor, torch.Tensor) for tensor in state.values())):
        raise ValueError(f"{path} 的模型、配置、数据或轮次与正式运行不一致")
    state_hash = state_dict_sha256(state)
    if checkpoint.get("state_sha256") != state_hash or summary.get("initial_state_sha256") != state_hash:
        raise ValueError(f"{path} 的 initial_state_sha256 与实际模型状态不一致")
    return path, checkpoint


def _check_transfer_rng(baseline, transfer, manifest, *, cutmix=False):
    import torch

    expected = {"torch", "loaders"}
    for state in (baseline, transfer):
        if (not isinstance(state, dict) or not expected <= state.keys()
                or not isinstance(state["loaders"], dict) or state["loaders"].keys() != manifest["indices"].keys()):
            raise ValueError("预训练对照缺少第 0 轮训练与 DataLoader 随机状态")
    pairs = [(baseline["torch"], transfer["torch"])]
    pairs.extend((baseline["loaders"][key], transfer["loaders"][key]) for key in baseline["loaders"])
    if any(not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or not torch.equal(a, b)
           for a, b in pairs):
        raise ValueError("两组的第 0 轮预处理或 DataLoader 随机状态不同")
    if cutmix:
        import numpy as np

        rng_a, rng_b = baseline.get("numpy"), transfer.get("numpy")
        if (any(not isinstance(rng, tuple) or len(rng) != 5 or not isinstance(rng[1], np.ndarray)
                for rng in (rng_a, rng_b))
                or rng_a[0] != rng_b[0] or not np.array_equal(rng_a[1], rng_b[1]) or rng_a[2:] != rng_b[2:]):
            raise ValueError("两组的第 0 轮 CutMix 随机状态不同")


def _matched_transfer_initials(baseline, transfer, protocol_fields=TRANSFER_PROTOCOL, *, defaults=None):
    if baseline["summary"]["parameter_count"] != transfer["summary"]["parameter_count"]:
        raise ValueError("预训练对照的参数量必须相同")
    path, initial_base = _load_transfer_initial(baseline, protocol_fields, defaults=defaults)
    _, initial_transfer = _load_transfer_initial(transfer, protocol_fields, defaults=defaults)
    model_a, model_b = initial_base["model"], initial_transfer["model"]
    if model_a.keys() != model_b.keys() or any(
            model_a[key].shape != model_b[key].shape or model_a[key].dtype != model_b[key].dtype for key in model_a):
        raise ValueError("两组初始模型的张量名称、形状或类型不同")
    return path, initial_base, initial_transfer


def _check_recorded_transfer_source(transfer, initial_transfer, experiment, **reused_hashes):
    source = transfer["summary"].get("initial_checkpoint_source")
    if (not isinstance(source, dict) or source != initial_transfer.get("initial_checkpoint_source")
            or source.get("experiment") != experiment or type(source.get("epoch")) is not int or source["epoch"] != 0
            or source.get("seed") != transfer["config"]["seed"]
            or any(source.get(key) != value for key, value in reused_hashes.items())):
        raise ValueError(f"initial_checkpoint_source 必须匹配所配对的 {experiment} 第 0 轮权重文件")
    for key in ("state_sha256", "file_sha256"):
        checksum = source.get(key)
        if (not isinstance(checksum, str) or len(checksum) != 64
                or not all(char in "0123456789abcdef" for char in checksum)):
            raise ValueError(f"initial_checkpoint_source 的 {key} 必须是有效的 SHA-256 值")
    return source


def _check_initial_transfer_source(path, baseline, transfer, initial_transfer, **reused_hashes):
    experiment = baseline["config"]["experiment"]
    source = _check_recorded_transfer_source(transfer, initial_transfer, experiment, **reused_hashes)
    with path.open("rb") as file:
        baseline_file_hash = hashlib.file_digest(file, "sha256").hexdigest()
    if (source["state_sha256"] != baseline["summary"].get("initial_state_sha256")
            or source["file_sha256"] != baseline_file_hash):
        raise ValueError(f"initial_checkpoint_source 必须匹配所配对的 {experiment} 第 0 轮权重文件")


def _check_pretrained_source(transfer, initial, backbone_hash, **initialization):
    experiment = transfer["config"]["experiment"]
    weights = transfer["summary"].get("pretrained_weight_source")
    if (not isinstance(weights, dict) or weights != initial.get("pretrained_weight_source")
            or weights.get("enum") != f"ResNet18_Weights.{TRANSFER_WEIGHTS}" or weights.get("url") != TRANSFER_WEIGHTS_URL
            or any(weights.get(key) != value for key, value in initialization.items())
            or weights.get("backbone_state_sha256") != backbone_hash):
        raise ValueError(f"{experiment} pretrained_weight_source 与 ImageNet 权重或实际初始化骨干不一致")
    if weights.get("sha256") != RESNET18_IMAGENET1K_V1_SHA256:
        raise ValueError(f"{experiment} pretrained_weight_source 的官方权重 SHA256 不匹配")


def _compare_pretrained_backbone(left, right):
    from src.models import state_dict_sha256

    runs = {run["config"]["experiment"]: run for run in (left, right)}
    if runs.keys() != TRANSFER_EXPERIMENTS:
        raise ValueError("预训练骨干对照只允许 E9/E10")
    baseline, transfer = runs["E9"], runs["E10"]
    path9, initial9, initial10 = _matched_transfer_initials(baseline, transfer)
    model9, model10 = initial9["model"], initial10["model"]
    classes = len(baseline["manifest"]["classes"])
    head_shapes = {"fc.weight": (classes, 512), "fc.bias": (classes,)}
    if any(key not in model9 or tuple(model9[key].shape) != shape for key, shape in head_shapes.items()):
        raise ValueError("预训练对照缺少有效的标准 ResNet-18 分类头")
    head9 = state_dict_sha256({key: model9[key] for key in head_shapes})
    head10 = state_dict_sha256({key: model10[key] for key in head_shapes})
    if (head9 != head10 or transfer["summary"].get("initial_classifier_state_sha256") != head9
            or initial10.get("initial_classifier_state_sha256") != head9):
        raise ValueError("E10 必须复用 E9 第 0 轮分类头的全部参数")
    if initial9["state_sha256"] == initial10["state_sha256"]:
        raise ValueError("E9/E10 的骨干初始状态必须不同")
    _check_transfer_rng(initial9.get("rng"), initial10.get("rng"), baseline["manifest"])
    _check_initial_transfer_source(path9, baseline, transfer, initial10, classifier_state_sha256=head9)
    backbone_hash = state_dict_sha256({key: tensor for key, tensor in model10.items() if not key.startswith("fc.")})
    _check_pretrained_source(transfer, initial10, backbone_hash)
    for run in runs.values():
        run["initial_classifier_state_sha256"] = head9
    return "pretrained_backbone", {"field": "pretrained", "left": left["config"]["pretrained"],
                                   "right": right["config"]["pretrained"]}


def _check_cutmix_sequence(baseline, transfer):
    pair = "/".join(run["config"]["experiment"] for run in (baseline, transfer))
    paths = [run["path"] / "cutmix_batches.csv" for run in (baseline, transfer)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{pair} 对照需要完整的 CutMix 逐批日志：{path}")
    config, manifest = baseline["config"], baseline["manifest"]
    batches_per_epoch = math.ceil(len(manifest["indices"]["train"]) / config["batch_size"])
    if batches_per_epoch < 1:
        raise ValueError("CutMix 对照需要非空的训练划分")
    expected_rows = config["epochs"] * batches_per_epoch
    required = {"epoch", "batch", "lambda_raw", "lambda_eff", "x1", "y1", "x2", "y2", "actual_area", "applied"}
    with paths[0].open(encoding="utf-8-sig", newline="") as fa, paths[1].open(encoding="utf-8-sig", newline="") as fb:
        a, b = csv.DictReader(fa), csv.DictReader(fb)
        if any(not required <= set(reader.fieldnames or ()) for reader in (a, b)):
            raise ValueError("CutMix 逐批日志缺少必要的随机参数字段")
        row_count = 0
        for row_count, (row_a, row_b) in enumerate(zip_longest(a, b), start=1):
            epoch, batch = divmod(row_count - 1, batches_per_epoch)
            if (row_a is None or row_b is None or row_a != row_b
                    or row_a["epoch"] != str(epoch + 1) or row_a["batch"] != str(batch + 1)
                    or row_a["applied"] != "True"):
                raise ValueError(f"{pair} 第 {epoch + 1} 轮第 {batch + 1} 批的 CutMix 日志不一致")
    if row_count != expected_rows:
        raise ValueError(f"{pair} 的 CutMix 逐批日志应含 {expected_rows} 条，实际 {row_count} 条")
    return row_count


def _compare_pretrained_residual_layers(left, right):
    from src.models import state_dict_sha256
    from src.models.factory import RESNET_RESIDUAL_PREFIXES

    runs = {run["config"]["experiment"]: run for run in (left, right)}
    if runs.keys() != RESIDUAL_TRANSFER_EXPERIMENTS:
        raise ValueError("32×32 残差阶段预训练对照只允许 E4/E11")
    baseline, transfer = runs["E4"], runs["E11"]
    for run in (baseline, transfer):
        _check_transfer_protocol(run["config"], run["summary"], run["metrics"], run["manifest"])
    fields = (*RESIDUAL_TRANSFER_PROTOCOL, "cutmix_alpha", "cutmix_probability", "label_smoothing")
    path4, initial4, initial11 = _matched_transfer_initials(
        baseline, transfer, fields, defaults=CIFAR_TRANSFER_DEFAULTS)
    model4, model11 = initial4["model"], initial11["model"]
    retained = [key for key in model4 if not key.startswith(RESNET_RESIDUAL_PREFIXES)]
    for prefix in RESNET_RESIDUAL_PREFIXES:
        if not any(key.startswith(prefix) for key in model4):
            raise ValueError(f"预训练对照缺少残差阶段 {prefix[:-1]} 的权重")
    if not {"conv1.weight", "bn1.weight", "fc.1.weight", "fc.1.bias"} <= set(retained):
        raise ValueError("预训练对照缺少 CIFAR ResNet-18 入口或分类头权重")
    hash4 = state_dict_sha256({key: model4[key] for key in retained})
    hash11 = state_dict_sha256({key: model11[key] for key in retained})
    if (hash4 != hash11 or transfer["summary"].get("initial_retained_state_sha256") != hash4
            or initial11.get("initial_retained_state_sha256") != hash4):
        raise ValueError("E11 必须逐张量复用 E4 第 0 轮的入口和分类头")
    if initial4["state_sha256"] == initial11["state_sha256"]:
        raise ValueError("E4/E11 的四个残差阶段初始状态必须不同")
    _check_transfer_rng(initial4.get("rng"), initial11.get("rng"), baseline["manifest"], cutmix=True)
    _check_initial_transfer_source(path4, baseline, transfer, initial11, retained_state_sha256=hash4)
    backbone_hash = state_dict_sha256({key: value for key, value in model11.items()
                                       if key.startswith(RESNET_RESIDUAL_PREFIXES)})
    _check_pretrained_source(transfer, initial11, backbone_hash, scope="layer1-layer4")
    batch_count = _check_cutmix_sequence(baseline, transfer)
    for run in runs.values():
        run.update(initial_retained_state_sha256=hash4, matched_cutmix_batches=batch_count)
    return "pretrained_residual_layers", {"field": "pretrained", "left": left["config"].get("pretrained", False),
                                          "right": right["config"].get("pretrained", False)}


def _compare_cifar_pretrained_stem(left, right):
    from src.models import state_dict_sha256
    from src.models.factory import RESNET_RESIDUAL_PREFIXES

    runs = {run["config"]["experiment"]: run for run in (left, right)}
    baseline_ids = runs.keys() & STEM_TRANSFER_BASELINES
    if len(runs) != 2 or len(baseline_ids) != 1:
        raise ValueError("CIFAR 首层适配对照需要 E4 或 E11 作为基线")
    baseline_id = next(iter(baseline_ids))
    baseline, transfer = runs[baseline_id], runs["E11_stem"]
    for run in (baseline, transfer):
        _check_transfer_protocol(run["config"], run["summary"], run["metrics"], run["manifest"])
    fields = (*RESIDUAL_TRANSFER_PROTOCOL, "cutmix_alpha", "cutmix_probability", "label_smoothing")
    base_path, base_init, stem_init = _matched_transfer_initials(
        baseline, transfer, fields, defaults=CIFAR_TRANSFER_DEFAULTS)
    base_model, stem_model = base_init["model"], stem_init["model"]
    classifier = [key for key in base_model if key.startswith("fc.")]
    residual = [key for key in base_model if key.startswith(RESNET_RESIDUAL_PREFIXES)]
    stem = [key for key in base_model if key.startswith(("conv1.", "bn1."))]
    if (not {"fc.1.weight", "fc.1.bias"} <= set(classifier)
            or not {"conv1.weight", "bn1.weight"} <= set(stem)
            or tuple(base_model["conv1.weight"].shape) != (64, 3, 3, 3)
            or not all(any(key.startswith(prefix) for key in residual) for prefix in RESNET_RESIDUAL_PREFIXES)
            or set(base_model) != set(classifier + residual + stem)):
        raise ValueError("首层适配对照缺少有效的 CIFAR ResNet-18 入口、残差阶段或分类头权重")

    head_hash = state_dict_sha256({key: base_model[key] for key in classifier})
    if (state_dict_sha256({key: stem_model[key] for key in classifier}) != head_hash
            or transfer["summary"].get("initial_classifier_state_sha256") != head_hash
            or stem_init.get("initial_classifier_state_sha256") != head_hash):
        raise ValueError("E11_stem 必须逐张量复用 E4 第 0 轮分类头")
    _check_transfer_rng(base_init.get("rng"), stem_init.get("rng"), baseline["manifest"], cutmix=True)
    if (state_dict_sha256({"conv1.weight": base_model["conv1.weight"]})
            == state_dict_sha256({"conv1.weight": stem_model["conv1.weight"]})
            or state_dict_sha256({key: base_model[key] for key in stem if key.startswith("bn1.")})
            == state_dict_sha256({key: stem_model[key] for key in stem if key.startswith("bn1.")})):
        raise ValueError("E11_stem 的首层卷积与入口 BatchNorm 都必须有不同的初始状态")

    base_residual_hash = state_dict_sha256({key: base_model[key] for key in residual})
    stem_residual_hash = state_dict_sha256({key: stem_model[key] for key in residual})
    if baseline_id == "E4":
        if base_residual_hash == stem_residual_hash:
            raise ValueError("E4/E11_stem 的四个残差阶段初始状态必须不同")
        _check_initial_transfer_source(base_path, baseline, transfer, stem_init, classifier_state_sha256=head_hash)
        base_scope = "none"
    else:
        if base_residual_hash != stem_residual_hash:
            raise ValueError("E11/E11_stem 必须使用相同的四个残差阶段初始张量")
        retained_hash = state_dict_sha256({key: base_model[key] for key in stem + classifier})
        if (baseline["summary"].get("initial_retained_state_sha256") != retained_hash
                or base_init.get("initial_retained_state_sha256") != retained_hash):
            raise ValueError("E11 的初始入口和分类头哈希与实际张量不一致")
        source_e11 = _check_recorded_transfer_source(baseline, base_init, "E4", retained_state_sha256=retained_hash)
        source_stem = _check_recorded_transfer_source(transfer, stem_init, "E4", classifier_state_sha256=head_hash)
        if any(source_e11[key] != source_stem[key] for key in ("state_sha256", "file_sha256")):
            raise ValueError("E11/E11_stem 必须复用同一个 E4 第 0 轮权重文件")
        base_scope = "layer1-layer4"
        _check_pretrained_source(baseline, base_init, base_residual_hash, scope=base_scope)

    backbone_hash = state_dict_sha256({key: stem_model[key] for key in stem + residual})
    _check_pretrained_source(transfer, stem_init, backbone_hash,
                             scope=STEM_TRANSFER_SCOPE, stem_kernel_adaptation=STEM_KERNEL_ADAPTATION)
    batch_count = _check_cutmix_sequence(baseline, transfer)
    scopes = {baseline_id: base_scope, "E11_stem": STEM_TRANSFER_SCOPE}
    for run in runs.values():
        run.update(initial_classifier_state_sha256=head_hash, matched_cutmix_batches=batch_count)
    return "pretrained_cifar_stem", {"field": "pretrained_scope", "left": scopes[left["config"]["experiment"]],
                                    "right": scopes[right["config"]["experiment"]]}


def _compare_pretrained_cbam(left, right):
    import torch
    from src.models import state_dict_sha256

    runs = {run["config"]["experiment"]: run for run in (left, right)}
    if runs.keys() != STEM_CBAM_EXPERIMENTS:
        raise ValueError("32×32 预训练注意力对照需要 E11_stem 和 E11_stem_cbam")
    baseline, attention = runs["E11_stem"], runs["E11_stem_cbam"]
    for run in runs.values():
        _check_transfer_protocol(run["config"], run["summary"], run["metrics"], run["manifest"])
    fields = (*RESIDUAL_TRANSFER_PROTOCOL, *CBAM_CONFIG, "cutmix_alpha", "cutmix_probability",
              "mixup_alpha", "mixup_probability", "label_smoothing")
    path, initial_base = _load_transfer_initial(baseline, fields)
    _, initial_attention = _load_transfer_initial(attention, fields)
    base_model = initial_base["model"]
    shared = {key: value for key, value in initial_attention["model"].items() if ".cbam." not in key}
    added = initial_attention["model"].keys() - shared.keys()
    shared_hash = initial_base["state_sha256"]
    if (state_dict_sha256(shared) != shared_hash
            or initial_attention.get("shared_state_sha256") != shared_hash
            or attention["summary"].get("initial_shared_state_sha256") != shared_hash):
        raise ValueError("E11_stem_cbam 的共有层必须匹配 E11_stem 第 0 轮初始张量")
    added_parameters = sum(initial_attention["model"][key].numel() for key in added)
    if (added_parameters <= 0 or
            attention["summary"]["parameter_count"] != baseline["summary"]["parameter_count"] + added_parameters):
        raise ValueError("E11_stem_cbam 的参数增量必须与新增的 CBAM 张量一致")
    _check_initial_transfer_source(path, baseline, attention, initial_attention)

    rng_base, rng_attention = initial_base.get("rng"), initial_attention.get("rng")
    _check_transfer_rng(rng_base, rng_attention, baseline["manifest"], cutmix=True)
    cuda_base, cuda_attention = rng_base.get("cuda"), rng_attention.get("cuda")
    if (not isinstance(rng_base.get("python"), tuple) or rng_base["python"] != rng_attention.get("python")
            or not isinstance(cuda_base, list) or not isinstance(cuda_attention, list)
            or (baseline["config"]["device"] == "cuda" and not cuda_base)
            or len(cuda_base) != len(cuda_attention)
            or any(not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or not torch.equal(a, b)
                   for a, b in zip(cuda_base, cuda_attention))):
        raise ValueError("两组的第 0 轮 Python 或 CUDA 随机状态不同")

    head_keys = {"fc.1.weight", "fc.1.bias"}
    if not head_keys <= base_model.keys():
        raise ValueError("E11_stem 缺少 CIFAR ResNet-18 分类头权重")
    head_hash = state_dict_sha256({key: base_model[key] for key in head_keys})
    backbone_hash = state_dict_sha256({key: value for key, value in base_model.items() if key not in head_keys})
    if initial_attention.get("pretrained_weight_source") != initial_base.get("pretrained_weight_source"):
        raise ValueError("E11_stem_cbam 必须继承 E11_stem 的预训练权重来源")
    batch_count = _check_cutmix_sequence(baseline, attention)
    for run, initial in ((baseline, initial_base), (attention, initial_attention)):
        _check_pretrained_source(run, initial, backbone_hash,
                                 scope=STEM_TRANSFER_SCOPE, stem_kernel_adaptation=STEM_KERNEL_ADAPTATION)
        if (initial.get("initial_classifier_state_sha256") != head_hash
                or run["summary"].get("initial_classifier_state_sha256") != head_hash):
            raise ValueError("32×32 预训练注意力对照的分类头哈希必须匹配实际初始张量")
        run.update(initial_classifier_state_sha256=head_hash, matched_cutmix_batches=batch_count)
    return "pretrained_cbam", {"field": "cbam", **{
        side: {"model": run["config"]["model"], **{field: run["config"][field] for field in CBAM_CONFIG}}
        for side, run in (("left", left), ("right", right))}}


def _comparison_design(left, right):
    """接受单项消融，以及预先约定的 E3/E4 混合方法方案比较。"""
    a, b = left["config"], right["config"]
    if "E11_stem_cbam" in (a["experiment"], b["experiment"]):
        return _compare_pretrained_cbam(left, right)
    if "E11_stem" in (a["experiment"], b["experiment"]):
        return _compare_cifar_pretrained_stem(left, right)
    if "E11" in (a["experiment"], b["experiment"]):
        return _compare_pretrained_residual_layers(left, right)
    if TRANSFER_EXPERIMENTS.intersection((a["experiment"], b["experiment"])):
        return _compare_pretrained_backbone(left, right)
    if "E8" in (a["experiment"], b["experiment"]):
        return _compare_label_smoothing(left, right)
    if "E7" in (a["experiment"], b["experiment"]):
        return _compare_early_stopping(left, right)
    if "E6" in (a["experiment"], b["experiment"]):
        return _compare_lr_schedule(left, right)
    if "E5" in (a["experiment"], b["experiment"]):
        return _compare_cbam(left, right)
    models = [config.get("model", "simple_cnn") for config in (a, b)]
    mixup_fields = ("mixup_alpha", "mixup_probability")
    cutmix_fields = ("cutmix_alpha", "cutmix_probability")
    augmentation_differs = a["train_augmentation"] != b["train_augmentation"]
    mixup_differs = any(a[field] != b[field] for field in mixup_fields)
    cutmix_differs = any(a[field] != b[field] for field in cutmix_fields)
    if augmentation_differs and (mixup_differs or cutmix_differs):
        methods = "MixUp" if mixup_differs else "CutMix"
        raise ValueError(f"不能作为单因素对照：两组同时改变基础增强与 {methods}")
    if augmentation_differs:
        if {a["experiment"], b["experiment"]} != {"E1", "E2"}:
            raise ValueError("train_augmentation 不同只允许 E1/E2 增强对照")
        _require_common_model_and_initial_state(left, right, "E1/E2 增强对照")
        return "augmentation", {"field": "train_augmentation", "left": a["train_augmentation"],
                                "right": b["train_augmentation"]}
    if mixup_differs and cutmix_differs:
        if {a["experiment"], b["experiment"]} != {"E3", "E4"}:
            raise ValueError("同时切换 MixUp/CutMix 只允许预设 E3/E4 混合方法方案比较")
        _require_common_model_and_initial_state(left, right, "E3/E4 混合方法比较")
        return "mixup_vs_cutmix", {"field": "mixed_augmentation", "left": _mixed_augmentation(a),
                                    "right": _mixed_augmentation(b)}
    if cutmix_differs:
        if {a["experiment"], b["experiment"]} != {"E1", "E4"}:
            raise ValueError("CutMix 不同只允许 E1/E4 单因素对照")
        _require_common_model_and_initial_state(left, right, "E1/E4 CutMix 对照")
        return "cutmix", {"field": "cutmix", "left": {key: a[key] for key in cutmix_fields},
                          "right": {key: b[key] for key in cutmix_fields}}
    if mixup_differs:
        if {a["experiment"], b["experiment"]} != {"E1", "E3"}:
            raise ValueError("MixUp 不同只允许 E1/E3 单因素对照")
        if not a["train_augmentation"] or not b["train_augmentation"]:
            raise ValueError("E1/E3 必须保留相同基础增强")
        _require_common_model_and_initial_state(left, right, "E1/E3 MixUp 对照")
        return "mixup", {"field": "mixup", "left": {key: a[key] for key in mixup_fields},
                         "right": {key: b[key] for key in mixup_fields}}
    if models[0] != models[1]:
        return "architecture", {"field": "model", "left": models[0], "right": models[1]}
    return "same_protocol", None


def _close(actual, expected, description, tolerance=1e-10):
    if not math.isfinite(float(actual)) or not math.isclose(
        float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance
    ):
        raise ValueError(f"{description} 不一致：{actual} / {expected}")


def _check_declared_lr_curve(run):
    config = run["config"]
    initial, minimum = config["learning_rate"], config["min_learning_rate"]
    if any(type(lr) not in (int, float) or not math.isfinite(lr) for lr in (initial, minimum)):
        raise ValueError("learning_rate 和 min_learning_rate 必须是有限数值")
    if initial <= 0 or not 0 <= minimum <= initial:
        raise ValueError("min_learning_rate 必须位于 0 与初始学习率之间，初始学习率须大于 0")
    for row in run["history"]:
        epoch = row["epoch"]
        expected = initial
        if config["scheduler"] == "cosine":
            # 历史记录的是本轮训练使用的 LR，余弦更新在整轮结束后执行。
            progress = (epoch - 1) / config["epochs"]
            expected = minimum + (initial - minimum) * (1 + math.cos(math.pi * progress)) / 2
        _close(row["lr"], expected, f"{config['experiment']} 第 {epoch} 轮实际学习率轨迹", tolerance=1e-12)


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _check_early_stopping_history(history, settings, summary):
    state = replay_early_stopping(settings, history)
    reason = "early_stopping" if state["stopped_epoch"] is not None else "max_epochs"
    if (summary.get("stop_reason") != reason
            or (reason == "max_epochs" and len(history) != summary["planned_epochs"])):
        raise ValueError("E7 停止原因或结束轮次与验证准确率回放不符")
    recorded = summary.get("early_stopping_state")
    if (not isinstance(recorded, dict) or recorded != state
            or any(type(recorded[key]) is not type(value) for key, value in state.items())):
        raise ValueError("E7 summary 的早停状态与验证准确率回放不符")


def _load_history(run_dir, config, summary):
    rows = _read_csv(run_dir / "history.csv")
    epochs = config["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("总轮数必须为正整数")
    completed = summary["completed_epochs"]
    settings = config["early_stopping"]
    if (summary.get("complete") is not True or type(completed) is not int or not 1 <= completed <= epochs
            or type(summary["planned_epochs"]) is not int or summary["planned_epochs"] != epochs
            or (settings is None and completed != epochs)):
        raise ValueError(f"运行尚未完成预定轮数：{run_dir}")
    if len(rows) != completed:
        raise ValueError("history 行数与已完成轮数不同")
    result = []
    for epoch, row in enumerate(rows, 1):
        if not set(HISTORY_FIELDS) <= set(row):
            raise ValueError("history 缺少训练/验证或计时字段")
        parsed = {key: float(row[key]) for key in HISTORY_FIELDS}
        if not all(math.isfinite(value) for value in parsed.values()):
            raise ValueError("history 含有非有限值")
        if parsed["epoch"] != epoch:
            raise ValueError("history 轮次必须从 1 开始连续记录")
        if any(parsed[key] < 0 for key in ("lr", "train_objective", "train_clean_loss", "val_loss", "epoch_seconds")):
            raise ValueError("history 的损失、学习率和时间不能为负")
        if any(not 0 <= parsed[key] <= 1 for key in ("train_clean_accuracy", "val_accuracy")):
            raise ValueError("history 准确率必须位于 0–1")
        parsed["epoch"] = epoch
        result.append(parsed)
    if settings is not None:
        _check_early_stopping_history(result, settings, summary)
    # max 保留第一条平局，符合训练器的准确率/损失/较早轮次选择规则。
    best = max(result, key=lambda row: (row["val_accuracy"], -row["val_loss"]))
    if best["epoch"] != summary["best_epoch"]:
        raise ValueError("summary 最佳轮次与 history 的选择规则不符")
    _close(best["val_accuracy"], summary["best_val_accuracy"], "summary 最佳准确率")
    _close(best["val_loss"], summary["best_val_loss"], "summary 配套验证 loss")
    _close(sum(row["epoch_seconds"] for row in result), summary["epoch_seconds_sum_excluding_checkpoint_io"],
           "summary 逐轮计时总和", tolerance=1e-7)
    return result


def _load_predictions(run_dir, manifest, summary):
    evaluation = run_dir / f"evaluation_validation_epoch{summary['best_epoch']}"
    if not (evaluation / "metrics.json").is_file() or not (evaluation / "predictions.csv").is_file():
        raise FileNotFoundError(f"缺少最佳权重的验证集结果，请先运行 evaluate.py：{evaluation}")
    metrics = read_json(evaluation / "metrics.json")
    if metrics.get("split") != "validation" or metrics.get("smoke") is not False:
        raise ValueError("比较工具只接受正式运行的 validation 结果")
    if metrics["epoch"] != summary["best_epoch"]:
        raise ValueError("评估轮次与最佳权重轮次不同")
    _close(metrics["overall_accuracy"], summary["best_val_accuracy"], "评估与 summary 准确率")
    _close(metrics["loss"], summary["best_val_loss"], "评估与 summary loss", tolerance=1e-5)
    names = manifest["classes"]
    expected = manifest["indices"]["validation"]
    if not expected or len(set(expected)) != len(expected) or any(type(i) is not int or i < 0 for i in expected):
        raise ValueError("manifest 的验证索引必须非空、非负、唯一且为整数")
    predictions = {}
    for row in _read_csv(evaluation / "predictions.csv"):
        try:
            index, true, pred = (int(row[key]) for key in ("index", "true_label", "predicted_label"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("预测 CSV 必须包含整数 index/true_label/predicted_label") from exc
        if index in predictions:
            raise ValueError("预测 CSV 存在重复索引")
        if not 0 <= true < len(names) or not 0 <= pred < len(names):
            raise ValueError("预测 CSV 的类别编号越界")
        if ("true_class" in row and row["true_class"] != names[true]) or (
            "predicted_class" in row and row["predicted_class"] != names[pred]
        ):
            raise ValueError("预测 CSV 的类别名称与编号不符")
        confidence = float(row["confidence"]) if "confidence" in row else None
        if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("预测置信度必须位于 0–1")
        predictions[index] = {"true_label": true, "predicted_label": pred, "confidence": confidence}
    if set(predictions) != set(expected):
        raise ValueError("预测索引与 manifest 的完整验证集不一致")
    ordered = [predictions[index] for index in sorted(expected)]
    computed = classification_metrics([row["true_label"] for row in ordered],
                                      [row["predicted_label"] for row in ordered], names)
    if metrics["sample_count"] != len(expected) or metrics["confusion_matrix"] != computed["confusion_matrix"]:
        raise ValueError("评估样本数或混淆矩阵与预测文件不一致")
    _close(metrics["overall_accuracy"], computed["overall_accuracy"], "预测复算准确率")
    _close(metrics["macro_f1"], computed["macro_f1"], "预测复算 Macro-F1")
    if len(metrics["per_class"]) != len(names):
        raise ValueError("评估每类结果数量错误")
    for recorded, actual in zip(metrics["per_class"], computed["per_class"]):
        if any(recorded[key] != actual[key] for key in ("name", "support", "correct")):
            raise ValueError("评估每类结果与预测文件不一致")
        for key in ("accuracy", "precision", "recall", "f1"):
            _close(recorded[key], actual[key], f"每类 {key}")
    return metrics, predictions


def load_run(run_dir):
    run_dir = Path(run_dir).resolve()
    config, summary = read_json(run_dir / "config.json"), read_json(run_dir / "summary.json")
    manifest = read_json(run_dir / "data_manifest.json")
    if config.get("smoke") is not False or summary.get("smoke") is not False:
        raise ValueError("smoke 结果不能用于正式架构比较")
    config["train_augmentation"] = _training_augmentation(config)
    config.update(_mixup_protocol(config))
    config.update(_cutmix_protocol(config))
    config.update(_cbam_protocol(config))
    config["scheduler"] = _scheduler_protocol(config)
    config["label_smoothing"] = normalize_label_smoothing(config.get("label_smoothing", 0.0), config["experiment"])
    if config["experiment"] == "E8":
        recorded = summary.get("label_smoothing")
        if type(recorded) not in (int, float) or recorded != config["label_smoothing"]:
            raise ValueError("E8 summary 的 label_smoothing 与运行配置不一致")
        if summary.get("train_objective_kind") != _training_objective(config)[0]:
            raise ValueError("E8 summary 的 train_objective_kind 必须是 label_smoothed_cross_entropy")
    settings = validate_early_stopping(config.get("early_stopping"))
    if (settings is not None) != (config["experiment"] == "E7"):
        raise ValueError("只有 E7 必须启用 early_stopping")
    config["early_stopping"] = settings
    for field in PROTOCOL_FIELDS:
        if field not in config:
            raise ValueError(f"运行配置缺少训练协议字段：{field}")
    if type(summary["parameter_count"]) is not int or summary["parameter_count"] < 1:
        raise ValueError("参数量必须为正整数")
    history = _load_history(run_dir, config, summary)
    metrics, predictions = _load_predictions(run_dir, manifest, summary)
    if config["experiment"] in TRANSFER_EXPERIMENTS:
        _check_transfer_protocol(config, summary, metrics, manifest)
    return {"path": run_dir, "config": config, "summary": summary, "manifest": manifest,
            "history": history, "metrics": metrics, "predictions": predictions}


def _overview(run):
    summary, metrics, config = run["summary"], run["metrics"], run["config"]
    best_row = run["history"][summary["best_epoch"] - 1]
    final_rows = run["history"][-10:]
    checkpoint = run["path"] / "best.pt"
    result = {
        "experiment": config["experiment"],
        "model": config.get("model", "simple_cnn"), "run_dir": str(run["path"]),
        "train_augmentation": config["train_augmentation"],
        "mixup_alpha": config["mixup_alpha"], "mixup_probability": config["mixup_probability"],
        "cutmix_alpha": config["cutmix_alpha"], "cutmix_probability": config["cutmix_probability"],
        **{field: config[field] for field in CBAM_CONFIG},
        "label_smoothing": config["label_smoothing"],
        "train_objective_kind": _training_objective(config)[0],
        "train_accuracy_semantics": (_mixing_method(config) + "_weighted"
                                     if _mixing_method(config) != "none" else "hard_label"),
        "initial_state_sha256": summary.get("initial_state_sha256"),
        "initial_shared_state_sha256": summary.get("initial_shared_state_sha256"),
        "scheduler": config["scheduler"],
        "first_epoch_lr": run["history"][0]["lr"], "last_epoch_lr": run["history"][-1]["lr"],
        "seed": config["seed"], "epochs": config["epochs"], "best_epoch": summary["best_epoch"],
        "completed_epochs": summary["completed_epochs"], "stop_reason": summary.get("stop_reason", "max_epochs"),
        "early_stopping_state": summary.get("early_stopping_state"),
        "best_val_accuracy": metrics["overall_accuracy"], "best_val_loss": metrics["loss"],
        "last10_mean_val_accuracy": math.fsum(row["val_accuracy"] for row in final_rows) / len(final_rows),
        "macro_f1": metrics["macro_f1"], "parameter_count": summary["parameter_count"],
        "clean_train_val_gap_pp_at_best": 100 * (best_row["train_clean_accuracy"] - metrics["overall_accuracy"]),
        "logged_epoch_seconds_excluding_checkpoint_io": summary["epoch_seconds_sum_excluding_checkpoint_io"],
        "peak_cuda_allocated_mib": summary.get("peak_cuda_allocated_mib"),
        "best_checkpoint_bytes_including_training_state": checkpoint.stat().st_size if checkpoint.is_file() else None,
    }
    for threshold in (90, 92, 93):
        result[f"first_epoch_val_ge_{threshold}pct"] = next(
            (row["epoch"] for row in run["history"] if row["val_accuracy"] >= threshold / 100), None)
    if "initial_classifier_state_sha256" in run:
        result.update(preprocessing=config.get("preprocessing", "cifar32"), input_size=config.get("input_size", 32),
                      optimizer=config.get("optimizer", "sgd"), pretrained=config.get("pretrained", False),
                      pretrained_weights=config.get("pretrained_weights"),
                      initial_classifier_state_sha256=run["initial_classifier_state_sha256"],
                      pretrained_weight_source=summary.get("pretrained_weight_source"))
    if "initial_retained_state_sha256" in run:
        result.update(preprocessing="cifar32", input_size=32, optimizer="sgd",
                      pretrained=config.get("pretrained", False), pretrained_weights=config.get("pretrained_weights"),
                      initial_retained_state_sha256=run["initial_retained_state_sha256"],
                      pretrained_weight_source=summary.get("pretrained_weight_source"))
    return result


def _plot_comparison(runs, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    objective_title = "Training objective"
    if any(run["config"]["label_smoothing"] for run in runs):
        objective_title = "Training objectives (different targets)"
    fields = [("train_objective", objective_title, False),
              ("train_clean_loss", "Clean training monitor CE", False),
              ("val_loss", "Validation CE", False),
              ("train_clean_accuracy", "Clean training monitor accuracy (%)", True),
              ("val_accuracy", "Validation accuracy (%)", True), ("lr", "Learning rate used", False)]
    for ax, (field, title, percent) in zip(axes.flat, fields):
        for position, run in enumerate(runs):
            label = f"{run['config']['experiment']} ({'left' if position == 0 else 'right'})"
            if field == "train_objective":
                label += " / " + _training_objective(run["config"])[1]
            elif field == "lr":
                label += f" / {run['config']['scheduler']}"
            ax.plot([row["epoch"] for row in run["history"]],
                    [row[field] * (100 if percent else 1) for row in run["history"]], label=label)
        ax.set(title=title, xlabel="Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def compare_runs(run_dirs, output_dir):
    """校验并比较两个 run；输入目录只读，返回已写入 comparison.json 的数据。"""
    if len(run_dirs) != 2:
        raise ValueError("每次必须提供两个运行目录")
    inputs = [Path(path).resolve() for path in run_dirs]
    output = Path(output_dir).resolve()
    if inputs[0] == inputs[1]:
        raise ValueError("两个输入运行目录必须不同")
    if any(output == path or output.is_relative_to(path) or path.is_relative_to(output) for path in inputs):
        raise ValueError("输出目录必须与输入运行目录独立，不能覆盖输入或使用其父目录")
    left, right = [load_run(path) for path in inputs]
    if left["manifest"] != right["manifest"]:
        raise ValueError("两组数据划分、类别或归一化清单不同")
    differences = [key for key in PROTOCOL_FIELDS if left["config"][key] != right["config"][key]]
    for key, default in (("optimizer", "sgd"), ("loss", "cross_entropy")):
        if left["config"].get(key, default) != right["config"].get(key, default):
            differences.append(key)
    if differences:
        raise ValueError("两组训练协议不同：" + ", ".join(differences))
    comparison_kind, controlled_difference = _comparison_design(left, right)
    if comparison_kind in {"lr_schedule", "early_stopping", "label_smoothing", "pretrained_backbone",
                           "pretrained_residual_layers", "pretrained_cifar_stem", "pretrained_cbam"}:
        for run in (left, right):
            _check_declared_lr_curve(run)
    if comparison_kind != "lr_schedule":
        for a, b in zip(left["history"], right["history"]):
            _close(a["lr"], b["lr"], f"第 {a['epoch']} 轮实际学习率轨迹", tolerance=1e-12)
    paired, transitions = [], {key: 0 for key in (
        "both_correct", "left_wrong_right_correct", "left_correct_right_wrong", "both_wrong")}
    for index in sorted(left["predictions"]):
        a, b = left["predictions"][index], right["predictions"][index]
        if a["true_label"] != b["true_label"]:
            raise ValueError(f"同一验证索引的真实标签不同：{index}")
        correct_a, correct_b = a["predicted_label"] == a["true_label"], b["predicted_label"] == b["true_label"]
        transition = ("both_correct" if correct_b else "left_correct_right_wrong") if correct_a else (
            "left_wrong_right_correct" if correct_b else "both_wrong")
        transitions[transition] += 1
        paired.append({"index": index, "true_label": a["true_label"],
                       "true_class": left["manifest"]["classes"][a["true_label"]],
                       "left_prediction": a["predicted_label"], "right_prediction": b["predicted_label"],
                       "left_confidence": a["confidence"], "right_confidence": b["confidence"],
                       "left_correct": correct_a, "right_correct": correct_b, "transition": transition})
    overview = [_overview(run) for run in (left, right)]
    per_class = []
    for a, b in zip(left["metrics"]["per_class"], right["metrics"]["per_class"]):
        row = {"class": a["name"], "support": a["support"]}
        for field in ("accuracy", "precision", "recall", "f1"):
            row.update({f"left_{field}": a[field], f"right_{field}": b[field],
                        f"delta_{field}_pp": 100 * (b[field] - a[field])})
        per_class.append(row)
    shared_protocol = {key: left["config"][key] for key in PROTOCOL_FIELDS}
    if left["config"]["scheduler"] == right["config"]["scheduler"]:
        shared_protocol["scheduler"] = left["config"]["scheduler"]
    if left["config"]["train_augmentation"] == right["config"]["train_augmentation"]:
        shared_protocol["train_augmentation"] = left["config"]["train_augmentation"]
    if left["config"]["label_smoothing"] == right["config"]["label_smoothing"]:
        shared_protocol["label_smoothing"] = left["config"]["label_smoothing"]
    if all(left["config"][key] == right["config"][key] for key in ("mixup_alpha", "mixup_probability")):
        shared_protocol.update({key: left["config"][key] for key in ("mixup_alpha", "mixup_probability")})
    if all(left["config"][key] == right["config"][key] for key in ("cutmix_alpha", "cutmix_probability")):
        shared_protocol.update({key: left["config"][key] for key in ("cutmix_alpha", "cutmix_probability")})
    if comparison_kind == "pretrained_backbone":
        shared_protocol.update(TRANSFER_PROTOCOL)
    if comparison_kind in {"pretrained_residual_layers", "pretrained_cifar_stem"}:
        shared_protocol.update(RESIDUAL_TRANSFER_PROTOCOL, matched_cutmix_batches=left["matched_cutmix_batches"])
    if comparison_kind == "pretrained_cbam":
        shared_protocol.update(input_size=32, preprocessing="cifar32", optimizer="sgd", pretrained=True,
                               pretrained_weights=TRANSFER_WEIGHTS, matched_cutmix_batches=left["matched_cutmix_batches"])
    if comparison_kind == "augmentation":
        design_note = "E1/E2 只改变训练随机裁剪和水平翻转，模型、参数量及初始状态哈希相同。"
    elif comparison_kind == "mixup":
        design_note = "E1/E3 保留相同基础增强，只改变 MixUp，模型、参数量及初始状态哈希相同。"
    elif comparison_kind == "cutmix":
        design_note = "E1/E4 保留相同基础增强，只改变 CutMix，模型、参数量及初始状态哈希相同。"
    elif comparison_kind == "mixup_vs_cutmix":
        design_note = "E3/E4 比较预设 MixUp 与 CutMix 两套方案，模型、参数量及初始状态哈希相同；alpha 与有效混合强度不完全匹配，不代表各自调参后的性能上限。"
    elif comparison_kind == "cbam":
        design_note = "E1/E5 共享基础增强与 ResNet 共有层的第 0 轮初始状态，E5 新增 CBAM 参数。"
    elif comparison_kind == "pretrained_cbam":
        design_note = "E11_stem 与 E11_stem_cbam 共享预训练骨干和分类头初始张量、随机状态与逐批 CutMix 记录。实验组在残差块加入 CBAM。"
    elif comparison_kind == "lr_schedule":
        design_note = "E1/E6 共享模型、基础增强和第 0 轮初始状态。E1 按轮次余弦衰减，E6 全程固定为初始学习率。min_learning_rate 仅用于 E1 的余弦策略。"
    elif comparison_kind == "early_stopping":
        design_note = "E1/E7 共享模型、初始状态和按计划轮数计算的余弦轨迹。E7 根据验证准确率和等待轮数决定是否提前结束。"
    elif comparison_kind == "label_smoothing":
        design_note = "E1/E8 共享模型、初始状态、基础增强和余弦学习率。E8 仅在训练更新时启用标签平滑，比较时采用两组统一的干净训练监测和验证准确率、普通交叉熵。"
    elif comparison_kind == "pretrained_backbone":
        design_note = "E9/E10 采用相同的 224×224 预处理和训练计划，第 0 轮分类头与数据加载随机状态一致。E9 的骨干随机初始化，E10 的骨干加载 ImageNet 预训练权重。"
    elif comparison_kind == "pretrained_residual_layers":
        design_note = "E4/E11 共享 32×32 输入、CIFAR ResNet-18、初始入口与分类头及逐批 CutMix 记录。E11 从 ImageNet 初始化四个残差阶段，两个模型均全量训练。"
    elif comparison_kind == "pretrained_cifar_stem":
        design_note = ("双方共享 32×32 输入、CIFAR ResNet-18、E4 第 0 轮分类头与逐批 CutMix 记录。"
                       "E11_stem 使用 ImageNet 的首层中心裁剪权重、入口 BatchNorm 和四个残差阶段。"
                       "E11/E11_stem 对照进一步校验四个残差阶段的初始张量一致。")
    else:
        design_note = "相同训练种子和增强策略不保证不同架构逐张获得相同的随机增强。"
    report = {
        "split": "validation", "data_manifest_sha256": digest(left["manifest"]),
        "comparison_kind": comparison_kind, "controlled_difference": controlled_difference,
        "shared_protocol": shared_protocol,
        "runs": overview, "validation_sample_count": len(paired),
        "right_minus_left_accuracy_pp": 100 * (overview[1]["best_val_accuracy"] - overview[0]["best_val_accuracy"]),
        "right_minus_left_macro_f1_pp": 100 * (overview[1]["macro_f1"] - overview[0]["macro_f1"]),
        "right_minus_left_validation_loss": overview[1]["best_val_loss"] - overview[0]["best_val_loss"],
        "right_minus_left_parameter_count": overview[1]["parameter_count"] - overview[0]["parameter_count"],
        "right_to_left_parameter_ratio": overview[1]["parameter_count"] / overview[0]["parameter_count"],
        "paired_counts": transitions,
        "notes": ["共享训练方案和数据遍历轮数不等于相同计算预算。",
                  design_note,
                  "这是单种子验证集比较，不是多种子稳定性或测试集结论。",
                  "历史逐轮耗时包含数据加载与验证、不含权重写盘，受运行负载影响；不作为严格推理速度基准。",
                  "checkpoint 文件大小包含优化器、随机状态和日志，不等于纯模型权重大小。"],
    }
    if any(_mixing_method(run["config"]) != "none" for run in (left, right)):
        report["notes"].append("MixUp/CutMix 的混合标签训练 loss 和加权训练准确率不能与普通训练直接比较；统一比较干净训练监测与验证指标。")
    if comparison_kind == "label_smoothing":
        report["notes"].append("两组训练目标使用不同的标签分布，训练损失曲线用于分别观察各自的优化过程。")
    if comparison_kind == "early_stopping":
        runs = {info["experiment"]: info for info in overview}
        baseline, stopped = runs["E1"], runs["E7"]
        time_field = "logged_epoch_seconds_excluding_checkpoint_io"
        report["early_stopping_savings"] = {
            "epochs_saved": baseline["completed_epochs"] - stopped["completed_epochs"],
            "logged_seconds_saved_excluding_checkpoint_io": baseline[time_field] - stopped[time_field],
        }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", report)
    _write_csv(output / "comparison.csv", overview)
    _write_csv(output / "per_class_comparison.csv", per_class)
    _write_csv(output / "paired_predictions.csv", paired)
    _plot_comparison((left, right), output / "curves.png")
    return report


def default_output_dir(run_dirs):
    """按实验对分开结果目录，保留既有 E0/E1、E1/E2 对照结果。"""
    names = [read_json(Path(path).resolve() / "config.json")["experiment"] for path in run_dirs]
    if any(not isinstance(name, str) or not name or
           not all(char.isalnum() or char in "_-" for char in name) for name in names):
        raise ValueError("默认输出目录需要安全的实验名称；请检查 experiment")
    if set(names) == {"E0", "E1"}:
        names = ["E0", "E1"]
    elif set(names) == {"E1", "E2"}:
        names = ["E1", "E2"]
    elif set(names) == {"E1", "E3"}:
        names = ["E1", "E3"]
    elif set(names) == {"E1", "E4"}:
        names = ["E1", "E4"]
    elif set(names) == {"E1", "E5"}:
        names = ["E1", "E5"]
    elif set(names) == {"E1", "E6"}:
        names = ["E1", "E6"]
    elif set(names) == {"E1", "E7"}:
        names = ["E1", "E7"]
    elif set(names) == {"E1", "E8"}:
        names = ["E1", "E8"]
    elif set(names) == TRANSFER_EXPERIMENTS:
        names = ["E9", "E10"]
    elif set(names) == RESIDUAL_TRANSFER_EXPERIMENTS:
        names = ["E4", "E11"]
    elif set(names) == STEM_CBAM_EXPERIMENTS:
        names = ["E11_stem", "E11_stem_cbam"]
    elif "E11_stem" in names and len(set(names) & STEM_TRANSFER_BASELINES) == 1:
        names = [next(iter(set(names) & STEM_TRANSFER_BASELINES)), "E11_stem"]
    elif set(names) == {"E3", "E4"}:
        names = ["E3", "E4"]
    return ROOT / "results" / "_vs_".join(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs=2, required=True, metavar=("LEFT", "RIGHT"))
    parser.add_argument("--output-dir", type=Path, help="默认按实验名称输出到 results/<LEFT>_vs_<RIGHT>")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.run_dirs)
    report = compare_runs(args.run_dirs, args.output_dir)
    for run in report["runs"]:
        print(f"{run['experiment']}: best epoch={run['best_epoch']}, validation accuracy={run['best_val_accuracy']:.2%}")
        if run["experiment"] == "E7":
            print(f"E7: 完成 {run['completed_epochs']}/{run['epochs']} 轮，结束原因 {run['stop_reason']}")
        if run["experiment"] in TRANSFER_EXPERIMENTS:
            first = run["first_epoch_val_ge_90pct"]
            reached = f"第 {first} 轮" if first is not None else "本次未达到"
            print(f"{run['experiment']}: 首次达到 90% 验证准确率 {reached}")
    print(f"右组减左组：{report['right_minus_left_accuracy_pp']:+.2f} 个百分点")
    print("对照结果：", args.output_dir.resolve())


if __name__ == "__main__":
    main()
