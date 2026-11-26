"""配置、随机状态和输出文件；路径统一相对于项目根目录。"""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from src.early_stopping import validate_early_stopping

ROOT = Path(__file__).resolve().parents[1]
RESNET18_IMAGENET1K_V1_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


CONFIG_FIELDS = frozenset({
    "experiment", "seed", "split_seed", "monitor_seed", "data_dir", "output_dir",
    "epochs", "batch_size", "num_workers", "learning_rate", "momentum",
    "weight_decay", "min_learning_rate", "dropout", "amp", "log_interval",
})
MODEL_NAMES = frozenset({"simple_cnn", "cifar_resnet18", "cifar_resnet18_cbam", "standard_resnet18"})
CIFAR_PRETRAINED_EXPERIMENTS = {"E11", "E11_stem", "E11_stem_cbam"}


def resolve_model_name(config):
    """旧 E0 没有 model 字段；其它实验必须显式声明，避免误用网络。"""
    name = config.get("model")
    if name is None and config.get("experiment", "E0") == "E0":
        name = "simple_cnn"
    if not isinstance(name, str) or name not in MODEL_NAMES:
        raise ValueError(f"请显式指定支持的 model：{', '.join(sorted(MODEL_NAMES))}。")
    required_model = {"E0": "simple_cnn", "E1": "cifar_resnet18",
                      "E2": "cifar_resnet18", "E3": "cifar_resnet18",
                      "E4": "cifar_resnet18", "E5": "cifar_resnet18_cbam",
                      "E6": "cifar_resnet18", "E7": "cifar_resnet18",
                      "E8": "cifar_resnet18", "E9": "standard_resnet18",
                      "E10": "standard_resnet18", "E11": "cifar_resnet18",
                      "E11_stem": "cifar_resnet18",
                      "E11_stem_cbam": "cifar_resnet18_cbam"}.get(config.get("experiment"))
    if required_model is not None and name != required_model:
        raise ValueError(f"{config['experiment']} 与模型 {name} 不匹配")
    return name


def normalize_label_smoothing(value, experiment=None):
    if type(value) not in (int, float) or not 0 <= value < 1:
        raise ValueError("label_smoothing 必须位于 [0, 1)")
    if experiment is not None and (experiment == "E8") != (value > 0):
        raise ValueError("E8 必须启用 label_smoothing，其他实验请设为 0")
    return float(value)


def normalize_config(config):
    """统一比较旧配置与新配置，不改写已有 E0 配置文件。"""
    config = dict(config)
    missing = CONFIG_FIELDS - config.keys()
    extra = config.keys() - CONFIG_FIELDS - {
        "model", "smoke", "device", "train_augmentation", "initial_checkpoint",
        "mixup_alpha", "mixup_probability",
        "cutmix_alpha", "cutmix_probability",
        "cbam_reduction", "cbam_spatial_kernel_size",
        "scheduler", "early_stopping", "label_smoothing",
        "optimizer", "preprocessing", "input_size", "pretrained", "pretrained_weights",
    }
    if missing or extra:
        raise ValueError(f"配置字段错误，缺少 {sorted(missing)}，未知 {sorted(extra)}")
    if not isinstance(config["experiment"], str) or not config["experiment"].strip():
        raise ValueError("experiment 必须是非空字符串")
    config["model"] = resolve_model_name(config)
    config.setdefault("smoke", False)
    config.setdefault("device", "cuda")
    # 历史 E0/E1 未保存此字段，但始终使用基础增强。
    config.setdefault("train_augmentation", True)
    config.setdefault("initial_checkpoint", None)
    config.setdefault("mixup_alpha", 0.0)
    config.setdefault("mixup_probability", 0.0)
    config.setdefault("cutmix_alpha", 0.0)
    config.setdefault("cutmix_probability", 0.0)
    config.setdefault("cbam_reduction", None)
    config.setdefault("cbam_spatial_kernel_size", None)
    config.setdefault("scheduler", "cosine")
    config.setdefault("optimizer", "sgd")
    config.setdefault("preprocessing", "cifar32")
    config.setdefault("input_size", 32)
    config.setdefault("pretrained", False)
    config.setdefault("pretrained_weights", None)
    config["early_stopping"] = validate_early_stopping(config.get("early_stopping"))
    if config["model"] == "cifar_resnet18_cbam":
        for field, required in (("cbam_reduction", 16), ("cbam_spatial_kernel_size", 7)):
            if type(config[field]) is not int or config[field] != required:
                raise ValueError(f"CBAM 要求 {field}={required}")
    elif config["cbam_reduction"] is not None or config["cbam_spatial_kernel_size"] is not None:
        raise ValueError("只有 CBAM 模型可设置 cbam_reduction 或 cbam_spatial_kernel_size")
    for field in ("amp", "smoke", "train_augmentation", "pretrained"):
        if type(config[field]) is not bool:
            raise ValueError(f"{field} 必须是 true 或 false")
    experiment = config["experiment"]
    config["label_smoothing"] = normalize_label_smoothing(config.get("label_smoothing", 0.0), experiment)
    if (experiment == "E7") != (config["early_stopping"] is not None):
        raise ValueError("E7 必须配置 early_stopping，其他实验请关闭早停")
    optimizer = config["optimizer"]
    if not isinstance(optimizer, str) or optimizer not in {"sgd", "adamw"}:
        raise ValueError("optimizer 必须是 sgd 或 adamw")
    if experiment in ({"E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"} | CIFAR_PRETRAINED_EXPERIMENTS) and optimizer != "sgd":
        raise ValueError(f"{experiment} 要求 optimizer=sgd")
    if experiment in {"E9", "E10"} and optimizer != "adamw":
        raise ValueError(f"{experiment} 要求 optimizer=adamw")
    preprocessing = config["preprocessing"]
    if not isinstance(preprocessing, str) or preprocessing not in {"cifar32", "imagenet224"}:
        raise ValueError("preprocessing 必须是 cifar32 或 imagenet224")
    expected_size = 32 if preprocessing == "cifar32" else 224
    if type(config["input_size"]) is not int or config["input_size"] != expected_size:
        raise ValueError(f"preprocessing={preprocessing} 要求 input_size={expected_size}")
    if (experiment in ({"E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"} | CIFAR_PRETRAINED_EXPERIMENTS)
            and preprocessing != "cifar32"):
        raise ValueError(f"{experiment} 要求 preprocessing=cifar32")
    if experiment in {"E9", "E10"} and preprocessing != "imagenet224":
        raise ValueError(f"{experiment} 要求 preprocessing=imagenet224 与 input_size=224")
    if experiment in {"E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"} and config["pretrained"]:
        raise ValueError(f"{experiment} 不使用预训练权重，pretrained 必须为 false")
    if experiment == "E9" and config["pretrained"]:
        raise ValueError("E9 为随机初始化对照，pretrained 必须为 false；预训练初始化请使用 E10")
    if experiment in ({"E10"} | CIFAR_PRETRAINED_EXPERIMENTS):
        if not config["pretrained"] or config["pretrained_weights"] != "IMAGENET1K_V1":
            raise ValueError(f"{experiment} 要求 pretrained=true 和 pretrained_weights=IMAGENET1K_V1")
    elif config["pretrained_weights"] is not None:
        raise ValueError("pretrained_weights 仅用于 E10 和 E11 系列")
    scheduler = config["scheduler"]
    if not isinstance(scheduler, str) or scheduler not in {"cosine", "fixed"}:
        raise ValueError("scheduler 必须是 cosine 或 fixed")
    if experiment == "E6" and scheduler != "fixed":
        raise ValueError("E6 要求 scheduler=fixed，整个训练使用初始学习率")
    if (experiment in ({"E0", "E1", "E2", "E3", "E4", "E5", "E7", "E8", "E9", "E10"} | CIFAR_PRETRAINED_EXPERIMENTS)
            and scheduler != "cosine"):
        raise ValueError(f"{experiment} 要求 scheduler=cosine；固定学习率对照请使用 E6")
    if (experiment in ({"E0", "E1", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"} | CIFAR_PRETRAINED_EXPERIMENTS)
            and not config["train_augmentation"]):
        raise ValueError(f"{experiment} 基线必须开启训练增强；无增强对照请使用 E2")
    if experiment == "E2" and config["train_augmentation"]:
        raise ValueError("E2 必须设置 train_augmentation=false，关闭随机裁剪和水平翻转")
    initial = config["initial_checkpoint"]
    if initial is not None:
        if not isinstance(initial, str) or not initial.strip():
            raise ValueError("initial_checkpoint 必须是非空路径字符串或 null")
        if experiment in {"E0", "E1", "E9"}:
            raise ValueError("E0/E1/E9 基线从随机初始化开始，不接受 initial_checkpoint")
        config["initial_checkpoint"] = str((ROOT / initial).resolve())
    if experiment in {"E2", "E3", "E4", "E5", "E6", "E7", "E8"} and initial is None:
        raise ValueError(f"{experiment} 需要 initial_checkpoint 指向对应 E1 的 initial.pt，以保持初始权重一致")
    if experiment == "E10" and initial is None:
        raise ValueError("E10 需要 initial_checkpoint 指向对应 E9 的 initial.pt，以保持分类头初始化一致")
    if experiment in {"E11", "E11_stem"} and initial is None:
        raise ValueError(f"{experiment} 需要 initial_checkpoint 指向对应 E4 的 initial.pt")
    if experiment == "E11_stem_cbam" and initial is None:
        raise ValueError("E11_stem_cbam 需要 initial_checkpoint 指向对应 E11_stem 的 initial.pt")
    for field in ("mixup_alpha", "mixup_probability"):
        value = config[field]
        if type(value) not in (float, int) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{field} 必须是有限非负数")
    if config["mixup_probability"] > 1:
        raise ValueError("mixup_probability 必须位于 0–1")
    if (config["mixup_alpha"] == 0) != (config["mixup_probability"] == 0):
        raise ValueError("MixUp 关闭时 alpha 和 probability 必须同时为 0，开启时必须同时大于 0")
    if (experiment in ({"E0", "E1", "E2", "E4", "E5", "E6", "E7", "E8", "E9", "E10"} | CIFAR_PRETRAINED_EXPERIMENTS)
            and config["mixup_alpha"] != 0):
        raise ValueError(f"{experiment} 不使用 MixUp；请使用独立 E3 配置")
    if experiment == "E3" and (config["mixup_alpha"] <= 0 or config["mixup_probability"] != 1):
        raise ValueError("E3 要求 mixup_alpha>0 且 mixup_probability=1，每个训练 batch 执行 MixUp")
    for field in ("cutmix_alpha", "cutmix_probability"):
        value = config[field]
        if type(value) not in (float, int) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{field} 必须是有限非负数")
    if config["cutmix_probability"] > 1:
        raise ValueError("cutmix_probability 必须位于 0–1")
    if (config["cutmix_alpha"] == 0) != (config["cutmix_probability"] == 0):
        raise ValueError("CutMix 关闭时 alpha 和 probability 必须同时为 0，开启时必须同时大于 0")
    if config["cutmix_alpha"] > 0 and config["mixup_alpha"] > 0:
        raise ValueError("MixUp 与 CutMix 必须互斥，单项对照不能同时开启")
    if experiment in {"E0", "E1", "E2", "E3", "E5", "E6", "E7", "E8", "E9", "E10"} and config["cutmix_alpha"] != 0:
        raise ValueError(f"{experiment} 不使用 CutMix；请使用独立 E4 配置")
    if experiment == "E4" and (config["cutmix_alpha"] <= 0 or config["cutmix_probability"] != 1):
        raise ValueError("E4 要求 cutmix_alpha>0 且 cutmix_probability=1，每个训练 batch 执行 CutMix")
    if experiment in CIFAR_PRETRAINED_EXPERIMENTS and (config["cutmix_alpha"] != 1 or config["cutmix_probability"] != 1):
        raise ValueError(f"{experiment} 要求沿用 E4 的 cutmix_alpha=1 和 cutmix_probability=1")
    if config["device"] not in {"cpu", "cuda"}:
        raise ValueError("device 必须是 cpu 或 cuda")
    for field in ("seed", "split_seed", "monitor_seed"):
        if type(config[field]) is not int or not 0 <= config[field] < 2 ** 32:
            raise ValueError(f"{field} 必须是 [0, 2**32) 内的整数")
    for field in ("epochs", "batch_size", "log_interval"):
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f"{field} 必须是正整数")
    if type(config["num_workers"]) is not int or config["num_workers"] < 0:
        raise ValueError("num_workers 必须是非负整数")
    for field in ("learning_rate", "momentum", "weight_decay", "min_learning_rate", "dropout"):
        value = config[field]
        if type(value) not in (float, int) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{field} 必须是有限非负数")
    if config["dropout"] >= 1 or config["momentum"] >= 1:
        raise ValueError("dropout 和 momentum 必须小于 1")
    if optimizer == "adamw" and config["momentum"] != 0:
        raise ValueError("AdamW 不使用 SGD momentum，请将 momentum 设为 0")
    if config["model"] == "standard_resnet18" and config["dropout"] != 0:
        raise ValueError("standard_resnet18 保留标准分类头，请将 dropout 设为 0")
    if not 0 <= config["min_learning_rate"] <= config["learning_rate"] or config["learning_rate"] == 0:
        raise ValueError("学习率必须为正，最小学习率不能超过初始值")
    for field in ("data_dir", "output_dir"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} 必须是非空路径字符串")
        config[field] = str((ROOT / config[field]).resolve())
    return config


def load_config(path):
    return normalize_config(read_json(path))


def seed_everything(seed):
    import torch
    # CUDA 矩阵乘法的确定性设置应在第一次 GPU 运算之前完成。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def choose_device(name):
    import torch
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。先运行 check_environment.py；CPU 联调可显式传 --device cpu。")
    return torch.device(name)


def capture_rng(generators):
    import torch
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loaders": {key: gen.get_state() for key, gen in generators.items()},
    }


def restore_rng(state, generators):
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    for key, gen in generators.items():
        gen.set_state(state["loaders"][key])


def environment_info():
    import torch
    import torchvision
    return {"python": sys.version, "executable": sys.executable,
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
