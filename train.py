"""E0–E11 训练入口。先 --smoke 检查，再正式训练；不自动访问测试集。"""
import argparse
import copy
import csv
import hashlib
import io
import os
import shutil
import time
from pathlib import Path

if __name__ == "__main__":
    print("正在加载训练依赖，首次启动可能需要等待一段时间……", flush=True)

import torch

from src.common import (RESNET18_IMAGENET1K_V1_SHA256, ROOT, capture_rng, choose_device,
                        environment_info, load_config, normalize_config, read_json, restore_rng,
                        seed_everything, write_json)
from src.data import load_manifest, make_loaders, manifest_hash, normalization_for
from src.early_stopping import new_early_stopping_state, replay_early_stopping, update_early_stopping
from src.engine import evaluate_model, save_checkpoint, train_one_epoch
from src.models import build_model, model_name_from_checkpoint, state_dict_sha256
from src.models.factory import RESNET_RESIDUAL_PREFIXES

RESNET_HEAD_KEYS = frozenset({"fc.weight", "fc.bias"})
CIFAR_STEM_TRANSFER_SCOPE = "stem-and-layer1-layer4"
CIFAR_STEM_KERNEL_ADAPTATION = "center_crop_7_to_3"


def resnet_state_parts(state):
    head = {name: state[name] for name in RESNET_HEAD_KEYS}
    backbone = {name: value for name, value in state.items() if name not in RESNET_HEAD_KEYS}
    return head, backbone


def cifar_resnet_state_parts(state, *, transfer_stem=False):
    retained, transferred = {}, {}
    for name, tensor in state.items():
        use_pretrained = not name.startswith("fc.") if transfer_stem else name.startswith(RESNET_RESIDUAL_PREFIXES)
        (transferred if use_pretrained else retained)[name] = tensor
    return retained, transferred


def snapshot_sources(run_dir, config_path):
    """保存可复原的源文件副本；后续扩展模型时仍能追溯本次代码。"""
    sources = sorted((ROOT / "src").rglob("*.py")) + sorted(ROOT.glob("*.py"))
    hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        target = run_dir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(config_path, run_dir / "source_snapshot" / "input_config.json")
    return hashes


def write_history(path, history):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(history[0]))
    writer.writeheader()
    writer.writerows(history)
    path = Path(path)
    temporary = path.with_suffix(".csv.tmp")
    temporary.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    os.replace(temporary, path)


def cbam_shared_state(state):
    """CBAM 子模块之外的参数和缓冲区沿用原模型的名称。"""
    return {name: value for name, value in state.items() if ".cbam." not in name}


def load_cbam_shared_weights(model, source_state, reference_experiment):
    destination = model.state_dict()
    shared = cbam_shared_state(destination)
    if (len(shared) == len(destination) or shared.keys() != source_state.keys()
            or any(value.shape != source_state[name].shape or value.dtype != source_state[name].dtype
                   for name, value in shared.items())):
        raise ValueError(f"CBAM 与 {reference_experiment} 的共有层名称、形状或类型不匹配")
    model.load_state_dict({**destination, **source_state}, strict=True)


def stem_pretrained_metadata(initial, shared_state):
    weights = initial.get("pretrained_weight_source") or {}
    classifier, backbone = cifar_resnet_state_parts(shared_state, transfer_stem=True)
    head_hash = state_dict_sha256(classifier)
    if (initial.get("initial_classifier_state_sha256") != head_hash
            or weights.get("enum") != "ResNet18_Weights.IMAGENET1K_V1"
            or weights.get("sha256") != RESNET18_IMAGENET1K_V1_SHA256
            or weights.get("scope") != CIFAR_STEM_TRANSFER_SCOPE
            or weights.get("stem_kernel_adaptation") != CIFAR_STEM_KERNEL_ADAPTATION
            or weights.get("backbone_state_sha256") != state_dict_sha256(backbone)):
        raise ValueError("首层适配的分类头或 ImageNet 骨干来源与初始权重不同")
    return {"pretrained_weight_source": weights, "initial_classifier_state_sha256": head_hash}


def verify_initial_checkpoint(run_dir, state, config):
    """新运行的起点必须可追溯；历史 E0 没保存过起点时明确允许缺省。"""
    expected = state.get("initial_state_sha256")
    if expected is None:
        if config["model"] != "simple_cnn":
            raise ValueError("新模型的续训权重缺少初始状态哈希")
        return
    path = run_dir / "initial.pt"
    if not path.is_file():
        raise FileNotFoundError("续训需要原始 initial.pt；请恢复该文件后继续。")
    initial = torch.load(path, map_location="cpu", weights_only=False)
    if (initial.get("epoch") != 0 or model_name_from_checkpoint(initial) != config["model"]
            or initial.get("manifest_hash") != state["manifest_hash"]
            or normalize_config(initial["config"]) != config
            or initial.get("state_sha256") != expected
            or state_dict_sha256(initial["model"]) != expected):
        raise ValueError("initial.pt 的模型、配置、数据或初始参数哈希不匹配")
    if config["model"] == "cifar_resnet18_cbam":
        shared = cbam_shared_state(initial["model"])
        shared_hash = state_dict_sha256(shared)
        reference_experiment = "E11_stem" if config["experiment"] == "E11_stem_cbam" else "E1"
        source = state.get("initial_checkpoint_source") or {}
        if (shared_hash != initial.get("shared_state_sha256")
                or shared_hash != state.get("initial_shared_state_sha256")
                or initial.get("initial_checkpoint_source") != source
                or source.get("state_sha256") != shared_hash
                or (source.get("experiment"), source.get("epoch"), source.get("seed"))
                != (reference_experiment, 0, config["seed"])):
            raise ValueError(f"CBAM 共有层的初始哈希或 {reference_experiment} 来源与原运行不同")
        if reference_experiment == "E11_stem":
            metadata = stem_pretrained_metadata(initial, shared)
            if any(state.get(key) != value for key, value in metadata.items()):
                raise ValueError("E11_stem_cbam 的预训练来源与原运行不同")
    if config["experiment"] == "E10":
        source = state.get("initial_checkpoint_source") or {}
        weights = state.get("pretrained_weight_source") or {}
        head, backbone = resnet_state_parts(initial["model"])
        head_hash = state_dict_sha256(head)
        if (initial.get("initial_checkpoint_source") != source
                or initial.get("pretrained_weight_source") != weights
                or initial.get("initial_classifier_state_sha256") != head_hash
                or state.get("initial_classifier_state_sha256") != head_hash
                or source.get("classifier_state_sha256") != head_hash
                or (source.get("experiment"), source.get("epoch"), source.get("seed")) != ("E9", 0, config["seed"])
                or weights.get("enum") != "ResNet18_Weights.IMAGENET1K_V1"
                or weights.get("backbone_state_sha256") != state_dict_sha256(backbone)):
            raise ValueError("E10 续训的初始分类头或 ImageNet 骨干来源与原运行不同")
    if config["experiment"] in {"E11", "E11_stem"}:
        source = state.get("initial_checkpoint_source") or {}
        weights = state.get("pretrained_weight_source") or {}
        transfer_stem = config["experiment"] == "E11_stem"
        reference_field = "classifier_state_sha256" if transfer_stem else "retained_state_sha256"
        checkpoint_field = "initial_" + reference_field
        scope = CIFAR_STEM_TRANSFER_SCOPE if transfer_stem else "layer1-layer4"
        retained, transferred = cifar_resnet_state_parts(initial["model"], transfer_stem=transfer_stem)
        retained_hash = state_dict_sha256(retained)
        if (initial.get("initial_checkpoint_source") != source
                or initial.get("pretrained_weight_source") != weights
                or initial.get(checkpoint_field) != retained_hash
                or state.get(checkpoint_field) != retained_hash
                or source.get(reference_field) != retained_hash
                or (source.get("experiment"), source.get("epoch"), source.get("seed")) != ("E4", 0, config["seed"])
                or weights.get("enum") != "ResNet18_Weights.IMAGENET1K_V1"
                or weights.get("sha256") != RESNET18_IMAGENET1K_V1_SHA256
                or weights.get("scope") != scope
                or (transfer_stem and weights.get("stem_kernel_adaptation") != CIFAR_STEM_KERNEL_ADAPTATION)
                or weights.get("backbone_state_sha256") != state_dict_sha256(transferred)):
            raise ValueError(f"{config['experiment']} 续训的 E4 共有初始状态或 ImageNet 权重来源与原运行不同")


def load_initial_weights(model, config, manifest):
    """消融实验复用 E1 第 0 轮参数；不载入训练后权重或优化器状态。"""
    reference = config.get("initial_checkpoint")
    if reference is None:
        return None
    path = Path(reference)
    if not path.is_file():
        raise FileNotFoundError(f"缺少 E1 初始状态：{path}。请恢复对应的 initial.pt，不要改用 best.pt。")
    initial = torch.load(path, map_location="cpu", weights_only=False)
    source_config = normalize_config(initial["config"])
    use_cbam = config["model"] == "cifar_resnet18_cbam"
    reference_model = "cifar_resnet18" if use_cbam else config["model"]
    if (initial.get("epoch") != 0 or source_config["experiment"] != "E1"
            or source_config["smoke"] or model_name_from_checkpoint(initial) != reference_model):
        raise ValueError("initial_checkpoint 必须是与当前模型相配的正式 E1 第 0 轮 initial.pt")
    if initial.get("manifest_hash") != manifest_hash(manifest):
        raise ValueError(f"E1 初始状态的数据划分或归一化与 {config['experiment']} 不同")
    for field in ("seed", "split_seed", "monitor_seed", "dropout"):
        if source_config[field] != config[field]:
            raise ValueError(f"E1 初始状态与 {config['experiment']} 的 {field} 不同")
    actual_hash = state_dict_sha256(initial["model"])
    if actual_hash != initial.get("state_sha256"):
        raise ValueError("E1 initial.pt 的初始参数内容哈希不匹配")
    if any(not torch.isfinite(value).all().item() for value in initial["model"].values()):
        raise ValueError("E1 initial.pt 含有非有限参数")
    source_state = initial["model"]
    if use_cbam:
        load_cbam_shared_weights(model, source_state, "E1")
    else:
        model.load_state_dict(source_state, strict=True)
    return {"path": str(path.resolve()), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "state_sha256": actual_hash, "experiment": source_config["experiment"],
            "seed": source_config["seed"], "epoch": 0}


def load_pretraining_reference(model_state, config, manifest, experiment):
    path = Path(config["initial_checkpoint"])
    if not path.is_file():
        raise FileNotFoundError(f"缺少对应 {experiment} 第 0 轮 initial.pt：{path}")
    initial = torch.load(path, map_location="cpu", weights_only=False)
    reference = normalize_config(initial["config"])
    adds_cbam = config["experiment"] == "E11_stem_cbam"
    reference_model = "cifar_resnet18" if adds_cbam else config["model"]
    if (initial.get("epoch") != 0 or reference["experiment"] != experiment or reference["smoke"]
            or model_name_from_checkpoint(initial) != reference_model):
        raise ValueError(f"{config['experiment']} 需要正式 {experiment} 第 0 轮的 initial.pt")
    if initial["manifest_hash"] != manifest_hash(manifest):
        raise ValueError(f"{experiment} 初始状态的数据清单与 {config['experiment']} 不同")
    excluded = {"experiment", "initial_checkpoint", "output_dir", "data_dir", "log_interval", "smoke"}
    if adds_cbam:
        excluded.update({"model", "cbam_reduction", "cbam_spatial_kernel_size"})
    else:
        excluded.update({"pretrained", "pretrained_weights"})
    if config["experiment"] == "E11_stem":
        # 首层适配可以用自己的微调学习率，仍复用 E4 第 0 轮的分类头。
        excluded.add("learning_rate")
    if config["smoke"]:
        excluded.update({"epochs", "device", "num_workers"})
    differences = sorted(key for key in config.keys() - excluded if config[key] != reference[key])
    if differences:
        raise ValueError(f"{experiment}/{config['experiment']} 的训练协议不同：" + ", ".join(differences))
    reference_state = initial["model"]
    reference_hash = state_dict_sha256(reference_state)
    if reference_hash != initial.get("state_sha256"):
        raise ValueError(f"{experiment} initial.pt 的模型状态哈希不匹配")
    if reference_state.keys() != model_state.keys():
        raise ValueError(f"{experiment} initial.pt 的张量名称与 {config['experiment']} 模型不匹配")
    with path.open("rb") as file:
        reference_file_hash = hashlib.file_digest(file, "sha256").hexdigest()
    return initial, {"path": str(path.resolve()), "file_sha256": reference_file_hash,
                     "state_sha256": reference_hash, "experiment": experiment, "epoch": 0,
                     "seed": reference["seed"]}


def load_e10_initial_weights(model, config, manifest):
    """严格复用 E9 第 0 轮分类头，并从官方 ImageNet 权重安装骨干。"""
    from src.models.standard_resnet import load_imagenet_resnet18

    model_state = model.state_dict()
    reference, source = load_pretraining_reference(model_state, config, manifest, "E9")
    head, _ = resnet_state_parts(reference["model"])
    if any(tensor.shape != model_state[name].shape or tensor.dtype != model_state[name].dtype
           or not torch.isfinite(tensor).all().item() for name, tensor in head.items()):
        raise ValueError("E9 初始分类头的张量形状、类型或取值无效")

    weights, pretrained_source = load_imagenet_resnet18(config["data_dir"])
    # 保留官方状态的版本元数据，由 PyTorch 严格加载并处理旧格式的 BN 缓冲区。
    state = copy.copy(weights)
    state.update(head)
    model.load_state_dict(state, strict=True)
    loaded_head, loaded_backbone = resnet_state_parts(model.state_dict())
    head_hash = state_dict_sha256(head)
    if state_dict_sha256(loaded_head) != head_hash:
        raise ValueError("E10 未能逐张量复用 E9 初始分类头")
    return {**source, "classifier_state_sha256": head_hash}, {
                **pretrained_source, "backbone_state_sha256": state_dict_sha256(loaded_backbone),
            }


def load_e11_initial_weights(model, config, manifest):
    """在 CIFAR 模型上从 ImageNet 初始化残差阶段，以及可选的适配入口。"""
    from src.models.standard_resnet import load_imagenet_resnet18

    transfer_stem = config["experiment"] == "E11_stem"
    model_state = model.state_dict()
    reference, source = load_pretraining_reference(model_state, config, manifest, "E4")
    retained, backbone = cifar_resnet_state_parts(reference["model"], transfer_stem=transfer_stem)
    if any(tensor.shape != model_state[name].shape or tensor.dtype != model_state[name].dtype
           or not torch.isfinite(tensor).all().item() for name, tensor in retained.items()):
        raise ValueError("E4 初始入口或分类头的张量形状、类型或取值无效")

    weights, pretrained_source = load_imagenet_resnet18(config["data_dir"])
    # 官方旧版没有 BN 批计数，沿用 E4 第 0 轮的计数。
    missing = [name for name in backbone if name not in weights and not name.endswith(".num_batches_tracked")]
    if missing:
        raise ValueError(f"ImageNet 权重缺少 {config['experiment']} 所需张量：" + ", ".join(missing))
    transferred = {name: weights.get(name, tensor) for name, tensor in backbone.items()}
    if transfer_stem:
        kernel = transferred["conv1.weight"]
        target = model_state["conv1.weight"]
        if (target.ndim != 4 or target.shape[-2:] != (3, 3)
                or kernel.shape != (*target.shape[:2], 7, 7)):
            raise ValueError("E11_stem 首层卷积需要同通道数的 ImageNet 7×7 核和 CIFAR 3×3 核")
        transferred["conv1.weight"] = kernel[..., 2:5, 2:5].contiguous()
    if any(tensor.shape != model_state[name].shape or tensor.dtype != model_state[name].dtype
           for name, tensor in transferred.items()):
        raise ValueError(f"ImageNet 迁移张量的形状或类型与 {config['experiment']} 不匹配")
    model.load_state_dict({**retained, **transferred}, strict=True)
    loaded_retained, loaded_backbone = cifar_resnet_state_parts(model.state_dict(), transfer_stem=transfer_stem)
    retained_hash = state_dict_sha256(retained)
    backbone_hash = state_dict_sha256(loaded_backbone)
    if state_dict_sha256(loaded_retained) != retained_hash or backbone_hash == state_dict_sha256(backbone):
        raise ValueError(f"{config['experiment']} 需要复用 E4 的共有张量并更新迁移范围内的初始状态")
    metadata = {**pretrained_source, "backbone_state_sha256": backbone_hash}
    if transfer_stem:
        metadata.update(scope=CIFAR_STEM_TRANSFER_SCOPE, stem_kernel_adaptation=CIFAR_STEM_KERNEL_ADAPTATION)
        return {**source, "classifier_state_sha256": retained_hash}, metadata
    return {**source, "retained_state_sha256": retained_hash}, {**metadata, "scope": "layer1-layer4"}


def load_e11_stem_cbam_initial_weights(model, config, manifest):
    shared = cbam_shared_state(model.state_dict())
    reference, source = load_pretraining_reference(shared, config, manifest, "E11_stem")
    reference_state = reference["model"]
    metadata = stem_pretrained_metadata(reference, reference_state)
    e4_source = reference.get("initial_checkpoint_source") or {}
    if ((e4_source.get("experiment"), e4_source.get("epoch"), e4_source.get("seed")) != ("E4", 0, config["seed"])
            or e4_source.get("classifier_state_sha256") != metadata["initial_classifier_state_sha256"]):
        raise ValueError("E11_stem initial.pt 的 E4 分类头来源无效")
    if any(not torch.isfinite(value).all().item() for value in reference_state.values()):
        raise ValueError("E11_stem initial.pt 含有非有限参数")
    load_cbam_shared_weights(model, reference_state, "E11_stem")
    return source, metadata


def restore_early_stopping(checkpoint, settings, planned_epochs):
    epoch = checkpoint["epoch"]
    history = checkpoint["history"]
    state = replay_early_stopping(settings, history)
    if len(history) != epoch or state != checkpoint.get("early_stopping_state"):
        raise ValueError("早停续训状态与已保存的逐轮验证结果不一致")
    reason = checkpoint.get("stop_reason")
    if state["stopped_epoch"] is not None:
        valid_reasons = {"early_stopping"}
    elif epoch == planned_epochs:
        valid_reasons = {"max_epochs"}
    else:
        valid_reasons = {None, "stop_after"}
    if epoch > planned_epochs or reason not in valid_reasons:
        raise ValueError("早停续训权重的结束原因或轮次不匹配")
    return state, reason


def build_optimizer(model, config):
    if config["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    return torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                           momentum=config["momentum"], weight_decay=config["weight_decay"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/e0.json"))
    parser.add_argument("--smoke", action="store_true", help="1000/500/500 样本、默认2轮，结果独立保存")
    parser.add_argument("--epochs", type=int, help="覆盖总轮数；使用余弦调度时也会改变调度总长度")
    parser.add_argument("--stop-after", type=int, help="本次最多运行几轮，保持配置总轮数/调度不变，之后可 --resume")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--run-name", help="自定义输出子目录名，便于另开一次实验")
    parser.add_argument("--resume", action="store_true", help="从该输出目录的 last.pt 恢复")
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after 必须为正整数")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs 必须为正整数")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers 不能为负数")
    if args.run_name and (args.run_name in {".", ".."} or any(c in args.run_name for c in '/\\:')):
        parser.error("--run-name 只能是一个子目录名")
    config = load_config(args.config)
    config["epochs"] = args.epochs if args.epochs is not None else (2 if args.smoke else config["epochs"])
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    config.update(smoke=args.smoke, device=args.device)
    seed_everything(config["seed"])
    device = choose_device(args.device)
    print(f"正在校验 {config['experiment']} 数据清单，模型：{config['model']}……", flush=True)
    manifest = load_manifest(config)
    name = args.run_name or f"{'smoke_' if args.smoke else ''}seed{config['seed']}"
    run_dir = Path(config["output_dir"]) / name
    early_stopping = config["early_stopping"]
    early_state = new_early_stopping_state() if early_stopping is not None else None
    stop_reason = None
    finished = False
    resume_state = None
    if args.resume:
        if normalize_config(read_json(run_dir / "config.json")) != config:
            raise ValueError("续训配置与原运行不同。请保持配置和 --smoke/--epochs/--device 等参数一致。")
        if not (run_dir / "last.pt").is_file():
            raise FileNotFoundError("没有完整的 last.pt；请另取 --run-name 重新联调。")
        resume_state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        if model_name_from_checkpoint(resume_state) != config["model"]:
            raise ValueError("续训权重的模型与当前配置不一致")
        if normalize_config(resume_state["config"]) != config:
            raise ValueError("续训权重中的配置与当前配置不一致")
        if resume_state["manifest_hash"] != manifest_hash(manifest):
            raise ValueError("续训数据清单与原运行不同")
        verify_initial_checkpoint(run_dir, resume_state, config)
        if early_stopping is not None:
            early_state, stop_reason = restore_early_stopping(resume_state, early_stopping, config["epochs"])
            finished = stop_reason in {"early_stopping", "max_epochs"}
            if finished:
                summary_path = run_dir / "summary.json"
                summary = read_json(summary_path) if summary_path.is_file() else {}
                if (summary.get("complete") is True and summary.get("stop_reason") == stop_reason
                        and summary.get("completed_epochs") == resume_state["epoch"]
                        and summary.get("early_stopping_state") == early_state
                        and (run_dir / "history.csv").is_file()):
                    print(f"该运行已在第 {resume_state['epoch']} 轮结束（{stop_reason}）。", flush=True)
                    return
        elif resume_state["epoch"] >= config["epochs"]:
            print(f"该运行已完成 {resume_state['epoch']} 轮，达到计划轮数；已有文件保持原样。", flush=True)
            return
    else:
        if run_dir.exists():
            raise FileExistsError(f"输出目录已存在：{run_dir}。继续请加 --resume；另开运行请指定 --run-name。")
    loaders, generators = make_loaders(config, manifest, args.smoke, device.type == "cuda")
    model = build_model(config).to(device)
    initial_source = None
    transfer_metadata = {}
    if resume_state is None:
        if config["experiment"] == "E10":
            initial_source, pretrained_source = load_e10_initial_weights(model, config, manifest)
            transfer_metadata.update(pretrained_weight_source=pretrained_source,
                                     initial_classifier_state_sha256=initial_source["classifier_state_sha256"])
        elif config["experiment"] in {"E11", "E11_stem"}:
            initial_source, pretrained_source = load_e11_initial_weights(model, config, manifest)
            transfer_metadata["pretrained_weight_source"] = pretrained_source
            field = "classifier_state_sha256" if config["experiment"] == "E11_stem" else "retained_state_sha256"
            transfer_metadata["initial_" + field] = initial_source[field]
        elif config["experiment"] == "E11_stem_cbam":
            initial_source, transfer_metadata = load_e11_stem_cbam_initial_weights(model, config, manifest)
        else:
            initial_source = load_initial_weights(model, config, manifest)
    optimizer = build_optimizer(model, config)
    scheduler = None
    if config["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"],
                                                              eta_min=config["min_learning_rate"])
    amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, start, best_accuracy, best_loss, best_epoch = [], 0, -1.0, float("inf"), 0
    mixup_history = []
    cutmix_history = []
    use_mixup = config["mixup_alpha"] > 0
    use_cutmix = config["cutmix_alpha"] > 0
    label_smoothing = config["label_smoothing"]
    if use_cutmix:
        accuracy_semantics, objective_kind = "cutmix_weighted", "cutmix_cross_entropy"
    elif use_mixup:
        accuracy_semantics, objective_kind = "mixup_weighted", "mixup_cross_entropy"
    else:
        accuracy_semantics = "hard_label"
        objective_kind = "label_smoothed_cross_entropy" if label_smoothing else "cross_entropy"
    initial_hash = None
    shared_metadata = {}
    peak_cuda_mib = None
    if resume_state is not None:
        # checkpoint 来自本项目自己的运行，包含优化器和随机状态，需完整载入。
        state = resume_state
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is None:
            if state["scheduler"] is not None or any(
                group["lr"] != config["learning_rate"] for group in optimizer.param_groups
            ):
                raise ValueError("固定学习率续训权重的调度器状态或优化器学习率不匹配")
        else:
            scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, start = state["history"], state["epoch"]
        mixup_history = state.get("mixup_history", [])
        cutmix_history = state.get("cutmix_history", [])
        if use_cutmix and len(cutmix_history) != start * len(loaders["train"]):
            raise ValueError("续训需要与已完成轮次一致的 CutMix batch 记录")
        best_accuracy, best_loss, best_epoch = state["best_accuracy"], state["best_loss"], state["best_epoch"]
        initial_hash = state.get("initial_state_sha256")
        initial_source = state.get("initial_checkpoint_source")
        if config["model"] == "cifar_resnet18_cbam":
            shared_metadata["initial_shared_state_sha256"] = state["initial_shared_state_sha256"]
        if config["experiment"] in {"E10", "E11_stem", "E11_stem_cbam"}:
            transfer_metadata = {key: state[key] for key in ("pretrained_weight_source", "initial_classifier_state_sha256")}
        elif config["experiment"] == "E11":
            transfer_metadata = {key: state[key] for key in ("pretrained_weight_source", "initial_retained_state_sha256")}
        peak_cuda_mib = state.get("peak_cuda_allocated_mib")
        restore_rng(state["rng"], generators)
        if history:
            write_history(run_dir / "history.csv", history)
        if mixup_history:
            write_history(run_dir / "mixup_batches.csv", mixup_history)
        if cutmix_history:
            write_history(run_dir / "cutmix_batches.csv", cutmix_history)
    else:
        # 先核验消融实验的起点，再创建输出，避免缺少参考权重时遗留半成品目录。
        run_dir.mkdir(parents=True)
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "data_manifest.json", manifest)
        info = environment_info()
        info["source_sha256"] = snapshot_sources(run_dir, args.config)
        write_json(run_dir / "environment.json", info)
        # 这是正式起点，不能用合成测试或 smoke 更新过的模型替代。
        initial_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        initial_hash = state_dict_sha256(initial_state)
        shared_init = {}
        if config["model"] == "cifar_resnet18_cbam" and initial_source:
            shared_hash = state_dict_sha256(cbam_shared_state(initial_state))
            if shared_hash != initial_source["state_sha256"]:
                raise ValueError(f"CBAM 模型的共有层未正确复用 {initial_source['experiment']} 初始状态")
            shared_metadata["initial_shared_state_sha256"] = shared_hash
            shared_init["shared_state_sha256"] = shared_hash
        save_checkpoint(run_dir / "initial.pt", {
            "format_version": 2, "epoch": 0, "model_name": config["model"], "config": config,
            "model": initial_state, "state_sha256": initial_hash,
            "initial_checkpoint_source": initial_source,
            "manifest_hash": manifest_hash(manifest), "rng": capture_rng(generators),
            **shared_init,
            **transfer_metadata,
        })
    (run_dir / "model.txt").write_text(str(model) + f"\nParameters: {sum(p.numel() for p in model.parameters())}\n", encoding="utf-8")
    print(f"输出：{run_dir}\n模式：{'SMOKE（非正式结果）' if args.smoke else '正式 ' + config['experiment']}；设备：{device}", flush=True)
    print(f"样本数：{ {k: len(v.dataset) for k, v in loaders.items()} }", flush=True)
    print("训练增强：" + ("随机裁剪 + 随机水平翻转" if config["train_augmentation"] else "关闭（仅张量转换与归一化）"), flush=True)
    if config["preprocessing"] == "imagenet224":
        print("预处理：训练 32×32 随机裁剪/翻转后双线性缩放至 224×224；监测/验证缩放至 256 后中心裁剪至 224，ImageNet 归一化。", flush=True)
    if config["optimizer"] == "adamw":
        print(f"优化器：AdamW（lr={config['learning_rate']:g}，weight_decay={config['weight_decay']:g}）。", flush=True)
    if config["model"] == "standard_resnet18":
        print(f"初始化：{'ImageNet 预训练' if config['pretrained'] else '随机初始化（pretrained=false）'}。", flush=True)
    if scheduler is None:
        print(f"学习率策略：固定学习率 {config['learning_rate']:g}，各轮保持相同。", flush=True)
    if early_stopping is not None:
        print(f"早停：前 {early_stopping['min_epochs']} 轮建立基准，之后连续 {early_stopping['patience']} 轮"
              f"未提升超过 {100 * early_stopping['min_delta']:g} 个百分点就结束。"
              f"余弦周期：{config['epochs']} 轮。", flush=True)
    if use_mixup:
        print(f"MixUp：alpha={config['mixup_alpha']}，应用概率={config['mixup_probability']}；训练准确率按混合标签加权。", flush=True)
    if use_cutmix:
        print(f"CutMix：alpha={config['cutmix_alpha']}，应用概率={config['cutmix_probability']}；标签按真实替换面积修正。", flush=True)
    if label_smoothing:
        print(f"训练目标使用标签平滑，epsilon={label_smoothing:g}。"
              "验证和干净训练监测使用普通交叉熵。", flush=True)
    if shared_metadata:
        print(f"CBAM：8 个残差块，通道压缩率={config['cbam_reduction']}，空间卷积核={config['cbam_spatial_kernel_size']}。", flush=True)
        print(f"共有层初始哈希（{initial_source['experiment']} 第 0 轮）："
              f"{shared_metadata['initial_shared_state_sha256']}", flush=True)
        print(f"{config['experiment']} 完整初始哈希（含 CBAM）：{initial_hash}", flush=True)
    elif config["experiment"] == "E11_stem":
        print(f"入口和四个残差阶段从 {transfer_metadata['pretrained_weight_source']['enum']} 初始化。"
              "首层使用 7×7 核的中心 3×3。"
              f"E4 第 0 轮分类头哈希 {transfer_metadata['initial_classifier_state_sha256']}。", flush=True)
    elif config["experiment"] == "E11":
        print(f"残差阶段 layer1–layer4 从 {transfer_metadata['pretrained_weight_source']['enum']} 初始化。"
              f"E4 第 0 轮入口和分类头哈希 {transfer_metadata['initial_retained_state_sha256']}。", flush=True)
    elif transfer_metadata:
        print(f"骨干权重：{transfer_metadata['pretrained_weight_source']['enum']}；E9 第 0 轮分类头哈希 "
              f"{transfer_metadata['initial_classifier_state_sha256']}。", flush=True)
    elif initial_source:
        print(f"初始模型：E1 第 0 轮，参数哈希 {initial_hash}", flush=True)
    end = min(config["epochs"], start + args.stop_after) if args.stop_after else config["epochs"]
    if finished:
        # 最后一轮 checkpoint 已保存时，允许续训命令补齐中断前来不及写出的汇总。
        end = start
    if device.type == "cuda" and not finished:
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, end):
        begin = time.perf_counter()
        lr_used = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{config['epochs']} lr={lr_used:.7f}", flush=True)
        training = train_one_epoch(model, loaders["train"], optimizer, scaler, device, amp, config["log_interval"],
                                   mixup_alpha=config["mixup_alpha"], mixup_probability=config["mixup_probability"],
                                   cutmix_alpha=config["cutmix_alpha"], cutmix_probability=config["cutmix_probability"],
                                   label_smoothing=label_smoothing)
        monitor = evaluate_model(model, loaders["monitor"], device)
        validation = evaluate_model(model, loaders["validation"], device)
        if not all(torch.isfinite(torch.tensor(v)) for v in [monitor["loss"], validation["loss"]]):
            raise FloatingPointError("评估损失非有限值")
        improved = validation["accuracy"] > best_accuracy or (
            validation["accuracy"] == best_accuracy and validation["loss"] < best_loss)
        if improved:
            best_accuracy, best_loss, best_epoch = validation["accuracy"], validation["loss"], epoch + 1
        row = {"epoch": epoch + 1, "lr": lr_used, "train_objective": training["loss"],
                        "train_clean_loss": monitor["loss"], "train_clean_accuracy": monitor["accuracy"],
                        "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
                        "epoch_seconds": time.perf_counter() - begin}
        accuracy_field = ("train_cutmix_weighted_accuracy" if use_cutmix else
                          ("train_mixup_weighted_accuracy" if use_mixup else "train_aug_accuracy"))
        row[accuracy_field] = training["accuracy"]
        if use_mixup:
            for key in ("mixup_batches", "mixup_total_batches", "mixup_lambda_mean", "mixup_lambda_min", "mixup_lambda_max"):
                row[key] = training[key]
            mixup_history.extend({"epoch": epoch + 1, **record} for record in training["mixup_trace"])
        if use_cutmix:
            for key in ("cutmix_batches", "cutmix_nonzero_batches", "cutmix_total_batches",
                        "cutmix_lambda_mean", "cutmix_lambda_min", "cutmix_lambda_max"):
                row[key] = training[key]
            cutmix_history.extend({"epoch": epoch + 1, **record} for record in training["cutmix_trace"])
        if early_state is not None:
            triggered = update_early_stopping(early_state, early_stopping, epoch + 1, validation["accuracy"])
            row.update(early_stop_reference_accuracy=early_state["reference_accuracy"], early_stop_wait=early_state["wait"])
            stop_reason = None
            if triggered:
                stop_reason = "early_stopping"
            elif epoch + 1 == config["epochs"]:
                stop_reason = "max_epochs"
            elif epoch + 1 == end:
                stop_reason = "stop_after"
        history.append(row)
        # 记录刚完成 epoch 所用 lr；保存的调度状态则准备好下一 epoch。
        if scheduler is not None:
            scheduler.step()
        if device.type == "cuda":
            peak_cuda_mib = max(peak_cuda_mib or 0, torch.cuda.max_memory_allocated(device) / 1024 ** 2)
        state = {"format_version": 2, "model_name": config["model"],
                 "initial_state_sha256": initial_hash,
                 "initial_checkpoint_source": initial_source,
                 **shared_metadata,
                 **transfer_metadata,
                 "train_accuracy_semantics": accuracy_semantics, "mixup_history": mixup_history,
                 "cutmix_history": cutmix_history,
                 "peak_cuda_allocated_mib": peak_cuda_mib,
                 "epoch": epoch + 1, "config": config, "manifest_hash": manifest_hash(manifest),
                 "normalization": normalization_for(manifest, config["preprocessing"]),
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict() if scheduler is not None else None,
                 "scaler": scaler.state_dict(),
                 "best_accuracy": best_accuracy, "best_loss": best_loss, "best_epoch": best_epoch,
                 "history": history, "rng": capture_rng(generators)}
        if early_state is not None:
            state.update(early_stopping_state=early_state, stop_reason=stop_reason)
        if improved:
            save_checkpoint(run_dir / "best.pt", state)
        save_checkpoint(run_dir / "last.pt", state)
        write_history(run_dir / "history.csv", history)
        if mixup_history:
            write_history(run_dir / "mixup_batches.csv", mixup_history)
        if cutmix_history:
            write_history(run_dir / "cutmix_batches.csv", cutmix_history)
        print(f"  train_clean_acc={monitor['accuracy']:.2%} val_acc={validation['accuracy']:.2%} "
              f"val_loss={validation['loss']:.4f} best_epoch={best_epoch}", flush=True)
        if early_state is not None and epoch + 1 > early_stopping["min_epochs"]:
            print(f"  早停监控：基准={early_state['reference_accuracy']:.2%} "
                  f"等待={early_state['wait']}/{early_stopping['patience']}", flush=True)
        if stop_reason == "early_stopping":
            print(f"已在第 {epoch + 1} 轮触发早停；最佳权重来自第 {best_epoch} 轮。", flush=True)
            break
    stopping_metadata = {} if early_state is None else {
        "stop_reason": stop_reason, "early_stopping_state": early_state,
    }
    write_json(run_dir / "summary.json", {"model_name": config["model"],
               "optimizer": config["optimizer"],
               "preprocessing": config["preprocessing"],
               "input_size": config["input_size"],
               "pretrained": config["pretrained"],
               "pretrained_weights": config["pretrained_weights"],
               "scheduler": config["scheduler"],
               "initial_state_sha256": initial_hash,
               "initial_checkpoint_source": initial_source,
               **shared_metadata,
               **transfer_metadata,
               **stopping_metadata,
               "train_augmentation": config["train_augmentation"],
               "mixup_alpha": config["mixup_alpha"], "mixup_probability": config["mixup_probability"],
               "cutmix_alpha": config["cutmix_alpha"], "cutmix_probability": config["cutmix_probability"],
               "label_smoothing": label_smoothing,
               "train_accuracy_semantics": accuracy_semantics,
               "train_objective_kind": objective_kind,
               "peak_cuda_allocated_mib": peak_cuda_mib,
               "smoke": args.smoke, "completed_epochs": len(history),
               "planned_epochs": config["epochs"],
               "complete": len(history) == config["epochs"] or stop_reason == "early_stopping",
               "best_epoch": best_epoch, "best_val_accuracy": best_accuracy,
               "best_val_loss": best_loss, "parameter_count": sum(p.numel() for p in model.parameters()),
               "epoch_seconds_sum_excluding_checkpoint_io": sum(row["epoch_seconds"] for row in history)})
    print("已保存。可单独运行 plot_history.py 查看曲线；训练过程没有评估测试集。", flush=True)


if __name__ == "__main__":
    main()
