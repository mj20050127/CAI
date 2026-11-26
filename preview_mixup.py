"""生成训练图片的基础增强与 MixUp 示例；仅 CPU，不训练或访问测试集。"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common import ROOT, load_config, write_json
from src.data import cifar10, load_manifest, manifest_hash, transforms_for


def _display_image(tensor, manifest):
    """反归一化只用于显示，裁剪显示范围不改变训练张量。"""
    mean = torch.as_tensor(manifest["mean"], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.as_tensor(manifest["std"], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def generate_preview(config, output_dir, seed=42):
    """固定四对不同类别训练图片；lambda 不筛选、不折叠，记录真实抽样值。"""
    alpha = config["mixup_alpha"]
    if alpha <= 0 or not config["train_augmentation"]:
        raise ValueError("预览需要启用基础增强并设置 mixup_alpha>0，请使用 E3 配置")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("预览 seed 必须是 [0, 2**32) 内的整数")
    output = Path(output_dir).resolve()
    protected = [(ROOT / "runs").resolve(), (ROOT / "data").resolve(), Path(config["data_dir"]).resolve()]
    if any(output == path or output.is_relative_to(path) for path in protected):
        raise ValueError("预览输出必须独立于 runs 和数据目录，建议使用 results/E3_mixup_preview")
    manifest = load_manifest(config)
    raw = cifar10(config["data_dir"], train=True, download=False)
    names = manifest["classes"]
    if raw.classes != names or len(names) < 8:
        raise ValueError("预览要求匹配类别顺序，且至少有八个类别")
    train_indices = np.asarray(manifest["indices"]["train"], dtype=np.int64)
    targets = np.asarray(raw.targets, dtype=np.int64)
    selection_rng = np.random.default_rng(seed)
    lambda_rng = np.random.default_rng(seed)
    selected_classes = selection_rng.permutation(len(names))[:8]
    selected = []
    for label in selected_classes:
        candidates = train_indices[targets[train_indices] == label]
        if not len(candidates):
            raise ValueError(f"训练部分缺少类别：{names[label]}")
        selected.append(int(selection_rng.choice(candidates)))
    transform = transforms_for(manifest, True)
    rows, displays = [], []
    # 仅固定并恢复 CPU Torch 随机状态；不初始化 CUDA，不使用 GPU。
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for position in range(0, 8, 2):
            index_a, index_b = selected[position:position + 2]
            original_a, label_a = raw[index_a]
            original_b, label_b = raw[index_b]
            augmented_a, augmented_b = transform(original_a), transform(original_b)
            lam = float(lambda_rng.beta(alpha, alpha))
            mixed = lam * augmented_a + (1.0 - lam) * augmented_b
            soft_label = [0.0] * len(names)
            soft_label[label_a], soft_label[label_b] = lam, 1.0 - lam
            rows.append({"pair": len(rows) + 1, "index_a": index_a, "index_b": index_b,
                         "label_a": int(label_a), "class_a": names[label_a],
                         "label_b": int(label_b), "class_b": names[label_b],
                         "lambda": lam, "weight_a": lam, "weight_b": 1.0 - lam,
                         "soft_label": soft_label})
            displays.append([np.asarray(original_a), _display_image(augmented_a, manifest),
                             np.asarray(original_b), _display_image(augmented_b, manifest),
                             _display_image(mixed, manifest)])
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / "mixup_examples.png"
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    headers = ["Original A", "Basic augmentation A", "Original B", "Basic augmentation B", "MixUp"]
    for axes_row, panels, record in zip(axes, displays, rows):
        for column, (ax, pixels) in enumerate(zip(axes_row, panels)):
            ax.imshow(pixels, interpolation="nearest")
            ax.axis("off")
            if column < 4:
                role = "a" if column < 2 else "b"
                detail = f"{record[f'class_{role}']} / train #{record[f'index_{role}']}"
            else:
                detail = (f"{record['class_a']} {record['weight_a']:.3%} + "
                          f"{record['class_b']} {record['weight_b']:.3%}\n"
                          f"lambda = {record['lambda']:.8f}")
            ax.set_title(headers[column] + "\n" + detail, fontsize=9)
    fig.suptitle(f"MixUp on CIFAR-10 training images | alpha={alpha:g} | preview seed={seed}", fontsize=16)
    fig.text(.5, .015,
             "Four seeded display pairs. Training pairs random samples within each batch. "
             "Lambda values are unfiltered Beta draws; no training or test evaluation.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95), h_pad=1.8, w_pad=.6)
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    report = {"experiment": config["experiment"], "preview_seed": seed,
              "training_seed": config["seed"], "mixup_alpha": alpha,
              "training_mixup_probability": config["mixup_probability"],
              "source_split": "train", "manifest_sha256": manifest_hash(manifest),
              "classes": names, "basic_transform": repr(transform),
              "pairing": "four fixed pairs from eight distinct classes, selected only from manifest train indices",
              "pairing_note": "预览使用固定 A/B 配对方便展示；训练使用每个 batch 内的随机排列，可能出现同类或自身配对。",
              "lambda_sampling": "four independent Beta(alpha, alpha) draws; no filtering or folding",
              "preview_rng": "independent local NumPy generators for pair selection and lambda; seeded CPU Torch transforms",
              "display": "original images and inverse-normalized transformed/mixed images; clipped to [0,1] for display",
              "training_started": False, "test_set_accessed": False, "gpu_used": False,
              "pairs": rows, "image": str(image_path)}
    write_json(output / "preview.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e3.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/E3_mixup_preview")
    parser.add_argument("--seed", type=int, default=42, help="仅控制预览，不改变训练配置")
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed 必须是 [0, 2**32) 内的整数")
    report = generate_preview(load_config(args.config), args.output_dir, args.seed)
    print("MixUp 示例：", report["image"])
    print("预览记录：", args.output_dir.resolve() / "preview.json")
    print("仅使用训练图片与 CPU，没有启动训练或访问测试集。")


if __name__ == "__main__":
    main()
