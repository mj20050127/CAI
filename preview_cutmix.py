"""生成训练图片的基础增强与 CutMix 示例；仅 CPU，不训练或访问测试集。"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from src.common import ROOT, load_config, write_json
from src.data import cifar10, load_manifest, manifest_hash, transforms_for


def _display_image(tensor, manifest):
    """反归一化仅用于显示；不改变参与区域替换的张量。"""
    mean = torch.as_tensor(manifest["mean"], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.as_tensor(manifest["std"], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _cutmix_display_pair(image_a, image_b, alpha, rng):
    """预选 A/B 配对，保留未筛选的 Beta 抽样及与训练一致的整数框规则。"""
    height, width = image_a.shape[-2:]
    lam = float(rng.beta(alpha, alpha))
    ratio = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(np.floor(width * ratio)), int(np.floor(height * ratio))
    cx, cy = int(rng.integers(width)), int(rng.integers(height))
    left, top = cx - cut_w // 2, cy - cut_h // 2
    right, bottom = left + cut_w, top + cut_h
    x1, x2 = int(np.clip(left, 0, width)), int(np.clip(right, 0, width))
    y1, y2 = int(np.clip(top, 0, height)), int(np.clip(bottom, 0, height))
    area = (x2 - x1) * (y2 - y1)
    mixed = image_a.clone()
    mixed[:, y1:y2, x1:x2] = image_b[:, y1:y2, x1:x2]
    return mixed, {"lambda_raw": lam, "lambda_eff": 1.0 - area / (width * height),
                   "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                   "actual_area": area, "image_width": width, "image_height": height,
                   "center_x": cx, "center_y": cy, "cut_width_raw": cut_w, "cut_height_raw": cut_h}


def generate_preview(config, output_dir, seed=42):
    """固定四对不同类训练图片；局部替换比例按裁剪后的真实像素面积计算。"""
    alpha = config["cutmix_alpha"]
    if alpha <= 0 or not config["train_augmentation"] or config.get("mixup_alpha", 0) != 0:
        raise ValueError("预览需要开启基础增强和 CutMix、关闭 MixUp，请使用 E4 配置")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("预览 seed 必须是 [0, 2**32) 内的整数")
    output = Path(output_dir).resolve()
    protected = [(ROOT / "runs").resolve(), (ROOT / "data").resolve(), Path(config["data_dir"]).resolve()]
    if any(output == path or output.is_relative_to(path) for path in protected):
        raise ValueError("预览输出必须独立于 runs 和数据目录，建议使用 results/E4_cutmix_preview")
    manifest = load_manifest(config)
    raw = cifar10(config["data_dir"], train=True, download=False)
    names = manifest["classes"]
    if raw.classes != names or len(names) < 8:
        raise ValueError("预览要求匹配类别顺序，且至少有八个类别")
    train_indices = np.asarray(manifest["indices"]["train"], dtype=np.int64)
    targets = np.asarray(raw.targets, dtype=np.int64)
    selection_rng = np.random.default_rng(seed)
    rectangle_rng = np.random.default_rng(seed)
    selected_classes = selection_rng.permutation(len(names))[:8]
    selected = []
    for label in selected_classes:
        candidates = train_indices[targets[train_indices] == label]
        if not len(candidates):
            raise ValueError(f"训练部分缺少类别：{names[label]}")
        selected.append(int(selection_rng.choice(candidates)))
    transform = transforms_for(manifest, True)
    rows, displays = [], []
    # 只固定并恢复 CPU Torch 随机状态，不初始化 CUDA。
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for position in range(0, 8, 2):
            index_a, index_b = selected[position:position + 2]
            original_a, label_a = raw[index_a]
            original_b, label_b = raw[index_b]
            augmented_a, augmented_b = transform(original_a), transform(original_b)
            mixed, rectangle = _cutmix_display_pair(augmented_a, augmented_b, alpha, rectangle_rng)
            weight_a, weight_b = rectangle["lambda_eff"], 1.0 - rectangle["lambda_eff"]
            soft_label = [0.0] * len(names)
            soft_label[label_a], soft_label[label_b] = weight_a, weight_b
            rows.append({"pair": len(rows) + 1, "index_a": index_a, "index_b": index_b,
                         "label_a": int(label_a), "class_a": names[label_a],
                         "label_b": int(label_b), "class_b": names[label_b],
                         **rectangle, "weight_a": weight_a, "weight_b": weight_b,
                         "soft_label": soft_label, "applied": True})
            displays.append([np.asarray(original_a), _display_image(augmented_a, manifest),
                             np.asarray(original_b), _display_image(augmented_b, manifest),
                             _display_image(mixed, manifest)])
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / "cutmix_examples.png"
    fig, axes = plt.subplots(4, 5, figsize=(15, 13))
    headers = ["Original A", "Basic augmentation A", "Original B", "Basic augmentation B", "CutMix"]
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
                          f"raw lambda={record['lambda_raw']:.4f}; effective={record['lambda_eff']:.4f}\n"
                          f"patch={record['actual_area']}/{record['image_width'] * record['image_height']} pixels")
            ax.set_title(headers[column] + "\n" + detail, fontsize=8.5)
            if column in {1, 3, 4} and record["actual_area"]:
                # imshow 像素中心在整数坐标，边界位于半整数；标记不改变图片张量。
                ax.add_patch(Rectangle((record["x1"] - .5, record["y1"] - .5),
                                       record["x2"] - record["x1"], record["y2"] - record["y1"],
                                       linewidth=1.5, edgecolor="#ec3434", facecolor="none"))
    fig.suptitle(f"CutMix on CIFAR-10 training images | alpha={alpha:g} | preview seed={seed}", fontsize=16)
    fig.text(.5, .014,
             "Red outlines mark the replacement rectangle. Four seeded display pairs; training uses batch permutations.\n"
             "Unfiltered Beta draws and clipped rectangles; labels use actual area. No training or test evaluation.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95), h_pad=2.1, w_pad=.6)
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    report = {"experiment": config["experiment"], "preview_seed": seed, "training_seed": config["seed"],
              "cutmix_alpha": alpha, "training_cutmix_probability": config["cutmix_probability"],
              "mixup_alpha": config.get("mixup_alpha", 0),
              "source_split": "train", "manifest_sha256": manifest_hash(manifest), "classes": names,
              "basic_transform": repr(transform),
              "pairing": "four fixed pairs from eight distinct classes, selected only from manifest train indices",
              "pairing_note": "预览固定不同类 A/B 配对方便展示；正式训练使用 batch 内随机排列，可能同类或自身配对。",
              "rectangle_format": "x1,y1,x2,y2; half-open [y1:y2,x1:x2]; x is width, y is height",
              "rectangle_policy": "floor(size*sqrt(1-lambda_raw)); left=center-size//2; right=left+size; independently clip both edges",
              "lambda_sampling": "independent Beta(alpha,alpha) draws; no filtering, folding or resampling of rectangles",
              "label_weight": "lambda_eff=1-actual_area/(width*height); not the raw Beta lambda",
              "preview_rng": "independent local NumPy generators for selection and rectangle; seeded/restored CPU Torch transforms",
              "preview_training_rng_note": "固定配对及独立预览 RNG 与正式训练不同；这些系数不代表正式训练的前四批，也不改变正式训练状态。",
              "display": "inverse-normalized transformed images, clipped to [0,1] only for display; red outlines do not alter tensors",
              "training_started": False, "test_set_accessed": False, "gpu_used": False,
              "pairs": rows, "image": str(image_path)}
    write_json(output / "preview.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e4.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/E4_cutmix_preview")
    parser.add_argument("--seed", type=int, default=42, help="仅控制预览，不改变训练配置")
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed 必须是 [0, 2**32) 内的整数")
    report = generate_preview(load_config(args.config), args.output_dir, args.seed)
    print("CutMix 示例：", report["image"])
    print("预览记录：", args.output_dir.resolve() / "preview.json")
    print("仅使用训练图片与 CPU，没有启动训练或访问测试集。")


if __name__ == "__main__":
    main()
