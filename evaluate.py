"""独立评估最佳权重。默认 validation；测试须显式指定 --split test。"""
import argparse
import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.common import choose_device, read_json, seed_everything, write_json
from src.data import cifar10, load_manifest, manifest_hash, transforms_for
from src.engine import evaluate_model
from src.metrics import classification_metrics
from src.models import build_model, model_name_from_checkpoint
from src.plots import plot_confusion, plot_errors


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-dir", type=Path, help="移动项目或在另一台机器复现时，指定当前数据目录")
    parser.add_argument("--output-dir", type=Path, help="可选独立评估输出目录，兼容检查时避免覆盖原结果")
    args = parser.parse_args()
    device = choose_device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = model_name_from_checkpoint(state)
    config = state["config"].copy()
    if args.data_dir is not None:
        config["data_dir"] = str(args.data_dir.resolve())
    seed_everything(config["seed"])
    if config["smoke"] and args.split == "test":
        raise ValueError("smoke 权重用于流程检查；请用正式训练权重进行测试集评估。")
    output = args.output_dir.resolve() if args.output_dir else args.checkpoint.resolve().parent / f"evaluation_{args.split}_epoch{state['epoch']}"
    if args.output_dir and output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("显式评估输出目录非空，请换一个新目录以保留原结果。")
    manifest = load_manifest(config)
    if manifest_hash(manifest) != state["manifest_hash"]:
        raise ValueError("当前数据清单与训练时不同")
    preprocessing = config.get("preprocessing", "cifar32")
    input_size = config.get("input_size", 32)
    if args.split == "test":
        raw = cifar10(config["data_dir"], train=False, transform=transforms_for(manifest, False, preprocessing))
        indices = list(range(len(raw)))
        if len(raw) != 10000:
            raise ValueError("预期官方测试集共 10,000 张")
    else:
        raw = cifar10(config["data_dir"], transform=transforms_for(manifest, False, preprocessing))
        indices = manifest["indices"]["validation"]
    if raw.classes != manifest["classes"]:
        raise ValueError("类别顺序与训练时不同")
    loader = DataLoader(Subset(raw, indices), batch_size=config["batch_size"], shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    model = build_model(config).to(device)
    model.load_state_dict(state["model"])
    result = evaluate_model(model, loader, device, predictions=True)
    if not math.isfinite(result["loss"]):
        raise FloatingPointError("评估 loss 非有限值")
    metrics = classification_metrics(result["targets"], result["predictions"], manifest["classes"])
    metrics.update(loss=result["loss"], epoch=state["epoch"], split=args.split, model_name=model_name,
                   preprocessing=preprocessing, input_size=input_size,
                   manifest_hash=state["manifest_hash"],
                   smoke=config["smoke"], checkpoint=str(args.checkpoint.resolve()))
    # 同一权重重复评估只更新该评估目录；训练日志与权重独立保留。
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", metrics)
    records = [{"index": index, "true_label": true, "predicted_label": pred,
                "true_class": manifest["classes"][true], "predicted_class": manifest["classes"][pred],
                "confidence": confidence}
               for index, true, pred, confidence in zip(indices, result["targets"], result["predictions"], result["confidences"])]
    assert len(records) == len(indices)
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == len(indices)
    write_csv(output / "predictions.csv", records)
    write_csv(output / "per_class.csv", metrics["per_class"])
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + manifest["classes"])
        for name, row in zip(manifest["classes"], metrics["confusion_matrix"]):
            writer.writerow([name] + row)
    plot_confusion(metrics["confusion_matrix"], manifest["classes"], output / "confusion_counts.png")
    plot_confusion(metrics["confusion_matrix"], manifest["classes"], output / "confusion_normalized.png", True)
    plot_errors(raw.data, records, manifest["classes"], output / "errors.png")
    print(f"{args.split}: n={len(indices)} accuracy={metrics['overall_accuracy']:.2%} loss={result['loss']:.4f}")
    for row in metrics["per_class"]:
        print(f"  {row['name']}: {row['accuracy']:.2%} ({row['correct']}/{row['support']})")
    print("结果：", output)


if __name__ == "__main__":
    main()
