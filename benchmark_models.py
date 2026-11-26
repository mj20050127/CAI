"""按检查点输入尺寸使用同设备上的 FP32 合成输入进行推理计时。"""
import argparse
import statistics
import time
from pathlib import Path

import torch

from src.common import ROOT, choose_device, seed_everything, write_json
from src.models import build_model, model_name_from_checkpoint


INPUT_SIZES = {"cifar32": 32, "imagenet224": 224}


def input_protocol(config):
    preprocessing = config.get("preprocessing", "cifar32")
    input_size = config.get("input_size", 32)
    if not isinstance(preprocessing, str) or preprocessing not in INPUT_SIZES:
        raise ValueError("检查点的 preprocessing 必须是 cifar32 或 imagenet224")
    if type(input_size) is not int or input_size != INPUT_SIZES[preprocessing]:
        raise ValueError(f"检查点的 input_size 与 preprocessing={preprocessing} 不匹配")
    return preprocessing, input_size


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure_models(models, batch_sizes, device, warmup=10, iterations=50, repeats=3, *, input_size=32):
    if any(type(v) is not int or v < 1 for v in [warmup, iterations, repeats, *batch_sizes]):
        raise ValueError("batch、预热、迭代和重复次数必须为正整数")
    if type(input_size) is not int or input_size not in INPUT_SIZES.values():
        raise ValueError("推理计时的输入边长必须是 32 或 224")
    results = []
    for batch in batch_sizes:
        # 相同 batch 下，两模型看到同一个合成输入；预处理与传输不计入耗时。
        generator = torch.Generator().manual_seed(2026 + batch)
        inputs = torch.randn(batch, 3, input_size, input_size, generator=generator).to(device)
        times = {name: [] for name, _ in models}
        if len(times) != len(models):
            raise ValueError("模型计时名称必须唯一")
        for _, model in models:
            model.eval()
            for _ in range(warmup):
                model(inputs)
        synchronize(device)
        for repeat in range(repeats):
            # 交替测量次序，减少总是先测同一个模型的次序偏差。
            for name, model in (models if repeat % 2 == 0 else list(reversed(models))):
                synchronize(device)
                start = time.perf_counter()
                for _ in range(iterations):
                    model(inputs)
                synchronize(device)
                times[name].append((time.perf_counter() - start) * 1000 / iterations)
        for name, model in models:
            latency = statistics.median(times[name])
            results.append({"name": name, "batch_size": batch,
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "batch_latency_ms_median": latency,
                            "amortized_ms_per_image": latency / batch,
                            "images_per_second": batch * 1000 / latency,
                            "repeat_batch_latency_ms": times[name]})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs=2, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 128])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path,
                        help="32×32 默认 results/E0_vs_E1/inference_benchmark.json，224×224 按实验对单独保存")
    args = parser.parse_args()
    seed_everything(42)
    device = choose_device(args.device)
    models, provenance, experiments = [], [], []
    manifest, protocol = None, None
    for index, path in enumerate(args.checkpoints):
        state = torch.load(path, map_location="cpu", weights_only=False)
        name = model_name_from_checkpoint(state)
        config = state["config"]
        if config.get("smoke", False):
            raise ValueError("请使用正式训练权重进行计时比较")
        current_protocol = input_protocol(config)
        if protocol is not None and current_protocol != protocol:
            raise ValueError("两个模型的输入尺寸或预处理不同，不能按同批合成输入比较")
        protocol = current_protocol
        if manifest is not None and state["manifest_hash"] != manifest:
            raise ValueError("两个模型的数据清单不同")
        manifest = state["manifest_hash"]
        model = build_model(config).to(device).eval()
        model.load_state_dict(state["model"])
        experiments.append(config["experiment"])
        label = f"{index + 1}_{config['experiment']}_{name}"
        models.append((label, model))
        provenance.append({"name": label, "checkpoint": str(path.resolve()), "epoch": state["epoch"]})
        del state
    preprocessing, input_size = protocol
    output = args.output
    if output is None:
        pair = "E0_vs_E1"
        if input_size == 224:
            if any(not isinstance(name, str) or not name
                   or not all(char.isalnum() or char in "_-" for char in name) for name in experiments):
                parser.error("默认计时目录需要有效的实验名称，请检查 experiment 或指定 --output")
            pair = "E9_vs_E10" if set(experiments) == {"E9", "E10"} else "_vs_".join(experiments)
        output = ROOT / "results" / pair / "inference_benchmark.json"
    output = output.resolve()
    if output.suffix.lower() != ".json" or any(output.is_relative_to(p.resolve().parent) for p in args.checkpoints):
        parser.error("计时输出应为输入运行目录之外的 JSON 文件")
    results = measure_models(models, args.batch_sizes, device, args.warmup, args.iterations, args.repeats,
                             input_size=input_size)
    write_json(output, {"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                        "torch": torch.__version__, "precision": "float32",
                        "input_shape": f"N,3,{input_size},{input_size}", "preprocessing": preprocessing,
                        "includes_preprocessing_or_host_transfer": False,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "warmup": args.warmup, "iterations": args.iterations, "repeats": args.repeats,
                        "manifest_hash": manifest, "models": provenance, "results": results,
                        "note": "合成输入的模型推理计时，不代表端到端数据加载耗时；摊销时间不等于 batch=1 延迟。"})
    for result in results:
        print(f"{result['name']} batch={result['batch_size']} "
              f"latency={result['batch_latency_ms_median']:.3f} ms "
              f"throughput={result['images_per_second']:.1f} images/s")
    print(output)


if __name__ == "__main__":
    main()
