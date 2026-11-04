"""题一对照：同一批地图上的零启发与曼哈顿，保留原始计时与输入。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from math import ceil
import os
from pathlib import Path
import platform
import random
from statistics import median
import sys
from time import perf_counter_ns
from typing import Any

from experiments.maze_dataset import bfs_distance, create_dataset, validate_path
from src.main import read_case
from src.problem1_maze import search_maze


ROOT = Path(__file__).resolve().parent.parent
HEURISTICS = ("zero", "manhattan")
COUNTERS = (
    "valid_pops", "expansions", "generated_successors", "pushes", "stale_pops", "heap_peak",
)
RAW_FIELDS = (
    "run_id", "instance_id", "map_file", "map_sha256", "source", "size", "density", "seed",
    "heuristic", "batch_index", "batch_loops", "order_in_pair", "path_cost", "bfs_cost", "correct",
    *COUNTERS, "search_ms_mean", "api_ms_mean", "batch_api_ms",
)


def input_hash(case: dict[str, Any]) -> str:
    payload = {key: case[key] for key in ("start", "end", "grid")}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _check_result(case: dict[str, Any], result: Any) -> None:
    if result.cost != case["bfs_cost"]:
        raise AssertionError(f"{case['instance_id']}：A* 代价与独立 BFS 不一致")
    validate_path(
        case["grid"], tuple(case["start"]), tuple(case["end"]), result.path, case["bfs_cost"]
    )


def _measure_once(case: dict[str, Any], heuristic: str):
    started = perf_counter_ns()
    result = search_maze(case["start"], case["end"], case["grid"], heuristic=heuristic, trace=False)
    api_ms = (perf_counter_ns() - started) / 1_000_000
    _check_result(case, result)  # 校验在计时区间之外。
    return result, api_ms


def choose_batch_loops(
    estimates: dict[str, float], target_batch_ms: float, max_batch_ms: float, max_loops: int
) -> int:
    """同图两算法用相同循环数；较慢配置的批次时长设上限，避免失控。"""
    fastest = max(min(estimates.values()), 0.000001)
    slowest = max(max(estimates.values()), 0.000001)
    desired = max(1, ceil(target_batch_ms / fastest))
    duration_limit = max(1, int(max_batch_ms / slowest))
    return min(desired, duration_limit, max_loops)


def measure_batch(case: dict[str, Any], heuristic: str, loops: int):
    """计时中不做校验或 I/O；批次结束后逐一检查全部输出。

    API 时间包含函数调用、循环和结果收集；搜索时间取各次内部计时的均值。
    保留每次结果便于在计时区间外校验，heap_peak 仍只是单次算法的堆长度。
    """
    started = perf_counter_ns()
    results = [
        search_maze(case["start"], case["end"], case["grid"], heuristic=heuristic, trace=False)
        for _ in range(loops)
    ]
    batch_api_ms = (perf_counter_ns() - started) / 1_000_000
    first = results[0]
    expected_counts = tuple(getattr(first.stats, field) for field in COUNTERS)
    for result in results:
        _check_result(case, result)
        if tuple(getattr(result.stats, field) for field in COUNTERS) != expected_counts:
            raise AssertionError("同一输入和启发函数的节点计数不稳定")
    return {
        "path_cost": first.cost,
        "correct": True,
        **{field: getattr(first.stats, field) for field in COUNTERS},
        "search_ms_mean": sum(result.stats.elapsed_ms for result in results) / loops,
        "api_ms_mean": batch_api_ms / loops,
        "batch_api_ms": batch_api_ms,
    }


def _load_cases(dataset_dir: Path, output_dir: Path):
    manifest = read_case(dataset_dir / "manifest.json")
    cases = []
    for relative in manifest["accepted_files"]:
        path = (dataset_dir / relative).resolve()
        if not path.is_relative_to(dataset_dir.resolve()):
            raise ValueError("manifest 中的地图路径越过数据集目录")
        case = read_case(path)
        if case["sha256"] != input_hash(case):
            raise ValueError(f"地图哈希不匹配：{path}")
        expected = bfs_distance(case["grid"], tuple(case["start"]), tuple(case["end"]))
        if expected is None or expected != case["bfs_cost"] or not case["accepted"]:
            raise ValueError(f"数据集可达性或参考代价错误：{path}")
        case.update(source="random", map_file=os.path.relpath(path, output_dir))
        cases.append(case)

    official = read_case(ROOT / "cases" / "problem1.json")
    official.update(
        instance_id="official_5x5", size=5, seed="official", source="official", density="",
        map_file="official_map.json", accepted=True,
    )
    official["bfs_cost"] = bfs_distance(official["grid"], tuple(official["start"]), tuple(official["end"]))
    official["sha256"] = input_hash(official)
    _write_json(output_dir / "official_map.json", official)
    cases.append(official)
    return manifest, cases


def run_experiment(
    output_dir: Path,
    *,
    sizes: tuple[int, ...] = (21, 51, 101),
    count_per_size: int = 30,
    density: float = 0.2,
    master_seed: int = 20261021,
    batches: int = 5,
    warmup: int = 2,
    target_batch_ms: float = 25.0,
    max_batch_ms: float = 200.0,
    max_loops: int = 128,
    dataset_dir: Path | None = None,
    analyze: bool = True,
) -> dict[str, Any]:
    """在新的结果目录运行；重跑时可复用已保存数据集但不覆盖旧结果。"""
    if batches < 1 or warmup < 1 or max_loops < 1:
        raise ValueError("batches、warmup、max_loops 必须至少为 1")
    if target_batch_ms <= 0 or max_batch_ms <= 0:
        raise ValueError("批次目标时长必须为正")
    if analyze and importlib.util.find_spec("matplotlib") is None:
        raise ValueError("绘图需要 matplotlib；可先安装，或用 --no-analysis 只采集数据")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("结果目录非空；请指定新的 --out，旧数据不会被覆盖")
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset_dir is None:
        dataset_dir = output_dir / "dataset"
        create_dataset(
            dataset_dir, sizes=sizes, count_per_size=count_per_size,
            density=density, master_seed=master_seed,
        )
    else:
        dataset_dir = dataset_dir.resolve()
    manifest, cases = _load_cases(dataset_dir, output_dir)
    if len({case["instance_id"] for case in cases}) != len(cases):
        raise ValueError("实例编号重复")

    run_id = "p1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    source_files = (
        "src/astar.py", "src/grid.py", "src/heuristics.py", "src/problem1_maze.py", "src/stats.py",
        "experiments/maze_dataset.py", "experiments/exp1_heuristic.py",
    )
    config = {
        "run_id": run_id,
        "status": "running",
        "analysis_status": "pending" if analyze else "skipped",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": os.path.relpath(dataset_dir, output_dir),
        "dataset_manifest_sha256": sha256((dataset_dir / "manifest.json").read_bytes()).hexdigest(),
        "dataset_config": manifest["config"],
        "random_instances": len(cases) - 1,
        "official_instances": 1,
        "batches": batches,
        "warmup_per_heuristic": warmup,
        "calibration_calls_per_heuristic": 2,
        "target_fast_batch_ms": target_batch_ms,
        "max_estimated_slow_batch_ms": max_batch_ms,
        "max_batch_loops": max_loops,
        "calibration_rule": "min(ceil(target/faster), floor(max_batch/slower), max_loops), at least 1",
        "loop_count_policy": "same loops for both heuristics on the same map",
        "map_order_seed": master_seed ^ 0x51A7,
        "pair_order": "alternate by (shuffled map index + batch index) parity; record each order",
        "neighbor_order": "up,right,down,left",
        "tie_break": "(f,h,insertion_counter)",
        "trace": False,
        "garbage_collection": "normal interpreter behavior; not explicitly disabled",
        "search_time_scope": "A* internal timer, includes counters and path reconstruction, excludes grid preparation",
        "api_time_scope": "search_maze calls including grid preparation, loop and result collection; excludes validation and I/O",
        "validation": "independent BFS once per input; every timed path checked outside timing",
        "statistics_unit": "one saved map; use median across batches before aggregating across maps",
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "source_sha256": {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files},
    }
    random.Random(config["map_order_seed"]).shuffle(cases)
    config["measurement_order"] = [case["instance_id"] for case in cases]
    _write_json(output_dir / "config.json", config)

    all_paths = []
    calibration_rows = []
    counts_by_size: dict[int, int] = {}
    records_written = 0
    print(f"Prepared {len(cases) - 1} random maps + 1 official map; batches={batches}", flush=True)
    for group in manifest["groups"]:
        print(f"  size={group['size']}: accepted={group['accepted']}, rejected={group['rejected']}", flush=True)
    try:
        with (output_dir / "raw.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
            writer.writeheader()
            for index, case in enumerate(cases):
                pair = HEURISTICS if index % 2 == 0 else tuple(reversed(HEURISTICS))
                estimates = {}
                for heuristic in pair:
                    for _ in range(warmup):
                        _measure_once(case, heuristic)
                    samples = [_measure_once(case, heuristic) for _ in range(2)]
                    estimates[heuristic] = median(value for _, value in samples)
                    all_paths.append({
                        "instance_id": case["instance_id"], "heuristic": heuristic,
                        "map_sha256": case["sha256"], "cost": samples[0][0].cost,
                        "path": samples[0][0].path,
                    })
                loops = choose_batch_loops(estimates, target_batch_ms, max_batch_ms, max_loops)
                calibration_rows.append({
                    "instance_id": case["instance_id"],
                    "zero_api_ms": estimates["zero"], "manhattan_api_ms": estimates["manhattan"],
                    "batch_loops": loops,
                })
                for batch_index in range(batches):
                    pair = HEURISTICS if (index + batch_index) % 2 == 0 else tuple(reversed(HEURISTICS))
                    for order, heuristic in enumerate(pair):
                        measured = measure_batch(case, heuristic, loops)
                        writer.writerow({
                            "run_id": run_id, "instance_id": case["instance_id"],
                            "map_file": case["map_file"], "map_sha256": case["sha256"],
                            "source": case["source"], "size": case["size"],
                            "density": manifest["config"]["density"] if case["source"] == "random" else "",
                            "seed": case["seed"], "heuristic": heuristic,
                            "batch_index": batch_index, "batch_loops": loops,
                            "order_in_pair": order, "bfs_cost": case["bfs_cost"], **measured,
                        })
                        records_written += 1
                stream.flush()
                counts_by_size[case["size"]] = counts_by_size.get(case["size"], 0) + 1
                if (index + 1) % 5 == 0 or index + 1 == len(cases):
                    print(f"Measured {index + 1}/{len(cases)} maps; rows={records_written}; last={case['instance_id']}", flush=True)
        with (output_dir / "calibration.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("instance_id", "zero_api_ms", "manhattan_api_ms", "batch_loops"))
            writer.writeheader()
            writer.writerows(calibration_rows)
        _write_json(output_dir / "paths.json", all_paths)
        config.update(
            status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
            raw_rows=records_written, measured_instances=len(cases),
            total_timed_calls=sum(row["batch_loops"] for row in calibration_rows) * batches * 2,
            all_results_correct=True,
        )
        _write_json(output_dir / "config.json", config)
    except Exception as error:
        config.update(status="failed", failure=f"{type(error).__name__}: {error}")
        _write_json(output_dir / "config.json", config)
        raise

    if analyze:
        from experiments.analyze_maze_heuristics import analyze_results
        analyze_results(output_dir)
        config["analysis_status"] = "completed"
        config["analysis_source_sha256"] = sha256(
            (ROOT / "experiments" / "analyze_maze_heuristics.py").read_bytes()
        ).hexdigest()
        _write_json(output_dir / "config.json", config)
    print(f"Completed. Results: {output_dir}", flush=True)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="题一：零启发 vs 曼哈顿的可复现实验")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "problem1_heuristics")
    parser.add_argument("--dataset", type=Path, help="复用已有数据集；必须有 manifest.json")
    parser.add_argument("--sizes", type=int, nargs="+", default=[21, 51, 101])
    parser.add_argument("--instances", type=int, default=30, help="每种规模的可达地图数量")
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--target-batch-ms", type=float, default=25.0)
    parser.add_argument("--max-batch-ms", type=float, default=200.0)
    parser.add_argument("--max-loops", type=int, default=128)
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_experiment(
            args.out, sizes=tuple(args.sizes), count_per_size=args.instances,
            density=args.density, master_seed=args.seed, batches=args.batches,
            warmup=args.warmup, target_batch_ms=args.target_batch_ms,
            max_batch_ms=args.max_batch_ms, max_loops=args.max_loops,
            dataset_dir=args.dataset, analyze=not args.no_analysis,
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"实验配置或文件错误：{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
