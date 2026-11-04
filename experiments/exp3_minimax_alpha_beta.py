"""题三基础对照：固定原顺序的 Minimax 与 Alpha-Beta 配对测量。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import csv
import json
import math
import os
from pathlib import Path
import platform
import random
from statistics import median
import sys
from time import perf_counter_ns
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.exp1_heuristic import choose_batch_loops
from experiments.game_tree_dataset import create_dataset, load_dataset
from src.game_tree import evaluate_tree


ROOT = Path(__file__).resolve().parent.parent
ALGORITHMS = ("minimax", "alpha_beta")
ORDERS = (ALGORITHMS, tuple(reversed(ALGORITHMS)))
COUNTERS = ("node_visits", "leaf_evals", "cutoff_events", "pruned_branches")
RAW_FIELDS = (
    "run_id", "instance_id", "tree_file", "tree_sha256", "source", "seed", "depth",
    "branch_factor", "node_count", "leaf_count", "algorithm", "batch_index", "batch_loops",
    "order_in_pair", "root_value", "reference_value", "correct", *COUNTERS,
    "search_ms_mean", "api_ms_mean", "batch_api_ms",
)
CASE_FIELDS = (
    "instance_id", "tree_file", "tree_sha256", "source", "seed", "depth", "branch_factor",
    "node_count", "leaf_count", "reference_value",
)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("不能写入空记录表")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _check_result(case: dict[str, Any], algorithm: str, result: Any) -> None:
    if algorithm not in ALGORITHMS:
        raise ValueError("未知的博弈搜索算法")
    if result.value != case["reference_value"]:
        raise AssertionError(f"{case['instance_id']}：根值与独立自底向上参考不一致")
    stats = result.stats
    if any(type(getattr(stats, field)) is not int or getattr(stats, field) < 0 for field in COUNTERS):
        raise AssertionError("搜索计数必须为非负整数")
    if not (1 <= stats.leaf_evals <= case["leaf_count"] and stats.leaf_evals <= stats.node_visits <= case["node_count"]):
        raise AssertionError("叶子评估或节点访问次数超过完整树范围")
    if stats.pruned_branches < stats.cutoff_events:
        raise AssertionError("一次有效截断必须至少跳过一个直接兄弟")
    if (stats.pruned_branches == 0) != (stats.cutoff_events == 0):
        raise AssertionError("有效截断与跳过直接兄弟数不一致")
    if algorithm == "minimax" and (
        stats.node_visits != case["node_count"] or stats.leaf_evals != case["leaf_count"]
        or stats.cutoff_events or stats.pruned_branches
    ):
        raise AssertionError("Minimax 应访问整棵树且不剪枝")


def _measure_once(case: dict[str, Any], algorithm: str):
    if algorithm not in ALGORITHMS:
        raise ValueError("未知的博弈搜索算法")
    started = perf_counter_ns()
    result = evaluate_tree(case["root"], use_alpha_beta=algorithm == "alpha_beta", trace=False)
    api_ms = (perf_counter_ns() - started) / 1_000_000
    _check_result(case, algorithm, result)
    return result, api_ms


def measure_batch(case: dict[str, Any], algorithm: str, loops: int) -> dict[str, Any]:
    """同一已构建树反复查询；输出核验全部位于计时区间之外。

    每次 evaluate_tree 都会完整校验输入并新建统计、递归和轨迹列表。
    API 时间包含校验、搜索、循环和结果收集；搜索时间来自原接口内部计时。
    """
    if algorithm not in ALGORITHMS:
        raise ValueError("未知的博弈搜索算法")
    if type(loops) is not int or loops < 1:
        raise ValueError("批次循环数必须为正整数")
    started = perf_counter_ns()
    results = [
        evaluate_tree(case["root"], use_alpha_beta=algorithm == "alpha_beta", trace=False)
        for _ in range(loops)
    ]
    batch_api_ms = (perf_counter_ns() - started) / 1_000_000
    first = results[0]
    counts = tuple(getattr(first.stats, field) for field in COUNTERS)
    for result in results:
        _check_result(case, algorithm, result)
        if tuple(getattr(result.stats, field) for field in COUNTERS) != counts:
            raise AssertionError("同树同算法的确定性计数不稳定")
        if result.trace:
            raise AssertionError("正式计时不能记录详细轨迹")
    return {
        "root_value": first.value, "reference_value": case["reference_value"], "correct": True,
        **{field: getattr(first.stats, field) for field in COUNTERS},
        "search_ms_mean": sum(result.stats.elapsed_ms for result in results) / loops,
        "api_ms_mean": batch_api_ms / loops, "batch_api_ms": batch_api_ms,
    }


def _schedule_cases(cases: list[dict[str, Any]], tree_order_seed: int):
    """打乱随机树；每个深度分别编号以平衡算法先后。官方另列在最后。"""
    random_cases = [dict(case) for case in cases if case["source"] == "random"]
    official = [dict(case) for case in cases if case["source"] == "official"]
    random.Random(tree_order_seed).shuffle(random_cases)
    next_index: dict[int, int] = defaultdict(int)
    for case in random_cases:
        case["random_order_index"] = next_index[case["depth"]]
        next_index[case["depth"]] += 1
    return random_cases + official


def _order_balance(cases: list[dict[str, Any]], batches: int):
    counts: dict[str, dict[str, int]] = {}
    official_counts = {",".join(order): 0 for order in ORDERS}
    for case in cases:
        if case["source"] == "random":
            current = counts.setdefault(str(case["depth"]), {",".join(order): 0 for order in ORDERS})
        else:
            current = official_counts
        for batch in range(batches):
            schedule = case["random_order_index"] * batches + batch if case["source"] == "random" else batch
            current[",".join(ORDERS[schedule % 2])] += 1
    return counts, official_counts


def _snapshot_sources(output_dir: Path):
    paths = sorted({
        path.relative_to(ROOT).as_posix()
        for pattern in ("src/*.py", "experiments/*.py", "tests/*.py", "cases/*.json", "requirements-experiments.txt")
        for path in ROOT.glob(pattern) if path.is_file()
    })
    hashes = {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    archive_path = output_dir / "source_baseline.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        for name in paths:
            archive.write(ROOT / name, name)
    return hashes, sha256(archive_path.read_bytes()).hexdigest()


def run_experiment(
    output_dir: Path, *, dataset_dir: Path | None = None, depths=(4, 6, 8),
    count_per_depth: int = 30, master_seed: int = 20261022, batches: int = 5,
    warmup: int = 2, target_batch_ms: float = 25, max_batch_ms: float = 200,
    max_loops: int = 128, analyze: bool = True,
) -> dict[str, Any]:
    if any(type(value) is not int or value < 1 for value in (count_per_depth, batches, warmup, max_loops)):
        raise ValueError("count_per_depth、batches、warmup、max_loops 必须为正整数")
    if type(master_seed) is not int:
        raise ValueError("master_seed 必须为整数")
    if not isinstance(depths, (list, tuple)) or not depths or any(type(value) is not int or value < 0 for value in depths):
        raise ValueError("depths 必须为非空非负整数列表")
    if len(set(depths)) != len(depths):
        raise ValueError("depths 不能重复")
    if type(analyze) is not bool:
        raise ValueError("analyze 必须为布尔值")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in (target_batch_ms, max_batch_ms)):
        raise ValueError("计时目标与上限必须为有限正数")
    if analyze and importlib.util.find_spec("matplotlib") is None:
        raise ValueError("生成图表需要 matplotlib；可加 --no-analysis")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("结果目录非空或不是目录，请指定新的输出目录")
    if dataset_dir is not None:
        dataset_dir = Path(dataset_dir).resolve()
        if output_dir == dataset_dir or output_dir.is_relative_to(dataset_dir) or dataset_dir.is_relative_to(output_dir):
            raise ValueError("输入数据集与输出目录不能互相包含")
        manifest, cases = load_dataset(dataset_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_dataset = output_dir / "dataset"
        saved_dataset.mkdir()
        (saved_dataset / "manifest.json").write_bytes((dataset_dir / "manifest.json").read_bytes())
        for case in cases:
            destination = saved_dataset / case["tree_file"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((dataset_dir / case["tree_file"]).read_bytes())
        # 按副本重新加载，保证实际保存的输入就是本轮参与测量的输入。
        manifest, cases = load_dataset(saved_dataset)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_dataset = output_dir / "dataset"
        create_dataset(saved_dataset, depths=tuple(depths), count_per_depth=count_per_depth, master_seed=master_seed)
        manifest, cases = load_dataset(saved_dataset)
    data_seed = manifest["config"]["master_seed"]
    tree_order_seed = data_seed ^ 0x33A9
    cases = _schedule_cases(cases, tree_order_seed)
    order_counts, official_counts = _order_balance(cases, batches)
    source_hashes, archive_hash = _snapshot_sources(output_dir)
    run_id = "p3_baseline_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    config = {
        "experiment": "problem3_minimax_alpha_beta_v1", "run_id": run_id, "status": "running",
        "analysis_status": "pending" if analyze else "skipped",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset_dir) if dataset_dir is not None else None,
        "dataset_dir": "dataset", "dataset_config": manifest["config"],
        "dataset_manifest_sha256": sha256((saved_dataset / "manifest.json").read_bytes()).hexdigest(),
        "random_instances": sum(case["source"] == "random" for case in cases),
        "official_instances": sum(case["source"] == "official" for case in cases),
        "measured_instances": len(cases), "algorithms": ALGORITHMS, "batches": batches,
        "warmup_per_algorithm": warmup, "calibration_calls_per_algorithm": 2,
        "target_fast_batch_ms": target_batch_ms, "max_estimated_slow_batch_ms": max_batch_ms,
        "max_batch_loops": max_loops,
        "calibration_rule": "same tree, same loops; min(ceil(target/fastest), floor(max_batch/slowest), max_loops), at least 1",
        "tree_order_seed": tree_order_seed,
        "measurement_order": [case["instance_id"] for case in cases],
        "random_order_indices": {case["instance_id"]: case["random_order_index"] for case in cases if case["source"] == "random"},
        "random_order_counts_by_depth": order_counts, "official_order_counts": official_counts,
        "pair_orders": ORDERS,
        "pair_order_rule": "random: (within-depth random_order_index*batches+batch_index)%2; official: batch_index%2",
        "tree_depth_unit": "number of edges from root (depth 0) to a deepest leaf",
        "child_order": "saved left-to-right order, unchanged between algorithms",
        "root_role": "MAX, alternating MAX/MIN at every level", "cutoff_rule": "alpha >= beta with at least one unvisited sibling",
        "trace": False, "cache": False, "sorting": False,
        "oracle": "independent bottom-up value and structure computation on input dictionaries, outside timing",
        "search_time_scope": "original evaluate_tree internal elapsed_ms: recursive search, counters and root return; excludes full input validation",
        "api_time_scope": "each original evaluate_tree call including full tree validation and search, plus batch loop/result collection; excludes file loading, construction, oracle, result checks, detailed trace and plotting",
        "non_search_time_note": "API minus internal search includes input validation, setup, return/collection and timer overhead; it is not an isolated validation measurement",
        "statistics_unit": "one tree; median across batches first, then paired aggregation within each depth",
        "counter_scope": {"node_visits": "recursive calls including root, internal nodes and leaves",
                          "leaf_evals": "visited terminal leaves",
                          "cutoff_events": "cutoffs that actually skip unvisited siblings",
                          "pruned_branches": "skipped direct siblings, not all skipped subtree nodes"},
        "garbage_collection": "normal interpreter behavior",
        "source_sha256": source_hashes, "source_baseline_sha256": archive_hash,
        "environment": {"python": sys.version, "python_executable": sys.executable,
                        "platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count()},
    }
    _write_json(output_dir / "config.json", config)
    n_rows = 0
    total_timed_calls = 0
    try:
        audits, expected_counts, official_trace = [], {}, {}
        for case in cases:
            for algorithm in ALGORITHMS:
                detailed = evaluate_tree(case["root"], use_alpha_beta=algorithm == "alpha_beta", trace=case["source"] == "official")
                _check_result(case, algorithm, detailed)
                counts = {field: getattr(detailed.stats, field) for field in COUNTERS}
                expected_counts[case["instance_id"], algorithm] = counts
                audits.append({**{field: case[field] for field in CASE_FIELDS}, "algorithm": algorithm,
                               "root_value": detailed.value, "correct": True, **counts})
                if case["source"] == "official":
                    official_trace[algorithm] = asdict(detailed)
        _write_csv(output_dir / "correctness.csv", audits)
        _write_json(output_dir / "official_trace.json", official_trace)
        calibrations = []
        print(f"Prepared {config['random_instances']} random trees + {config['official_instances']} official; two algorithms; batches={batches}", flush=True)
        with (output_dir / "raw.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
            writer.writeheader()
            for index, case in enumerate(cases):
                estimates = {}
                for algorithm in ORDERS[index % 2]:
                    for _ in range(warmup):
                        _measure_once(case, algorithm)
                    samples = [_measure_once(case, algorithm) for _ in range(2)]
                    estimates[algorithm] = median(api_ms for _, api_ms in samples)
                loops = choose_batch_loops(estimates, target_batch_ms, max_batch_ms, max_loops)
                calibrations.append({"instance_id": case["instance_id"], "source": case["source"], "depth": case["depth"],
                                     **{f"{algorithm}_api_ms": estimates[algorithm] for algorithm in ALGORITHMS}, "batch_loops": loops})
                for batch in range(batches):
                    schedule = case["random_order_index"] * batches + batch if case["source"] == "random" else batch
                    for position, algorithm in enumerate(ORDERS[schedule % 2]):
                        measured = measure_batch(case, algorithm, loops)
                        if any(measured[field] != expected_counts[case["instance_id"], algorithm][field] for field in COUNTERS):
                            raise AssertionError("计时结果与计时外正确性审计的计数不一致")
                        writer.writerow({"run_id": run_id, **{field: case[field] for field in CASE_FIELDS},
                                         "algorithm": algorithm, "batch_index": batch, "batch_loops": loops,
                                         "order_in_pair": position, **measured})
                        n_rows += 1
                        total_timed_calls += loops
                stream.flush()
                if (index + 1) % 5 == 0 or index + 1 == len(cases):
                    print(f"Measured {index+1}/{len(cases)} trees; rows={n_rows}; last={case['instance_id']}", flush=True)
        _write_csv(output_dir / "calibration.csv", calibrations)
        config.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      raw_rows=n_rows, total_timed_calls=total_timed_calls, all_results_correct=True)
        _write_json(output_dir / "config.json", config)
    except Exception as error:
        config.update(status="failed", failure=f"{type(error).__name__}: {error}", raw_rows=n_rows,
                      total_timed_calls=total_timed_calls)
        _write_json(output_dir / "config.json", config)
        raise
    if analyze:
        from experiments.analyze_game_tree import analyze_results
        analyze_results(output_dir)
        config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    print(f"Completed. Results: {output_dir}", flush=True)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="题三：原顺序 Minimax 与 Alpha-Beta 的基础配对对照")
    parser.add_argument("--out", type=Path, default=ROOT / "results/problem3_minimax_alpha_beta")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--depths", type=int, nargs="+", default=(4, 6, 8))
    parser.add_argument("--count-per-depth", type=int, default=30)
    parser.add_argument("--master-seed", type=int, default=20261022)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--target-batch-ms", type=float, default=25)
    parser.add_argument("--max-batch-ms", type=float, default=200)
    parser.add_argument("--max-loops", type=int, default=128)
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_experiment(args.out, dataset_dir=args.dataset, depths=args.depths, count_per_depth=args.count_per_depth,
                       master_seed=args.master_seed, batches=args.batches, warmup=args.warmup,
                       target_batch_ms=args.target_batch_ms, max_batch_ms=args.max_batch_ms,
                       max_loops=args.max_loops, analyze=not args.no_analysis)
    except (ValueError, OSError) as error:
        parser.exit(2, f"实验配置或文件错误：{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
