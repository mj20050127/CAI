"""题三：原顺序、固定随机重排与精确子树值排序的完整成本对照。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import csv
import gzip
import importlib.util
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

from experiments.exp1_heuristic import choose_batch_loops
from experiments.exp3_minimax_alpha_beta import (
    CASE_FIELDS, COUNTERS, _check_result, _snapshot_sources, _write_csv, _write_json,
)
from experiments.game_tree_dataset import load_dataset, reference_tree
from experiments.game_tree_ordering import (
    ordering_spec, prepare_ordered_tree, recipe_sha256, run_ordered, tree_to_dict,
)
from src.game_tree import evaluate_tree


ROOT = Path(__file__).resolve().parent.parent
TIMINGS = ("copy_ms", "oracle_ms", "order_ms", "prepare_ms", "search_ms", "query_ms", "total_ms")
RAW_FIELDS = (
    "run_id", "instance_id", "tree_file", "tree_sha256", "source", "seed", "depth",
    "branch_factor", "node_count", "leaf_count", "variant", "ordering", "order_seed",
    "recipe_sha256", "batch_index", "batch_loops", "order_in_group", "schedule_row",
    "root_value", "reference_value", "correct", *COUNTERS, "oracle_node_evals",
    *(f"{timing}_mean" for timing in TIMINGS), "batch_total_ms",
)


def variants_for_count(random_count: int) -> tuple[str, ...]:
    """保留偶数规模，以使用位置和组内相邻关系均衡的 Williams 设计。"""
    if type(random_count) is not int or random_count < 2 or random_count > 10 or random_count % 2:
        raise ValueError("random_count 必须是 2 至 10 之间的偶数")
    return ("original", *(f"random_{index:02d}" for index in range(random_count)), "ideal")


def group_orders(variants: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    n = len(variants)
    if n < 2 or n % 2 or len(set(variants)) != n:
        raise ValueError("Williams 设计要求互异的偶数个变体")
    first = [0] + [(index + 1) // 2 if index % 2 else n - index // 2 for index in range(1, n)]
    return tuple(tuple(variants[(index + shift) % n] for index in first) for shift in range(n))


def _schedule_cases(cases: list[dict[str, Any]], tree_order_seed: int, batches: int, n_variants: int):
    random_cases = [dict(case) for case in cases if case["source"] == "random"]
    official = [dict(case) for case in cases if case["source"] == "official"]
    random.Random(tree_order_seed).shuffle(random_cases)
    sizes = Counter(case["depth"] for case in random_cases)
    offsets, cumulative = {}, 0
    for depth in sorted(sizes):
        offsets[str(depth)] = cumulative % n_variants
        cumulative += sizes[depth] * batches
    next_index: dict[int, int] = defaultdict(int)
    for case in random_cases:
        case["random_order_index"] = next_index[case["depth"]]
        case["order_offset"] = offsets[str(case["depth"])]
        next_index[case["depth"]] += 1
    return random_cases + official, offsets


def schedule_row(case: dict[str, Any], batch: int, batches: int, n_variants: int) -> int:
    if case["source"] == "official":
        return batch % n_variants
    return (case["order_offset"] + case["random_order_index"] * batches + batch) % n_variants


def _order_balance(cases: list[dict[str, Any]], batches: int, orders: tuple[tuple[str, ...], ...]):
    variants = orders[0]

    def empty():
        return {
            "row_counts": {str(index): 0 for index in range(len(orders))},
            "position_counts": {variant: {str(index): 0 for index in range(len(variants))} for variant in variants},
            "adjacent_pair_counts": {f"{left}->{right}": 0 for left in variants for right in variants if left != right},
        }

    random_all, official, by_depth = empty(), empty(), {}
    for case in cases:
        if case["source"] == "random":
            targets = [random_all, by_depth.setdefault(str(case["depth"]), empty())]
        else:
            targets = [official]
        for batch in range(batches):
            row = schedule_row(case, batch, batches, len(orders))
            order = orders[row]
            for target in targets:
                target["row_counts"][str(row)] += 1
                for position, variant in enumerate(order):
                    target["position_counts"][variant][str(position)] += 1
                for left, right in zip(order, order[1:]):
                    target["adjacent_pair_counts"][f"{left}->{right}"] += 1
    return random_all, by_depth, official


def _audit_variant(case: dict[str, Any], variant: str):
    prepared = prepare_ordered_tree(case["root"], variant, capture_recipe=True)
    if prepared.recipe is None:
        raise AssertionError("正确性审计需要实际排列配方")
    reference = reference_tree(tree_to_dict(prepared.root))
    if any(reference[field] != case[field] for field in ("depth", "branch_factor", "node_count", "leaf_count")):
        raise AssertionError("重排改变了树的结构规模")
    if reference["value"] != case["reference_value"]:
        raise AssertionError("重排改变了独立参考根值")
    mm = evaluate_tree(prepared.root, use_alpha_beta=False, trace=False)
    ab = evaluate_tree(prepared.root, use_alpha_beta=True, trace=False)
    _check_result(case, "minimax", mm)
    _check_result(case, "alpha_beta", ab)
    ordering, seed = ordering_spec(variant)
    if prepared.oracle_node_evals != (case["node_count"] if ordering == "ideal" else 0):
        raise AssertionError("精确排序必须每次计算完整树，其他排序不能调用该 oracle")
    identity = recipe_sha256(prepared.recipe)
    meta = {field: case[field] for field in CASE_FIELDS}
    variant_meta = {"variant": variant, "ordering": ordering, "order_seed": seed, "recipe_sha256": identity}
    proof = {
        **meta, **variant_meta, "root_value": ab.value, "correct": True,
        **{field: getattr(ab.stats, field) for field in COUNTERS},
        "minimax_value": mm.value, "minimax_node_visits": mm.stats.node_visits,
        "minimax_leaf_evals": mm.stats.leaf_evals, "oracle_node_evals": prepared.oracle_node_evals,
    }
    recipe = {**meta, **variant_meta, "permutations": prepared.recipe}
    trace = None
    if case["source"] == "official" and variant in ("original", "random_00", "ideal"):
        result = evaluate_tree(prepared.root, use_alpha_beta=True, trace=True)
        _check_result(case, "alpha_beta", result)
        if any(getattr(result.stats, field) != proof[field] for field in COUNTERS):
            raise AssertionError("官方详细轨迹与不记录轨迹的计数不一致")
        trace = asdict(result)
    return proof, recipe, trace


def _check_run(case: dict[str, Any], variant: str, run) -> None:
    _check_result(case, "alpha_beta", run.result)
    expected_evals = case["node_count"] if variant == "ideal" else 0
    if run.oracle_node_evals != expected_evals or run.result.trace:
        raise AssertionError("正式查询的预计算工作量或 trace 不符合配置")
    times = (run.copy_ms, run.oracle_ms, run.order_ms, run.prepare_ms, run.query_ms, run.result.stats.elapsed_ms)
    if any(not math.isfinite(value) or value < 0 for value in times) or min(times[-2:]) <= 0:
        raise AssertionError("阶段计时必须有限、非负；搜索和查询必须为正")
    if variant == "original" and any(times[:4]):
        raise AssertionError("原顺序不需要复制、oracle 或重排准备")
    if variant != "ideal" and run.oracle_ms:
        raise AssertionError("只有理想顺序可以计算精确子树值")


def _measure_once(case: dict[str, Any], variant: str):
    started = perf_counter_ns()
    run = run_ordered(case["root"], variant, trace=False)
    total_ms = (perf_counter_ns() - started) / 1_000_000
    _check_run(case, variant, run)
    return run, total_ms


def measure_batch(case: dict[str, Any], variant: str, loops: int, *, expected_counts=None):
    """完整成本包含每次新复制/预计算/重排；原顺序直接复用只读输入。"""
    ordering_spec(variant)
    if type(loops) is not int or loops < 1:
        raise ValueError("批次循环数必须为正整数")
    started = perf_counter_ns()
    runs = [run_ordered(case["root"], variant, trace=False) for _ in range(loops)]
    batch_total_ms = (perf_counter_ns() - started) / 1_000_000
    first = runs[0]
    counts = {field: getattr(first.result.stats, field) for field in COUNTERS}
    for run in runs:
        _check_run(case, variant, run)
        if any(getattr(run.result.stats, field) != counts[field] for field in COUNTERS):
            raise AssertionError("同树同排列的确定性计数不稳定")
    if expected_counts is not None and any(counts[field] != expected_counts[field] for field in COUNTERS):
        raise AssertionError("正式查询与计时外排列审计的计数不一致")
    result = {
        "root_value": first.result.value, "reference_value": case["reference_value"], "correct": True,
        **counts, "oracle_node_evals": first.oracle_node_evals,
        **{f"{timing}_mean": sum(getattr(run, timing) for run in runs) / loops
           for timing in ("copy_ms", "oracle_ms", "order_ms", "prepare_ms", "query_ms")},
        "search_ms_mean": sum(run.result.stats.elapsed_ms for run in runs) / loops,
        "total_ms_mean": batch_total_ms / loops, "batch_total_ms": batch_total_ms,
    }
    return result


def run_experiment(
    output_dir: Path, *, dataset_dir: Path | None = None, random_count: int = 10,
    batches: int = 5, warmup: int = 2, target_batch_ms: float = 25,
    max_batch_ms: float = 200, max_loops: int = 128, analyze: bool = True,
) -> dict[str, Any]:
    variants = variants_for_count(random_count)
    orders = group_orders(variants)
    if any(type(value) is not int or value < 1 for value in (batches, warmup, max_loops)):
        raise ValueError("batches、warmup、max_loops 必须为正整数")
    if type(analyze) is not bool:
        raise ValueError("analyze 必须为布尔值")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in (target_batch_ms, max_batch_ms)):
        raise ValueError("计时目标和上限必须为有限正数")
    if analyze and importlib.util.find_spec("matplotlib") is None:
        raise ValueError("生成图表需要 matplotlib；可加 --no-analysis")
    output_dir = Path(output_dir).resolve()
    dataset_dir = Path(dataset_dir or ROOT / "results/problem3_minimax_alpha_beta/dataset").resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("结果目录非空，请指定新的输出目录")
    if output_dir == dataset_dir or output_dir.is_relative_to(dataset_dir) or dataset_dir.is_relative_to(output_dir):
        raise ValueError("输入数据集与输出目录不能互相包含")
    manifest, source_cases = load_dataset(dataset_dir)
    for case in source_cases:
        relative = Path(case["tree_file"])
        if relative.is_absolute() or not (output_dir / "dataset" / relative).resolve().is_relative_to(output_dir / "dataset"):
            raise ValueError("输入副本必须使用数据目录内的相对文件路径")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_dataset = output_dir / "dataset"
    saved_dataset.mkdir()
    (saved_dataset / "manifest.json").write_bytes((dataset_dir / "manifest.json").read_bytes())
    for case in source_cases:
        destination = saved_dataset / case["tree_file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((dataset_dir / case["tree_file"]).read_bytes())
    del source_cases
    manifest, cases = load_dataset(saved_dataset)
    tree_order_seed = manifest["config"]["master_seed"] ^ 0xC433
    cases, depth_offsets = _schedule_cases(cases, tree_order_seed, batches, len(variants))
    random_balance, depth_balance, official_balance = _order_balance(cases, batches, orders)
    source_hashes, source_archive_hash = _snapshot_sources(output_dir)
    run_id = "p3_order_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    config = {
        "experiment": "problem3_visit_order_v1", "run_id": run_id, "status": "running",
        "analysis_status": "pending" if analyze else "skipped", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset_dir), "dataset_dir": "dataset", "dataset_config": manifest["config"],
        "dataset_manifest_sha256": sha256((saved_dataset / "manifest.json").read_bytes()).hexdigest(),
        "random_instances": sum(case["source"] == "random" for case in cases),
        "official_instances": sum(case["source"] == "official" for case in cases), "measured_instances": len(cases),
        "variants": variants, "random_count": random_count, "batches": batches,
        "warmup_per_variant": warmup, "calibration_calls_per_variant": 2,
        "target_fast_batch_ms": target_batch_ms, "max_estimated_slow_batch_ms": max_batch_ms, "max_batch_loops": max_loops,
        "calibration_rule": "same tree, all variants same loops; min(ceil(target/fastest total), floor(max_batch/slowest total), max_loops), at least 1",
        "tree_order_seed": tree_order_seed, "measurement_order": [case["instance_id"] for case in cases],
        "random_order_indices": {case["instance_id"]: case["random_order_index"] for case in cases if case["source"] == "random"},
        "depth_order_offsets": depth_offsets, "group_orders": orders,
        "group_order_rule": "random: (depth_offset+within_depth_index*batches+batch_index)%n_variants; official: batch_index%n_variants",
        "random_order_counts": random_balance["row_counts"],
        "random_order_counts_by_depth": {depth: balance["row_counts"] for depth, balance in depth_balance.items()},
        "random_position_counts": random_balance["position_counts"],
        "random_position_counts_by_depth": {depth: balance["position_counts"] for depth, balance in depth_balance.items()},
        "random_adjacent_pair_counts": random_balance["adjacent_pair_counts"],
        "random_adjacent_pair_counts_by_depth": {depth: balance["adjacent_pair_counts"] for depth, balance in depth_balance.items()},
        "official_order_counts": official_balance["row_counts"],
        "adjacency_balance_scope": "within each complete variant group; excludes boundaries across groups and trees; incomplete Williams blocks only approximately balanced",
        "search_time_scope": "original evaluate_tree internal elapsed_ms, counters included, trace off; excludes full tree validation and ordering preparation",
        "query_time_scope": "evaluate_tree on the prepared tree: full input validation plus search and result return; excludes permutation preparation",
        "prepare_time_scope": "fresh tree copy (random/ideal), exact all-subtree computation (ideal only), shuffle/sort and setup; original needs no preparation",
        "total_time_scope": "fresh complete single-query cost including preparation, query, wrapper, object release, batch loop and result collection; excludes input loading/conversion, independent reference, proof, recipe hashing, trace, I/O and plotting",
        "preparation_components": "copy_ms, oracle_ms and order_ms are separately timed parts of prepare_ms; timing/setup overhead may remain",
        "random_order_definition": "Random(order_seed) for every fresh call; all node permutations fixed before Alpha-Beta; seeds 0..random_count-1",
        "ideal_order_definition": "full exact subtree values recomputed for every fresh call; stable child-value descending at MAX, ascending at MIN",
        "ideal_order_caveat": "oracle information is an ideal reference, not a free practical heuristic; all oracle/copy/sort costs charged to total",
        "recipe_definition": "per internal node in original left-to-right preorder, list original child indices in visit order; compressed actual recipes saved",
        "statistics_unit": "one original tree; batch medians per variant, then 10 fixed random-order ratios summarized within tree before depth-wise aggregation",
        "trace": False, "cache": False, "cutoff_rule": "alpha >= beta with unvisited siblings", "garbage_collection": "normal interpreter behavior",
        "source_sha256": source_hashes, "source_baseline_sha256": source_archive_hash,
        "environment": {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                        "processor": platform.processor(), "logical_cpus": os.cpu_count()},
    }
    _write_json(output_dir / "config.json", config)
    n_rows = total_timed_calls = 0
    try:
        proofs, expected, traces = [], {}, {}
        with gzip.open(output_dir / "recipes.jsonl.gz", "wt", encoding="utf-8", newline="\n") as stream:
            for index, case in enumerate(cases):
                for variant in variants:
                    proof, recipe, trace = _audit_variant(case, variant)
                    proofs.append(proof)
                    expected[case["instance_id"], variant] = proof
                    stream.write(json.dumps(recipe, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if trace is not None:
                        traces[variant] = trace
                if (index + 1) % 10 == 0 or index + 1 == len(cases):
                    print(f"Audited {index+1}/{len(cases)} trees and actual permutations", flush=True)
        _write_csv(output_dir / "proof.csv", proofs)
        _write_json(output_dir / "official_trace.json", traces)
        config["recipes_sha256"] = sha256((output_dir / "recipes.jsonl.gz").read_bytes()).hexdigest()
        config["proof_rows"] = len(proofs)
        _write_json(output_dir / "config.json", config)
        calibrations = []
        print(f"Prepared {config['random_instances']} random trees + {config['official_instances']} official; variants={len(variants)}; batches={batches}", flush=True)
        with (output_dir / "raw.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
            writer.writeheader()
            for index, case in enumerate(cases):
                estimates = {}
                for variant in orders[index % len(orders)]:
                    for _ in range(warmup):
                        _measure_once(case, variant)
                    samples = [_measure_once(case, variant) for _ in range(2)]
                    estimates[variant] = median(total_ms for _, total_ms in samples)
                loops = choose_batch_loops(estimates, target_batch_ms, max_batch_ms, max_loops)
                calibrations.append({"instance_id": case["instance_id"], "source": case["source"], "depth": case["depth"],
                                     **{f"{variant}_total_ms": estimates[variant] for variant in variants}, "batch_loops": loops})
                for batch in range(batches):
                    row = schedule_row(case, batch, batches, len(variants))
                    for position, variant in enumerate(orders[row]):
                        proof = expected[case["instance_id"], variant]
                        measured = measure_batch(case, variant, loops, expected_counts=proof)
                        writer.writerow({"run_id": run_id, **{field: case[field] for field in CASE_FIELDS},
                                         **{field: proof[field] for field in ("variant", "ordering", "order_seed", "recipe_sha256")},
                                         "batch_index": batch, "batch_loops": loops, "order_in_group": position,
                                         "schedule_row": row, **measured})
                        n_rows += 1
                        total_timed_calls += loops
                stream.flush()
                if (index + 1) % 5 == 0 or index + 1 == len(cases):
                    print(f"Measured {index+1}/{len(cases)} trees; rows={n_rows}; last={case['instance_id']}", flush=True)
        _write_csv(output_dir / "calibration.csv", calibrations)
        config.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(), raw_rows=n_rows,
                      total_timed_calls=total_timed_calls, all_results_correct=True)
        _write_json(output_dir / "config.json", config)
    except Exception as error:
        config.update(status="failed", failure=f"{type(error).__name__}: {error}", raw_rows=n_rows, total_timed_calls=total_timed_calls)
        _write_json(output_dir / "config.json", config)
        raise
    if analyze:
        from experiments.analyze_visit_order import analyze_results
        analyze_results(output_dir)
        config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    print(f"Completed. Results: {output_dir}", flush=True)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="题三：访问顺序与预计算完整成本对照")
    parser.add_argument("--out", type=Path, default=ROOT / "results/problem3_visit_order")
    parser.add_argument("--dataset", type=Path, default=ROOT / "results/problem3_minimax_alpha_beta/dataset")
    parser.add_argument("--random-count", type=int, default=10)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--target-batch-ms", type=float, default=25)
    parser.add_argument("--max-batch-ms", type=float, default=200)
    parser.add_argument("--max-loops", type=int, default=128)
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_experiment(args.out, dataset_dir=args.dataset, random_count=args.random_count, batches=args.batches,
                       warmup=args.warmup, target_batch_ms=args.target_batch_ms, max_batch_ms=args.max_batch_ms,
                       max_loops=args.max_loops, analyze=not args.no_analysis)
    except (ValueError, OSError) as error:
        parser.exit(2, f"访问顺序实验配置或文件错误：{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
