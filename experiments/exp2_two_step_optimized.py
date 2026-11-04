"""题二：两步前瞻替换快速内层的四策略完整成本对照。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
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
from experiments.exp2_lookahead import (
    COUNTERS, RAW_FIELDS as LEGACY_RAW_FIELDS, _check_result, _load_cases, _write_csv, _write_json,
)
from experiments.lookahead_heuristics import prepare_terrain, search_lookahead
from experiments.optimized_lookahead import make_optimized_h1, search_variant as search_h1_variant
from experiments.optimized_two_step import HEURISTICS, make_optimized_h2, search_variant
from experiments.weighted_dataset import reference_to_goal
from src.main import read_case
from src.problem2_terrain import search_weighted


ROOT = Path(__file__).resolve().parent.parent
ORDER_INDICES = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
ORDERS = tuple(tuple(HEURISTICS[index] for index in row) for row in ORDER_INDICES)
RAW_FIELDS = tuple("order_in_quartet" if field == "order_in_triplet" else field for field in LEGACY_RAW_FIELDS)


def _order_balance(cases: list[dict[str, Any]], batches: int):
    """记录随机图实际顺序；每完整四组平衡位置与有向相邻关系。"""
    counts = {",".join(order): 0 for order in ORDERS}
    positions = {name: {str(position): 0 for position in range(4)} for name in HEURISTICS}
    adjacent = {f"{left}->{right}": 0 for left in HEURISTICS for right in HEURISTICS if left != right}
    for case in cases:
        if case["source"] != "random":
            continue
        for batch in range(batches):
            schedule = case["random_order_index"] * batches + batch
            order = ORDERS[schedule % len(ORDERS)]
            counts[",".join(order)] += 1
            for position, name in enumerate(order):
                positions[name][str(position)] += 1
            for left, right in zip(order, order[1:]):
                adjacent[f"{left}->{right}"] += 1
    return counts, positions, adjacent


def _same_search(left, right) -> bool:
    return (
        (left.cost, left.path, left.expanded_order) == (right.cost, right.path, right.expanded_order)
        and all(getattr(left.stats, name) == getattr(right.stats, name) for name in COUNTERS)
    )


def _audit_and_verify(case: dict[str, Any], exact: dict[tuple[int, int], int]):
    """计时外检查 H0<=快速H1<=旧H2=快速H2 与全部搜索行为。"""
    terrain = prepare_terrain(case["start"], case["end"], case["grid"])
    positions = list(terrain.positions())
    factories = (terrain.heuristic(0), make_optimized_h1(terrain), terrain.heuristic(2), make_optimized_h2(terrain))
    tables = {name: {point: estimate(point) for point in positions}
              for name, estimate in zip(HEURISTICS, factories)}
    base, one_step, original, optimized = (tables[name] for name in HEURISTICS)
    mismatches = sum(original[point] != optimized[point] for point in positions)
    edges = [(point, other, weight) for point in positions for other, weight in terrain.neighbors(point)]
    audits = []
    for name in HEURISTICS:
        values = tables[name]
        over = [point for point in exact if values[point] > exact[point]]
        row = {
            "instance_id": case["instance_id"], "source": case["source"], "size": case["size"],
            "heuristic": name, "n_states": len(positions), "n_reachable_states": len(exact),
            "n_directed_edges": len(edges), "admissibility_violations": len(over),
            "consistency_violations": sum(values[u] > weight + values[v] for u, v, weight in edges),
            "max_overestimate": max((values[point] - exact[point] for point in over), default=0),
            "goal_h": values[terrain.end],
            "dominance_violations": sum(values[point] < base[point] for point in positions),
            "strictly_stronger_states": sum(values[point] > base[point] for point in positions),
            "same_formula_violations": mismatches if name == "lookahead_2_optimized" else 0,
            "h2_over_h1_violations": sum(values[point] < one_step[point] for point in positions)
                if name in ("lookahead_2", "lookahead_2_optimized") else 0,
        }
        if any(row[field] for field in (
            "admissibility_violations", "consistency_violations", "goal_h", "dominance_violations",
            "same_formula_violations", "h2_over_h1_violations",
        )):
            raise AssertionError(f"启发性质或等值检查失败：{case['instance_id']} {name}")
        audits.append(row)
    results = {
        name: search_variant(case["start"], case["end"], case["grid"], heuristic=name, trace=True).result
        for name in HEURISTICS
    }
    baseline = search_weighted(case["start"], case["end"], case["grid"], trace=True)
    legacy = search_lookahead(case["start"], case["end"], case["grid"], depth=2, trace=True).result
    fast_one = search_h1_variant(case["start"], case["end"], case["grid"],
                                heuristic="lookahead_1_optimized", trace=True).result
    old, fast = results["lookahead_2"], results["lookahead_2_optimized"]
    equivalence = {
        "instance_id": case["instance_id"], "source": case["source"], "size": case["size"],
        "map_sha256": case["sha256"], "n_states": len(positions),
        "heuristic_value_mismatches": mismatches, "cost_equal": old.cost == fast.cost,
        "path_equal": old.path == fast.path, "expanded_order_equal": old.expanded_order == fast.expanded_order,
        "counters_equal": all(getattr(old.stats, name) == getattr(fast.stats, name) for name in COUNTERS),
        "baseline_equal": _same_search(results["corrected_manhattan"], baseline),
        "old_implementation_equal": _same_search(old, legacy),
        "fast_h1_equal": _same_search(results["lookahead_1_optimized"], fast_one),
    }
    if mismatches or not all(equivalence[field] for field in (
        "cost_equal", "path_equal", "expanded_order_equal", "counters_equal", "baseline_equal",
        "old_implementation_equal", "fast_h1_equal",
    )):
        raise AssertionError(f"实现优化改变了搜索行为：{case['instance_id']}")
    for result in results.values():
        _check_result(case, result)
    return audits, equivalence


def _measure_once(case: dict[str, Any], heuristic: str):
    started = perf_counter_ns()
    measured = search_variant(case["start"], case["end"], case["grid"], heuristic=heuristic)
    total_ms = (perf_counter_ns() - started) / 1_000_000
    _check_result(case, measured.result)
    return measured, total_ms


def measure_batch(case: dict[str, Any], heuristic: str, loops: int):
    """新查询新缓存；路径与代价均在计时结束后检查。"""
    if type(loops) is not int or loops < 1:
        raise ValueError("批次循环数必须为正整数")
    started = perf_counter_ns()
    runs = [search_variant(case["start"], case["end"], case["grid"], heuristic=heuristic)
            for _ in range(loops)]
    total_ms = (perf_counter_ns() - started) / 1_000_000
    first = runs[0].result
    counts = tuple(getattr(first.stats, field) for field in COUNTERS)
    for run in runs:
        _check_result(case, run.result)
        if tuple(getattr(run.result.stats, field) for field in COUNTERS) != counts:
            raise AssertionError("同图同策略的确定性计数不稳定")
    return {
        "path_cost": first.cost, "reference_cost": case["best_cost"],
        "optimal_gap": first.cost - case["best_cost"], "correct": True,
        **{field: getattr(first.stats, field) for field in COUNTERS},
        "prepare_ms_mean": sum(run.prepare_ms for run in runs) / loops,
        "search_ms_mean": sum(run.result.stats.elapsed_ms for run in runs) / loops,
        "api_ms_mean": total_ms / loops, "batch_api_ms": total_ms,
    }


def run_experiment(
    output_dir: Path, *, dataset_dir: Path | None = None, batches: int = 5,
    warmup: int = 2, target_batch_ms: float = 25, max_batch_ms: float = 200,
    max_loops: int = 128, analyze: bool = True,
) -> dict[str, Any]:
    if any(type(value) is not int or value < 1 for value in (batches, warmup, max_loops)):
        raise ValueError("batches、warmup 和 max_loops 必须为正整数")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in (target_batch_ms, max_batch_ms)):
        raise ValueError("计时目标和上限必须为有限正数")
    if analyze and importlib.util.find_spec("matplotlib") is None:
        raise ValueError("生成图表需要 matplotlib；可加 --no-analysis")
    output_dir = Path(output_dir).resolve()
    dataset_dir = Path(dataset_dir or ROOT / "results/problem2_lookahead/dataset").resolve()
    if output_dir == dataset_dir or output_dir.is_relative_to(dataset_dir):
        raise ValueError("结果目录不能位于输入数据目录内")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("结果目录非空，请指定新的输出目录")
    if not (dataset_dir / "manifest.json").is_file():
        raise ValueError("数据集缺少 manifest.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, cases = _load_cases(dataset_dir, output_dir)
    audits, equivalences = [], []
    for case in cases:
        exact = reference_to_goal(case["grid"], case["end"])
        if exact.get(tuple(case["start"])) != case["best_cost"]:
            raise ValueError(f"{case['instance_id']} 的参考代价不一致")
        if case["best_cost"] is None:
            raise ValueError("正式性能数据集要求起终点可达；不可达情况由正确性测试覆盖")
        entries, equivalent = _audit_and_verify(case, exact)
        audits.extend(entries)
        equivalences.append(equivalent)
    _write_csv(output_dir / "heuristic_audit.csv", audits)
    _write_csv(output_dir / "equivalence.csv", equivalences)

    source_files = sorted({path.relative_to(ROOT).as_posix()
                           for pattern in ("src/*.py", "experiments/*.py", "tests/*.py", "cases/*.json", "requirements-experiments.txt")
                           for path in ROOT.glob(pattern) if path.is_file()})
    source_hashes = {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}
    with ZipFile(output_dir / "source_baseline.zip", "w", ZIP_DEFLATED) as archive:
        for name in source_files:
            archive.write(ROOT / name, name)
    run_id = "p2_h2_optimized_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    master_seed = manifest["config"]["master_seed"]
    config = {
        "experiment": "problem2_two_step_optimized_v1", "run_id": run_id,
        "status": "running", "analysis_status": "pending" if analyze else "skipped",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset_dir), "dataset_dir": "dataset", "dataset_config": manifest["config"],
        "dataset_manifest_sha256": sha256((output_dir / "dataset/manifest.json").read_bytes()).hexdigest(),
        "random_instances": len(cases) - 1, "official_instances": 1, "measured_instances": len(cases),
        "heuristics": HEURISTICS, "batches": batches, "warmup_per_heuristic": warmup,
        "calibration_calls_per_heuristic": 2, "target_fast_batch_ms": target_batch_ms,
        "max_estimated_slow_batch_ms": max_batch_ms, "max_batch_loops": max_loops,
        "calibration_rule": "same loops; min(ceil(target/fastest), floor(max_batch/slowest), max_loops), at least 1",
        "quartet_orders": ORDERS,
        "quartet_order_rule": "four-row position/adjacency-balanced design; random: (saved random_order_index*batches+batch_index)%4; official: batch_index%4; incomplete final blocks are near-balanced only",
        "adjacency_balance_scope": "within each quartet; excludes boundaries between quartets and maps",
        "map_order_seed": master_seed ^ 0x109A, "neighbor_order": "up,right,down,left", "tie_break": "(f,h,insertion_counter)",
        "trace": False, "goal_entering_cost": 0, "start_initial_cost": 0,
        "heuristic_implementation": "same h2 outer Bellman layer and direction order; only its inner h1 uses the existing optimized implementation; fresh per-query layered caches",
        "heuristic_formula": "h0=m*max(0,d-1); hk+1(n)=min_v(c(n,v)+hk(v)); old h2 and optimized h2 are pointwise identical",
        "oracle": "independent reverse Dijkstra used only for validation and audit, never in timed heuristic construction",
        "audit_scope": "all traversable states: h0<=fast h1<=old h2=optimized h2; goal-reachable states: admissibility; all directed edges: consistency",
        "prepare_time_scope": "validation/conversion, min-weight scan and fresh heuristic/cache construction; excludes lazy estimates",
        "search_time_scope": "A* including lazy heuristic/cache queries, counters and path reconstruction",
        "api_time_scope": "fresh complete query including preparation, heuristic, search, loop and result collection; excludes oracle/validation/I/O",
        "statistics_unit": "one map; first median across batches, then paired aggregation across maps",
        "baseline_equivalence_verified": True, "optimized_equivalence_verified": True, "fast_h1_equivalence_verified": True,
        "garbage_collection": "normal interpreter behavior", "source_sha256": source_hashes,
        "source_baseline_sha256": sha256((output_dir / "source_baseline.zip").read_bytes()).hexdigest(),
        "environment": {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                        "processor": platform.processor(), "logical_cpus": os.cpu_count()},
    }
    random.Random(config["map_order_seed"]).shuffle(cases)
    config["measurement_order"] = [case["instance_id"] for case in cases]
    order_counts, position_counts, adjacent_counts = _order_balance(cases, batches)
    config["random_quartet_order_counts"] = order_counts
    config["random_strategy_position_counts"] = position_counts
    config["random_adjacent_pair_counts"] = adjacent_counts
    _write_json(output_dir / "config.json", config)
    paths, calibrations, cost_trace = [], [], []
    n_rows = 0
    print(f"Prepared {len(cases)-1} reused random maps + 1 official; h2 implementation comparison; four strategies; batches={batches}", flush=True)
    try:
        with (output_dir / "raw.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
            writer.writeheader()
            for index, case in enumerate(cases):
                estimates = {}
                for heuristic in ORDERS[index % len(ORDERS)]:
                    for _ in range(warmup):
                        _measure_once(case, heuristic)
                    samples = [_measure_once(case, heuristic) for _ in range(2)]
                    estimates[heuristic] = median(timing for _, timing in samples)
                    result = samples[0][0].result
                    paths.append({"instance_id": case["instance_id"], "heuristic": heuristic,
                                  "map_sha256": case["sha256"], "cost": result.cost, "path": result.path})
                    if case["source"] == "official":
                        terrain = prepare_terrain(case["start"], case["end"], case["grid"])
                        total = 0
                        for step, (row, col) in enumerate(result.path):
                            entering = 0 if step == 0 or (row, col) == terrain.end else terrain.costs[row][col]
                            total += entering
                            cost_trace.append({"heuristic": heuristic, "step": step, "row": row, "col": col,
                                               "cell": case["grid"][row][col], "entering_cost": entering, "cumulative_cost": total})
                loops = choose_batch_loops(estimates, target_batch_ms, max_batch_ms, max_loops)
                calibrations.append({"instance_id": case["instance_id"],
                                     **{f"{name}_api_ms": estimates[name] for name in HEURISTICS}, "batch_loops": loops})
                for batch in range(batches):
                    schedule = case["random_order_index"] * batches + batch if case["source"] == "random" else batch
                    for order, heuristic in enumerate(ORDERS[schedule % len(ORDERS)]):
                        data = measure_batch(case, heuristic, loops)
                        writer.writerow({"run_id": run_id, "instance_id": case["instance_id"], "map_file": case["map_file"],
                                         "map_sha256": case["sha256"], "source": case["source"], "size": case["size"],
                                         "seed": case["seed"], "heuristic": heuristic, "batch_index": batch,
                                         "batch_loops": loops, "order_in_quartet": order, **data})
                        n_rows += 1
                stream.flush()
                if (index + 1) % 5 == 0 or index + 1 == len(cases):
                    print(f"Measured {index+1}/{len(cases)} maps; rows={n_rows}; last={case['instance_id']}", flush=True)
        _write_csv(output_dir / "calibration.csv", calibrations)
        _write_csv(output_dir / "official_cost_trace.csv", cost_trace)
        _write_json(output_dir / "paths.json", paths)
        config.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(), raw_rows=n_rows,
                      all_results_correct=True, total_timed_calls=sum(row["batch_loops"] for row in calibrations) * batches * len(HEURISTICS))
        _write_json(output_dir / "config.json", config)
    except Exception as error:
        config.update(status="failed", failure=f"{type(error).__name__}: {error}")
        _write_json(output_dir / "config.json", config)
        raise
    if analyze:
        from experiments.analyze_two_step_optimized import analyze_results
        analyze_results(output_dir)
        config = read_case(output_dir / "config.json")
    print(f"Completed. Results: {output_dir}", flush=True)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="题二：两步前瞻等值实现优化的四策略配对实验")
    parser.add_argument("--out", type=Path, default=ROOT / "results/problem2_two_step_optimized")
    parser.add_argument("--dataset", type=Path, default=ROOT / "results/problem2_lookahead/dataset")
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--target-batch-ms", type=float, default=25)
    parser.add_argument("--max-batch-ms", type=float, default=200)
    parser.add_argument("--max-loops", type=int, default=128)
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_experiment(args.out, dataset_dir=args.dataset, batches=args.batches, warmup=args.warmup,
                       target_batch_ms=args.target_batch_ms, max_batch_ms=args.max_batch_ms,
                       max_loops=args.max_loops, analyze=not args.no_analysis)
    except (ValueError, OSError) as error:
        parser.exit(2, f"实验配置或文件错误：{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
