"""题二：零启发、原始曼哈顿与修正曼哈顿的配对对照。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
from itertools import permutations
import json
import os
from pathlib import Path
import platform
import random
from statistics import median
import sys
from time import perf_counter_ns
from typing import Any

from experiments.exp1_heuristic import choose_batch_loops
from experiments.weighted_dataset import (
    create_weighted_dataset, input_sha256, reference_to_goal, validate_weighted_path,
)
from src.heuristics import corrected_manhattan, manhattan
from src.main import read_case
from src.problem2_terrain import search_weighted


ROOT = Path(__file__).resolve().parent.parent
HEURISTICS = ("zero", "manhattan", "corrected_manhattan")
ORDERS = tuple(permutations(HEURISTICS))
COUNTERS = ("valid_pops", "expansions", "generated_successors", "pushes", "stale_pops", "heap_peak")
RAW_FIELDS = (
    "run_id", "instance_id", "map_file", "map_sha256", "source", "size", "seed", "heuristic",
    "batch_index", "batch_loops", "order_in_triplet", "path_cost", "reference_cost", "optimal_gap", "correct",
    *COUNTERS, "search_ms_mean", "api_ms_mean", "batch_api_ms",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cell_weight(cell: Any) -> int:
    if type(cell) is int:
        return cell
    if isinstance(cell, str) and cell.isascii() and cell.isdigit():
        return int(cell)
    if cell in ("S", "E"):
        return 1
    raise ValueError(f"未定义的权值格：{cell!r}")


def audit_heuristics(case: dict[str, Any], exact: dict[tuple[int, int], int]):
    """逐状态和有向边检查可采纳性/一致性，不参与搜索性能计时。"""
    grid, goal = case["grid"], tuple(case["end"])
    minimum = min(
        (_cell_weight(cell) for r, row in enumerate(grid) for c, cell in enumerate(row)
         if cell != "X" and (r, c) != goal), default=0,
    )
    estimates = {
        "zero": {point: 0 for point in exact},
        "manhattan": {point: manhattan(point, goal) for point in exact},
        "corrected_manhattan": {point: corrected_manhattan(point, goal, minimum) for point in exact},
    }
    rows, witnesses = [], []
    for name in HEURISTICS:
        values = estimates[name]
        over = [point for point in exact if values[point] > exact[point]]
        inconsistent = 0
        edge_count = 0
        for point in exact:
            r, c = point
            for other in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if other not in exact:
                    continue
                edge_count += 1
                weight = 0 if other == goal else _cell_weight(grid[other[0]][other[1]])
                inconsistent += values[point] > weight + values[other]
        rows.append({
            "instance_id": case["instance_id"], "source": case["source"], "size": case["size"],
            "heuristic": name, "n_reachable_states": len(exact),
            "n_directed_edges": edge_count,
            "admissibility_violations": len(over), "consistency_violations": inconsistent,
            "max_overestimate": max((values[point] - exact[point] for point in over), default=0),
            "goal_h": values[goal],
        })
        if over:
            point = min(over, key=lambda p: (manhattan(p, goal), p))
            witnesses.append({
                "instance_id": case["instance_id"], "heuristic": name, "position": point,
                "goal": goal, "heuristic_value": values[point], "exact_remaining_cost": exact[point],
                "overestimate": values[point] - exact[point],
            })
    return rows, witnesses


def _check_result(case: dict[str, Any], result: Any) -> None:
    if result.cost != case["best_cost"]:
        raise AssertionError(f"{case['instance_id']}：搜索代价与独立 Dijkstra 不一致")
    validate_weighted_path(case["grid"], case["start"], case["end"], result.path, case["best_cost"], goal_free=True)


def _measure_once(case: dict[str, Any], heuristic: str):
    started = perf_counter_ns()
    result = search_weighted(case["start"], case["end"], case["grid"], heuristic=heuristic, trace=False)
    api_ms = (perf_counter_ns() - started) / 1_000_000
    _check_result(case, result)
    return result, api_ms


def measure_batch(case: dict[str, Any], heuristic: str, loops: int):
    """重复结果批外校验；调用均摊时间包含循环和结果收集。"""
    started = perf_counter_ns()
    results = [
        search_weighted(case["start"], case["end"], case["grid"], heuristic=heuristic, trace=False)
        for _ in range(loops)
    ]
    total_ms = (perf_counter_ns() - started) / 1_000_000
    first = results[0]
    counts = tuple(getattr(first.stats, field) for field in COUNTERS)
    for result in results:
        _check_result(case, result)
        if tuple(getattr(result.stats, field) for field in COUNTERS) != counts:
            raise AssertionError("同一地图和启发函数的计数不稳定")
    return {
        "path_cost": first.cost, "reference_cost": case["best_cost"],
        "optimal_gap": first.cost - case["best_cost"], "correct": True,
        **{field: getattr(first.stats, field) for field in COUNTERS},
        "search_ms_mean": sum(result.stats.elapsed_ms for result in results) / loops,
        "api_ms_mean": total_ms / loops, "batch_api_ms": total_ms,
    }


def _load_cases(dataset_dir: Path, output_dir: Path):
    manifest = read_case(dataset_dir / "manifest.json")
    if manifest.get("status") not in ("complete", "completed"):
        raise ValueError("数据集尚未完整生成")
    cases = []
    for random_order_index, relative in enumerate(manifest["accepted_files"]):
        path = (dataset_dir / relative).resolve()
        if not path.is_relative_to(dataset_dir.resolve()):
            raise ValueError("地图路径越过数据集目录")
        case = read_case(path)
        if case["sha256"] != input_sha256(case):
            raise ValueError(f"地图哈希不一致：{relative}")
        case.update(source="random", map_file=os.path.relpath(path, output_dir), random_order_index=random_order_index)
        cases.append(case)
    official = read_case(ROOT / "cases" / "problem2.json")
    official.update(instance_id="official_5x5", source="official", size=5, seed="official", map_file="official_map.json")
    official["best_cost"] = reference_to_goal(official["grid"], official["end"])[tuple(official["start"])]
    official["sha256"] = input_sha256(official)
    _write_json(output_dir / "official_map.json", official)
    cases.append(official)
    if len({case["instance_id"] for case in cases}) != len(cases):
        raise ValueError("地图实例编号重复")
    return manifest, cases


def boundary_demo(output_dir: Path) -> None:
    """单独展示普通格允许 0 时的失效边界，不混入正式性能统计。"""
    case = {
        "instance_id": "outside_domain_zero_cost", "source": "boundary", "size": 2,
        "start": [0, 0], "end": [0, 2], "grid": [["S", 1, "E"], [0, 0, 0]],
    }
    exact = reference_to_goal(case["grid"], case["end"], goal_free=True)
    reference = exact[tuple(case["start"])]
    outcomes = {}
    for name in HEURISTICS:
        result = search_weighted(case["start"], case["end"], case["grid"], heuristic=name)
        validate_weighted_path(case["grid"], case["start"], case["end"], result.path, result.cost)
        outcomes[name] = {
            "cost": result.cost, "reference_cost": reference,
            "optimal_gap": result.cost - reference, "path": result.path,
        }
    audit, _ = audit_heuristics(case, exact)
    _write_json(output_dir / "boundary_demo.json", {
        "scope": "separate example outside the positive-weight domain; excluded from performance rows and main all_results_correct",
        "changed_assumption": "ordinary terrain may have cost 0 instead of at least 1",
        "case": case, "outcomes": outcomes, "heuristic_audit": audit,
    })


def run_experiment(
    output_dir: Path,
    *,
    size: int = 31,
    count: int = 30,
    master_seed: int = 20261022,
    batches: int = 5,
    warmup: int = 2,
    target_batch_ms: float = 25,
    max_batch_ms: float = 200,
    max_loops: int = 128,
    dataset_dir: Path | None = None,
    analyze: bool = True,
) -> dict[str, Any]:
    if type(batches) is not int or type(warmup) is not int or batches < 1 or warmup < 1:
        raise ValueError("batches 和 warmup 必须为正整数")
    if max_loops < 1 or target_batch_ms <= 0 or max_batch_ms <= 0:
        raise ValueError("批次循环上限和目标时长必须为正")
    if analyze and importlib.util.find_spec("matplotlib") is None:
        raise ValueError("生成图表需要 matplotlib；可使用 --no-analysis 只采集数据")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("结果目录非空，请指定新的 --out，避免覆盖旧实验")
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset_dir is None:
        dataset_dir = output_dir / "dataset"
        create_weighted_dataset(dataset_dir, size=size, count=count, master_seed=master_seed)
    else:
        dataset_dir = dataset_dir.resolve()
    manifest, cases = _load_cases(dataset_dir, output_dir)
    audits, witnesses = [], []
    for case in cases:
        exact = reference_to_goal(case["grid"], case["end"], goal_free=True)
        if exact.get(tuple(case["start"])) != case["best_cost"]:
            raise ValueError(f"{case['instance_id']} 的参考代价不一致")
        entries, examples = audit_heuristics(case, exact)
        audits.extend(entries)
        witnesses.extend(examples)
    _write_csv(output_dir / "heuristic_audit.csv", audits)
    _write_json(output_dir / "heuristic_witnesses.json", witnesses)
    boundary_demo(output_dir)

    source_files = (
        "src/astar.py", "src/grid.py", "src/heuristics.py", "src/problem2_terrain.py", "src/stats.py",
        "experiments/exp1_heuristic.py", "experiments/maze_dataset.py",
        "experiments/weighted_dataset.py", "experiments/exp2_weighted_terrain.py",
    )
    run_id = "p2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    config = {
        "experiment": "problem2_weighted_heuristics_v1", "run_id": run_id,
        "status": "running", "analysis_status": "pending" if analyze else "skipped",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": os.path.relpath(dataset_dir, output_dir), "dataset_config": manifest["config"],
        "dataset_manifest_sha256": sha256((dataset_dir / "manifest.json").read_bytes()).hexdigest(),
        "random_instances": len(cases) - 1, "official_instances": 1, "measured_instances": len(cases),
        "heuristics": HEURISTICS, "batches": batches, "warmup_per_heuristic": warmup,
        "calibration_calls_per_heuristic": 2, "target_fast_batch_ms": target_batch_ms,
        "max_estimated_slow_batch_ms": max_batch_ms, "max_batch_loops": max_loops,
        "calibration_rule": "same loops; min(ceil(target/fastest), floor(max_batch/slowest), max_loops), at least 1",
        "triplet_order_rule": "random maps: (saved random_order_index*batches+batch_index)%6; official: batch_index%6",
        "triplet_orders": ORDERS, "map_order_seed": master_seed ^ 0x21A7,
        "neighbor_order": "up,right,down,left", "tie_break": "(f,h,insertion_counter)",
        "trace": False, "goal_entering_cost": 0, "start_initial_cost": 0,
        "oracle": "independent reverse Dijkstra; reversed v->u retains original entering weight w(v)",
        "audit_scope": "all reachable states for admissibility, all directed traversable edges for consistency",
        "search_time_scope": "internal A* including counters/path reconstruction; excludes grid preparation",
        "api_time_scope": "batch time per call incl grid validation/conversion, loop/result collection; excludes reference/validation/I/O",
        "statistics_unit": "one saved map; median across batches before cross-map aggregation",
        "boundary_demo": "boundary_demo.json; outside positive-weight domain, excluded from main performance data",
        "garbage_collection": "normal interpreter behavior",
        "environment": {
            "python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        },
        "source_sha256": {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files},
    }
    random.Random(config["map_order_seed"]).shuffle(cases)
    config["measurement_order"] = [case["instance_id"] for case in cases]
    permutation_counts = {",".join(order): 0 for order in ORDERS}
    for case in cases:
        if case["source"] == "random":
            for batch in range(batches):
                order = ORDERS[(case["random_order_index"] * batches + batch) % len(ORDERS)]
                permutation_counts[",".join(order)] += 1
    config["random_triplet_permutation_counts"] = permutation_counts
    _write_json(output_dir / "config.json", config)
    paths, calibrations, cost_trace = [], [], []
    n_rows = 0
    print(f"Prepared {len(cases)-1} random weighted maps + 1 official; three heuristics; batches={batches}", flush=True)
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
                    result = samples[0][0]
                    paths.append({
                        "instance_id": case["instance_id"], "heuristic": heuristic,
                        "map_sha256": case["sha256"], "cost": result.cost, "path": result.path,
                    })
                    if case["source"] == "official":
                        total = 0
                        for step, (r, c) in enumerate(result.path):
                            entering = 0 if step == 0 or (r, c) == tuple(case["end"]) else _cell_weight(case["grid"][r][c])
                            total += entering
                            cost_trace.append({
                                "heuristic": heuristic, "step": step, "row": r, "col": c,
                                "cell": case["grid"][r][c], "entering_cost": entering, "cumulative_cost": total,
                            })
                loops = choose_batch_loops(estimates, target_batch_ms, max_batch_ms, max_loops)
                calibrations.append({
                    "instance_id": case["instance_id"],
                    **{f"{name}_api_ms": estimates[name] for name in HEURISTICS}, "batch_loops": loops,
                })
                for batch in range(batches):
                    schedule_index = case["random_order_index"] * batches + batch if case["source"] == "random" else batch
                    for order, heuristic in enumerate(ORDERS[schedule_index % len(ORDERS)]):
                        data = measure_batch(case, heuristic, loops)
                        writer.writerow({
                            "run_id": run_id, "instance_id": case["instance_id"], "map_file": case["map_file"],
                            "map_sha256": case["sha256"], "source": case["source"], "size": case["size"],
                            "seed": case["seed"], "heuristic": heuristic, "batch_index": batch,
                            "batch_loops": loops, "order_in_triplet": order, **data,
                        })
                        n_rows += 1
                stream.flush()
                if (index + 1) % 5 == 0 or index + 1 == len(cases):
                    print(f"Measured {index+1}/{len(cases)} maps; rows={n_rows}; last={case['instance_id']}", flush=True)
        _write_csv(output_dir / "calibration.csv", calibrations)
        _write_csv(output_dir / "official_cost_trace.csv", cost_trace)
        _write_json(output_dir / "paths.json", paths)
        config.update(
            status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
            raw_rows=n_rows, all_results_correct=True,
            total_timed_calls=sum(row["batch_loops"] for row in calibrations) * batches * len(HEURISTICS),
        )
        _write_json(output_dir / "config.json", config)
    except Exception as error:
        config.update(status="failed", failure=f"{type(error).__name__}: {error}")
        _write_json(output_dir / "config.json", config)
        raise
    if analyze:
        from experiments.analyze_weighted_heuristics import analyze_results
        analyze_results(output_dir)
        config = read_case(output_dir / "config.json")
    print(f"Completed. Results: {output_dir}", flush=True)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="题二：三种启发的最优性、可采纳性和性能对照")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "problem2_heuristics")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--size", type=int, default=31)
    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--target-batch-ms", type=float, default=25)
    parser.add_argument("--max-batch-ms", type=float, default=200)
    parser.add_argument("--max-loops", type=int, default=128)
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_experiment(
            args.out, size=args.size, count=args.instances, master_seed=args.seed,
            batches=args.batches, warmup=args.warmup, target_batch_ms=args.target_batch_ms,
            max_batch_ms=args.max_batch_ms, max_loops=args.max_loops,
            dataset_dir=args.dataset, analyze=not args.no_analysis,
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"实验配置或文件错误：{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
