"""题二补充：当前修正曼哈顿与一步、两步地形前瞻的正式配对实验。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
from itertools import permutations
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
from statistics import median
import sys
from time import perf_counter_ns
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.exp1_heuristic import choose_batch_loops
from experiments.lookahead_heuristics import prepare_terrain, search_lookahead
from experiments.weighted_dataset import input_sha256, reference_to_goal, validate_weighted_path
from src.main import read_case
from src.problem2_terrain import search_weighted


ROOT = Path(__file__).resolve().parent.parent
HEURISTICS = ("corrected_manhattan", "lookahead_1", "lookahead_2")
DEPTHS = dict(zip(HEURISTICS, range(3)))
ORDERS = tuple(permutations(HEURISTICS))
COUNTERS = ("valid_pops", "expansions", "generated_successors", "pushes", "stale_pops", "heap_peak")
RAW_FIELDS = (
    "run_id", "instance_id", "map_file", "map_sha256", "source", "size", "seed", "heuristic",
    "batch_index", "batch_loops", "order_in_triplet", "path_cost", "reference_cost", "optimal_gap", "correct",
    *COUNTERS, "prepare_ms_mean", "search_ms_mean", "api_ms_mean", "batch_api_ms",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("不能写入空记录表")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_heuristics(case: dict[str, Any], exact: dict[tuple[int, int], int]):
    """全格单调性/全边一致性审计；真值仅在审计中使用。"""
    terrain = prepare_terrain(case["start"], case["end"], case["grid"])
    positions = list(terrain.positions())
    tables = []
    for depth in range(3):
        estimate = terrain.heuristic(depth)
        tables.append({point: estimate(point) for point in positions})
    rows = []
    for depth, name in enumerate(HEURISTICS):
        values = tables[depth]
        previous = tables[max(0, depth - 1)]
        over = [point for point in exact if values[point] > exact[point]]
        edges = [(point, other, weight) for point in positions for other, weight in terrain.neighbors(point)]
        row = {
            "instance_id": case["instance_id"], "source": case["source"], "size": case["size"],
            "heuristic": name, "n_states": len(positions), "n_reachable_states": len(exact),
            "n_directed_edges": len(edges), "admissibility_violations": len(over),
            "consistency_violations": sum(values[u] > weight + values[v] for u, v, weight in edges),
            "max_overestimate": max((values[p] - exact[p] for p in over), default=0),
            "goal_h": values[terrain.end],
            "dominance_violations": sum(values[p] < previous[p] for p in positions),
            "strictly_stronger_states": sum(values[p] > previous[p] for p in positions),
        }
        if any(row[field] for field in ("admissibility_violations", "consistency_violations", "goal_h", "dominance_violations")):
            raise AssertionError(f"启发性质未通过：{case['instance_id']} {name}")
        rows.append(row)
    return rows


def _check_result(case: dict[str, Any], result: Any) -> None:
    if result.cost != case["best_cost"]:
        raise AssertionError(f"{case['instance_id']} 的代价与独立 Dijkstra 不一致")
    validate_weighted_path(case["grid"], case["start"], case["end"], result.path, case["best_cost"])


def _measure_once(case: dict[str, Any], heuristic: str):
    started = perf_counter_ns()
    measured = search_lookahead(case["start"], case["end"], case["grid"], depth=DEPTHS[heuristic])
    total_ms = (perf_counter_ns() - started) / 1_000_000
    _check_result(case, measured.result)
    return measured, total_ms


def measure_batch(case: dict[str, Any], heuristic: str, loops: int):
    """每次新建缓存；计时后逐一核验，API 时间包含全部准备与搜索。"""
    if type(loops) is not int or loops < 1:
        raise ValueError("批次循环数必须是正整数")
    started = perf_counter_ns()
    measured = [
        search_lookahead(case["start"], case["end"], case["grid"], depth=DEPTHS[heuristic])
        for _ in range(loops)
    ]
    total_ms = (perf_counter_ns() - started) / 1_000_000
    first = measured[0].result
    counts = tuple(getattr(first.stats, field) for field in COUNTERS)
    for run in measured:
        _check_result(case, run.result)
        if tuple(getattr(run.result.stats, field) for field in COUNTERS) != counts:
            raise AssertionError("同图同策略的确定性计数不一致")
    reference = case["best_cost"]
    return {
        "path_cost": first.cost, "reference_cost": reference,
        "optimal_gap": 0 if reference is None else first.cost - reference, "correct": True,
        **{field: getattr(first.stats, field) for field in COUNTERS},
        "prepare_ms_mean": sum(run.prepare_ms for run in measured) / loops,
        "search_ms_mean": sum(run.result.stats.elapsed_ms for run in measured) / loops,
        "api_ms_mean": total_ms / loops, "batch_api_ms": total_ms,
    }


def _load_cases(dataset_dir: Path, output_dir: Path):
    """复制所使用的实际输入，补充实验可脱离旧结果独立复核。"""
    manifest_path = dataset_dir / "manifest.json"
    manifest = read_case(manifest_path)
    if manifest.get("status") not in ("complete", "completed"):
        raise ValueError("数据集尚未生成完成")
    accepted = manifest.get("accepted_files")
    if not isinstance(accepted, list) or not accepted or len(accepted) != len(set(accepted)):
        raise ValueError("数据集地图列表为空或重复")
    destination = output_dir / "dataset"
    destination.mkdir()
    shutil.copyfile(manifest_path, destination / "manifest.json")
    cases = []
    for random_order_index, relative in enumerate(accepted):
        path = (dataset_dir / relative).resolve()
        copied = (destination / relative).resolve()
        if not path.is_relative_to(dataset_dir) or not copied.is_relative_to(destination.resolve()):
            raise ValueError("地图路径越过数据集目录")
        case = read_case(path)
        if case.get("sha256") != input_sha256(case):
            raise ValueError(f"地图哈希不一致：{relative}")
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copied)
        case.update(source="random", map_file=os.path.relpath(copied, output_dir), random_order_index=random_order_index)
        cases.append(case)
    official = read_case(ROOT / "cases" / "problem2.json")
    official.update(instance_id="official_5x5", source="official", size=len(official["grid"]), seed="official", map_file="official_map.json")
    official["best_cost"] = reference_to_goal(official["grid"], official["end"]).get(tuple(official["start"]))
    official["sha256"] = input_sha256(official)
    _write_json(output_dir / "official_map.json", official)
    cases.append(official)
    if len({case["instance_id"] for case in cases}) != len(cases):
        raise ValueError("地图实例编号重复")
    return manifest, cases


def run_experiment(
    output_dir: Path, *, dataset_dir: Path | None = None, batches: int = 5,
    warmup: int = 2, target_batch_ms: float = 25, max_batch_ms: float = 200,
    max_loops: int = 128, analyze: bool = True,
) -> dict[str, Any]:
    if any(type(value) is not int or value < 1 for value in (batches, warmup, max_loops)):
        raise ValueError("batches、warmup 和 max_loops 必须为正整数")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
           for value in (target_batch_ms, max_batch_ms)):
        raise ValueError("计时目标和上限必须为有限正数")
    if analyze and importlib.util.find_spec("matplotlib") is None:
        raise ValueError("生成图表需要 matplotlib；可加 --no-analysis")
    output_dir = Path(output_dir).resolve()
    dataset_dir = Path(dataset_dir or ROOT / "results/problem2_heuristics/dataset").resolve()
    if output_dir == dataset_dir or output_dir.is_relative_to(dataset_dir):
        raise ValueError("补充结果目录不能位于输入数据目录内")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("结果目录非空，请使用新目录避免覆盖")
    if not (dataset_dir / "manifest.json").is_file():
        raise ValueError("输入数据集缺少 manifest.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, cases = _load_cases(dataset_dir, output_dir)
    audits = []
    for case in cases:
        exact = reference_to_goal(case["grid"], case["end"])
        if exact.get(tuple(case["start"])) != case["best_cost"]:
            raise ValueError(f"{case['instance_id']} 的参考代价不符")
        audits.extend(audit_heuristics(case, exact))
        # 适配器基线必须与原接口具有完全相同的确定性搜索行为。
        baseline = search_weighted(case["start"], case["end"], case["grid"], trace=True)
        adapted = search_lookahead(case["start"], case["end"], case["grid"], depth=0, trace=True).result
        if (baseline.cost, baseline.path, baseline.expanded_order) != (adapted.cost, adapted.path, adapted.expanded_order):
            raise AssertionError("统一实验适配器改变了旧基线的搜索行为")
        if any(getattr(baseline.stats, key) != getattr(adapted.stats, key) for key in COUNTERS):
            raise AssertionError("统一实验适配器改变了旧基线计数")
    _write_csv(output_dir / "heuristic_audit.csv", audits)

    # 明确的代码/测试输入白名单，包含 src.main 的全部间接导入，便于独立复现。
    # 文档可在运行后补写，不作为执行源码冻结检查的一部分。
    existing_sources = sorted({
        path.relative_to(ROOT).as_posix()
        for pattern in ("src/*.py", "experiments/*.py", "tests/*.py", "cases/*.json", "requirements-experiments.txt")
        for path in ROOT.glob(pattern) if path.is_file()
    })
    source_hashes = {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in existing_sources}
    with ZipFile(output_dir / "source_baseline.zip", "w", ZIP_DEFLATED) as archive:
        for name in existing_sources:
            archive.write(ROOT / name, name)
    run_id = "p2_lookahead_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    master_seed = manifest["config"]["master_seed"]
    config = {
        "experiment": "problem2_lookahead_v1", "run_id": run_id,
        "status": "running", "analysis_status": "pending" if analyze else "skipped",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset_dir), "dataset_dir": "dataset", "dataset_config": manifest["config"],
        "dataset_manifest_sha256": sha256((output_dir / "dataset/manifest.json").read_bytes()).hexdigest(),
        "random_instances": len(cases) - 1, "official_instances": 1, "measured_instances": len(cases),
        "heuristics": HEURISTICS, "depths": DEPTHS, "batches": batches, "warmup_per_heuristic": warmup,
        "calibration_calls_per_heuristic": 2, "target_fast_batch_ms": target_batch_ms,
        "max_estimated_slow_batch_ms": max_batch_ms, "max_batch_loops": max_loops,
        "calibration_rule": "same loops; min(ceil(target/fastest), floor(max_batch/slowest), max_loops), at least 1",
        "triplet_orders": ORDERS, "triplet_order_rule": "random: (saved random_order_index*batches+batch_index)%6; official: batch_index%6",
        "map_order_seed": master_seed ^ 0x109A, "neighbor_order": "up,right,down,left", "tie_break": "(f,h,insertion_counter)",
        "trace": False, "goal_entering_cost": 0, "start_initial_cost": 0,
        "heuristic_implementation": "lazy per-query memoization; no cross-query cache; depth 0 evaluated directly",
        "heuristic_formula": "h0=m*max(0,Manhattan-1); h(k+1)(n)=min_v(c(n,v)+hk(v)); every layer h(goal)=0; isolated states retain previous layer",
        "oracle": "independent reverse Dijkstra; never supplied to heuristic construction or timed search",
        "audit_scope": "all traversable states for dominance, all goal-reachable states for admissibility, all traversable directed edges for consistency",
        "prepare_time_scope": "grid validation/conversion, minimum scan, fresh heuristic/cache construction; excludes lazy estimates",
        "search_time_scope": "A* including lazy heuristic/cache queries, counters and path reconstruction; excludes grid preparation",
        "api_time_scope": "complete fresh query including preparation, lazy estimates, search, loop and result collection; excludes oracle/validation/I/O",
        "statistics_unit": "one saved map; median across batches before cross-map aggregation",
        "baseline_equivalence_verified": True, "garbage_collection": "normal interpreter behavior",
        "source_sha256": source_hashes,
        "environment": {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                        "processor": platform.processor(), "logical_cpus": os.cpu_count()},
    }
    random.Random(config["map_order_seed"]).shuffle(cases)
    config["measurement_order"] = [case["instance_id"] for case in cases]
    permutation_counts = {",".join(order): 0 for order in ORDERS}
    for case in cases:
        if case["source"] == "random":
            for batch in range(batches):
                permutation_counts[",".join(ORDERS[(case["random_order_index"] * batches + batch) % 6])] += 1
    config["random_triplet_permutation_counts"] = permutation_counts
    _write_json(output_dir / "config.json", config)
    paths, calibrations, cost_trace = [], [], []
    n_rows = 0
    print(f"Prepared {len(cases)-1} reused random maps + 1 official; three corrected heuristics; batches={batches}", flush=True)
    try:
        with (output_dir / "raw.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
            writer.writeheader()
            for index, case in enumerate(cases):
                estimates = {}
                for heuristic in ORDERS[index % 6]:
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
                    for order, heuristic in enumerate(ORDERS[schedule % 6]):
                        data = measure_batch(case, heuristic, loops)
                        writer.writerow({"run_id": run_id, "instance_id": case["instance_id"], "map_file": case["map_file"],
                                         "map_sha256": case["sha256"], "source": case["source"], "size": case["size"],
                                         "seed": case["seed"], "heuristic": heuristic, "batch_index": batch,
                                         "batch_loops": loops, "order_in_triplet": order, **data})
                        n_rows += 1
                stream.flush()
                if (index + 1) % 5 == 0 or index + 1 == len(cases):
                    print(f"Measured {index+1}/{len(cases)} maps; rows={n_rows}; last={case['instance_id']}", flush=True)
        _write_csv(output_dir / "calibration.csv", calibrations)
        _write_csv(output_dir / "official_cost_trace.csv", cost_trace)
        _write_json(output_dir / "paths.json", paths)
        config.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(), raw_rows=n_rows,
                      all_results_correct=True, total_timed_calls=sum(row["batch_loops"] for row in calibrations) * batches * 3)
        _write_json(output_dir / "config.json", config)
    except Exception as error:
        config.update(status="failed", failure=f"{type(error).__name__}: {error}")
        _write_json(output_dir / "config.json", config)
        raise
    if analyze:
        from experiments.analyze_lookahead import analyze_results
        analyze_results(output_dir)
        config = read_case(output_dir / "config.json")
    print(f"Completed. Results: {output_dir}", flush=True)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="题二补充：三种修正曼哈顿的按需前瞻对照")
    parser.add_argument("--out", type=Path, default=ROOT / "results/problem2_lookahead")
    parser.add_argument("--dataset", type=Path, default=ROOT / "results/problem2_heuristics/dataset")
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
