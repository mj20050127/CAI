"""题四：H=4 的 24 种固定全局宝藏优先级对照。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
from itertools import permutations
import json
import math
from pathlib import Path
import platform
import random
from statistics import median
import sys
from time import perf_counter_ns

from experiments.exp1_heuristic import choose_batch_loops
from experiments.exp3_minimax_alpha_beta import COUNTERS, _snapshot_sources, _write_csv, _write_json
from experiments.exp4_depths import load_input
from experiments.treasure_reference import reference_distances, reference_game
from src.problem4_treasure import build_distance_matrix, evaluate_treasure


ROOT = Path(__file__).resolve().parent.parent
DEPTH = 4
ALGORITHMS = ("minimax", "alpha_beta")
ORDER_SEED = 20261029


def priority_id(order):
    """可直接在 CSV 中辨认的全局优先级。"""
    return ">".join(order)


def _check_result(result, reference, algorithm, expected=None):
    if algorithm not in ALGORITHMS:
        raise ValueError("未知博弈算法")
    if result.best_value != reference.value or result.best_target != reference.best_targets[0]:
        raise AssertionError("求解结果与独立排列参考的根值或按当前顺序打破平局的首步不符")
    stats = result.stats
    if any(type(getattr(stats, key)) is not int or getattr(stats, key) < 0 for key in COUNTERS):
        raise AssertionError("计数必须为非负整数")
    if not (1 <= stats.leaf_evals <= reference.leaf_count
            and stats.leaf_evals <= stats.node_visits <= reference.node_count):
        raise AssertionError("访问次数超出完整搜索树")
    if stats.cutoff_events > stats.pruned_branches or (stats.cutoff_events == 0) != (stats.pruned_branches == 0):
        raise AssertionError("有效截断与跳过兄弟数不一致")
    if algorithm == "minimax" and (stats.node_visits != reference.node_count
            or stats.leaf_evals != reference.leaf_count or stats.cutoff_events or stats.pruned_branches):
        raise AssertionError("Minimax 必须遍历完整树而不能剪枝")
    if expected is not None and any(getattr(stats, key) != expected[key] for key in COUNTERS):
        raise AssertionError("计时结果的搜索计数与审计结果不一致")
    if result.trace:
        raise AssertionError("正式运行不得开启轨迹")


def _measure_once(matrix, values, algorithm):
    started = perf_counter_ns()
    result = evaluate_treasure(matrix, values, DEPTH, use_alpha_beta=algorithm == "alpha_beta", trace=False)
    return result, (perf_counter_ns() - started) / 1_000_000


def _measure_batch(matrix, values, algorithm, loops, reference, expected):
    """保留所有返回值，批次计时结束后验证每次结果。"""
    started = perf_counter_ns()
    runs = [_measure_once(matrix, values, algorithm) for _ in range(loops)]
    batch_ms = (perf_counter_ns() - started) / 1_000_000
    for result, _ in runs:
        _check_result(result, reference, algorithm, expected)
    first = runs[0][0]
    return {"best_target": first.best_target, "best_value": first.best_value,
            "reference_value": reference.value, "correct": True,
            **{field: getattr(first.stats, field) for field in COUNTERS},
            "search_ms_mean": sum(result.stats.elapsed_ms for result, _ in runs) / loops,
            "query_ms_mean": sum(query for _, query in runs) / loops,
            "total_ms_mean": batch_ms / loops, "batch_total_ms": batch_ms}


def run_experiment(output_dir, *, input_path=None, batches=5, warmup=2, max_loops=128,
                   target_batch_ms=25, max_batch_ms=200, analyze=True):
    """在一个输入上核验所有排列，然后对 24 个剪枝顺序及原序 Minimax 配对计时。"""
    if any(type(number) is not int or number < 1 for number in (batches, warmup, max_loops)):
        raise ValueError("批次、预热和循环上限必须是正整数")
    if type(analyze) is not bool:
        raise ValueError("分析选项必须为布尔值")
    if any(type(number) not in (int, float) or not math.isfinite(number) or number <= 0
           for number in (target_batch_ms, max_batch_ms)):
        raise ValueError("计时目标必须有限且为正")
    if analyze and importlib.util.find_spec("matplotlib") is None:
        raise ValueError("绘图需要 matplotlib")
    out = Path(output_dir).resolve()
    source = Path(input_path or ROOT / "cases/problem4.json").resolve()
    if source.is_relative_to(out) or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise ValueError("结果目录非空，或输入位于结果目录内；请指定新的空目录")
    payload, case = load_input(source)
    values = case["treasure_values"]
    if len(values) != 4 or any(">" in name for name in values):
        raise ValueError("本实验要求四个宝藏且标签不得含 >")
    independent = reference_distances(case["grid"], case["positions"])
    reference_game(independent, values, DEPTH)  # 验证输入，再创建输出目录。
    orders = list(permutations(values))
    identifiers = [priority_id(order) for order in orders]
    original = priority_id(tuple(values))
    ordered_values = {priority_id(order): {target: values[target] for target in order} for order in orders}
    references = {key: reference_game(independent, ordered_values[key], DEPTH) for key in identifiers}
    variants = [(key, "alpha_beta") for key in identifiers] + [(original, "minimax")]

    out.mkdir(parents=True, exist_ok=True)
    (out / "input").mkdir()
    (out / "input/problem4.json").write_bytes(payload)
    source_hashes, archive_hash = _snapshot_sources(out)
    run_id = "p4_order_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    config = {"experiment": "p4_order_v1", "run_id": run_id, "status": "running",
              "analysis_status": "pending" if analyze else "skipped",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "source_input": str(source),
              "input_file": "input/problem4.json", "input_sha256": sha256(payload).hexdigest(),
              "depth": DEPTH, "depth_unit": "one player action", "original_priority": original,
              "orders": identifiers, "algorithms": list(ALGORITHMS), "batches": batches, "warmup": warmup,
              "calibration_calls": 2, "target_fast_batch_ms": target_batch_ms,
              "max_estimated_slow_batch_ms": max_batch_ms, "max_loops": max_loops,
              "order_seed": ORDER_SEED,
              "execution_order_rule": "seeded shuffle of 25 configurations independently for each batch",
              "calibration_rule": "one shared loop count estimated from all 25 variants; fastest target and slowest cap",
              "priority_rule": "each state filters the same preconstructed global priority by remaining treasures",
              "time_scope": {"search_ms": "production search and counter updates, excluding input validation",
                             "query_ms": "evaluate_treasure input validation, matrix/value copying and search",
                             "total_ms": "outer batch elapsed divided by loops, includes wrappers and result collection"},
              "excluded": "distance matrix construction, preparing ordered dictionaries, correctness oracles, I/O and plotting",
              "statistics_unit": "one official map; batches are repeated measurements, not independent input maps",
              "utility": "score_a-score_b", "gain": "treasure_value-entering_cost_distance",
              "goal_free": False, "cache": False, "trace": False,
              "source_sha256": source_hashes, "source_baseline_sha256": archive_hash,
              "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}}
    _write_json(out / "config.json", config)
    try:
        matrix = build_distance_matrix(case["grid"], case["positions"])
        if matrix != independent:
            raise AssertionError("A* 距离矩阵与独立 Dijkstra 不一致")
        _write_json(out / "distance_matrix.json", matrix)
        _write_json(out / "dijkstra_reference.json", independent)
        proof, expected = [], {}
        for key in identifiers:
            reference = references[key]
            for algorithm in ALGORITHMS:
                result = evaluate_treasure(matrix, ordered_values[key], DEPTH,
                                           use_alpha_beta=algorithm == "alpha_beta", trace=False)
                _check_result(result, reference, algorithm)
                row = {"priority_id": key, "algorithm": algorithm, "depth": DEPTH,
                       "best_target": result.best_target, "best_value": result.best_value,
                       "reference_value": reference.value, "correct": True,
                       **{field: getattr(result.stats, field) for field in COUNTERS}}
                proof.append(row)
                expected[key, algorithm] = row
        _write_csv(out / "proof.csv", proof)
        print(f"Distance matrix and {len(proof)} search audits passed; priorities={len(identifiers)}", flush=True)

        estimates, calibration = {}, []
        calibration_order = variants.copy()
        rng = random.Random(ORDER_SEED)
        rng.shuffle(calibration_order)
        for key, algorithm in calibration_order:
            for _ in range(warmup):
                result, _ = _measure_once(matrix, ordered_values[key], algorithm)
                _check_result(result, references[key], algorithm, expected[key, algorithm])
            samples = []
            for _ in range(2):
                started = perf_counter_ns()
                result, _ = _measure_once(matrix, ordered_values[key], algorithm)
                samples.append((perf_counter_ns() - started) / 1_000_000)
                _check_result(result, references[key], algorithm, expected[key, algorithm])
            estimate = median(samples)
            estimates[f"{algorithm}:{key}"] = estimate
            calibration.append({"priority_id": key, "algorithm": algorithm, "estimate_ms": estimate})
        loops = choose_batch_loops(estimates, target_batch_ms, max_batch_ms, max_loops)
        _write_csv(out / "calibration.csv", calibration)
        config["batch_loops"] = loops
        _write_json(out / "config.json", config)
        raw = []
        for batch in range(batches):
            execution_order = variants.copy()
            rng.shuffle(execution_order)
            for position, (key, algorithm) in enumerate(execution_order):
                measured = _measure_batch(matrix, ordered_values[key], algorithm, loops,
                                          references[key], expected[key, algorithm])
                raw.append({"run_id": run_id, "priority_id": key, "algorithm": algorithm,
                            "depth": DEPTH, "batch_index": batch, "batch_loops": loops,
                            "execution_position": position, **measured})
            print(f"Measured batch {batch + 1}/{batches}; rows={len(raw)}", flush=True)
        _write_csv(out / "raw.csv", raw)
        config.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      raw_rows=len(raw), total_timed_calls=sum(row["batch_loops"] for row in raw),
                      all_results_correct=True)
        _write_json(out / "config.json", config)
    except Exception as error:
        config.update(status="failed", failure=f"{type(error).__name__}: {error}")
        _write_json(out / "config.json", config)
        raise
    if analyze:
        from experiments.analyze_treasure_order import analyze_results
        analyze_results(out)
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="题四：24种固定全局优先级对照")
    parser.add_argument("--out", type=Path, default=ROOT / "results/problem4_order")
    parser.add_argument("--input", type=Path, default=ROOT / "cases/problem4.json")
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-loops", type=int, default=128)
    parser.add_argument("--target-batch-ms", type=float, default=25)
    parser.add_argument("--max-batch-ms", type=float, default=200)
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = run_experiment(args.out, input_path=args.input, batches=args.batches, warmup=args.warmup,
                                max_loops=args.max_loops, target_batch_ms=args.target_batch_ms,
                                max_batch_ms=args.max_batch_ms, analyze=not args.no_analysis)
    except (ValueError, OSError) as error:
        parser.exit(2, f"实验配置或输入错误：{error}\n")
    print(f"Completed: {args.out}; rows={config['raw_rows']}; calls={config['total_timed_calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
