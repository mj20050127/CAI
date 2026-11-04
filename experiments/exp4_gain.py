"""题四：相同 Alpha-Beta 实现中原顺序与即时收益动态排序的对照。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter_ns

from experiments.exp1_heuristic import choose_batch_loops
from experiments.exp3_minimax_alpha_beta import COUNTERS, _snapshot_sources, _write_csv, _write_json
from experiments.exp4_depths import load_input
from experiments.treasure_reference import reference_distances, reference_game
from experiments.treasure_gain import evaluate_gain_treasure
from src.problem4_treasure import build_distance_matrix, evaluate_treasure


ROOT = Path(__file__).resolve().parent.parent
DEPTH = 4
VARIANTS = ("original", "gain")


def _check_result(result, reference, expected=None, *, allow_trace=False):
    if result.best_value != reference.value or result.best_target not in reference.best_targets:
        raise AssertionError("根值或首步与独立完整博弈参考不符")
    stats = result.stats
    if any(type(getattr(stats, field)) is not int or getattr(stats, field) < 0 for field in COUNTERS):
        raise AssertionError("搜索统计必须是非负整数")
    if not (1 <= stats.leaf_evals <= reference.leaf_count
            and stats.leaf_evals <= stats.node_visits <= reference.node_count):
        raise AssertionError("搜索计数超出完整树范围")
    if stats.cutoff_events > stats.pruned_branches or (stats.cutoff_events == 0) != (stats.pruned_branches == 0):
        raise AssertionError("剪枝事件与跳过直接分支数不一致")
    if expected is not None and any(getattr(stats, field) != expected[field] for field in COUNTERS):
        raise AssertionError("重复求解的计数与审计结果不一致")
    if not allow_trace and result.trace:
        raise AssertionError("正式计时不能开启详细轨迹")


def _measure_once(matrix, values, variant):
    started = perf_counter_ns()
    result = evaluate_gain_treasure(matrix, values, DEPTH, ordering=variant,
                                    use_alpha_beta=True, trace=False)
    return result, (perf_counter_ns() - started) / 1_000_000


def _measure_batch(matrix, values, variant, loops, reference, expected):
    """外层计时覆盖输入校验、搜索、排序及结果收集；核验位于计时后。"""
    if variant not in VARIANTS or type(loops) is not int or loops < 1:
        raise ValueError("未知策略或批内调用次数无效")
    started = perf_counter_ns()
    runs = [_measure_once(matrix, values, variant) for _ in range(loops)]
    elapsed = (perf_counter_ns() - started) / 1_000_000
    for result, _ in runs:
        _check_result(result, reference, expected)
    first = runs[0][0]
    return {"best_target": first.best_target, "best_value": first.best_value,
            "reference_value": reference.value, "correct": True,
            **{field: getattr(first.stats, field) for field in COUNTERS},
            "search_ms_mean": sum(result.stats.elapsed_ms for result, _ in runs) / loops,
            "query_ms_mean": sum(query for _, query in runs) / loops,
            "total_ms_mean": elapsed / loops, "batch_total_ms": elapsed}


def run_experiment(output_dir, *, input_path=None, batches=10, warmup=2, max_loops=128,
                   target_batch_ms=25, max_batch_ms=200, analyze=True):
    """在同一个官方案例上配对比较原顺序和即时收益降序。"""
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
    if len(values) != 4:
        raise ValueError("本实验固定四个宝藏和 H=4")
    independent = reference_distances(case["grid"], case["positions"])
    reference = reference_game(independent, values, DEPTH)

    out.mkdir(parents=True, exist_ok=True)
    (out / "input").mkdir()
    (out / "input/problem4.json").write_bytes(payload)
    source_hashes, archive_hash = _snapshot_sources(out)
    run_id = "p4_gain_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    config = {"experiment": "p4_gain_v1", "run_id": run_id, "status": "running",
              "analysis_status": "pending" if analyze else "skipped",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "source_input": str(source),
              "input_file": "input/problem4.json", "input_sha256": sha256(payload).hexdigest(),
              "depth": DEPTH, "depth_unit": "one player action", "variants": list(VARIANTS),
              "reference_value": reference.value, "reference_best_targets": reference.best_targets,
              "original_order": list(values), "batches": batches, "warmup": warmup,
              "calibration_calls": 2, "target_fast_batch_ms": target_batch_ms,
              "max_estimated_slow_batch_ms": max_batch_ms, "max_loops": max_loops,
              "execution_order_rule": "batch_index % 2; 0 means original first",
              "calibration_rule": "one shared loop count from both variants; fastest target and slowest cap",
              "ordering_rule": "at each state sort remaining actions by acting player's gain descending; stable ties use original order",
              "time_scope": {"search_ms": "common recursive search including gain calculation and dynamic sorting, excludes input validation",
                             "query_ms": "game input validation, matrix/value copying and recursive search including ordering",
                             "total_ms": "outer batch elapsed divided by loops, includes wrappers and result collection"},
              "excluded": "A* matrix construction, independent reference, proof/trace, result checks, I/O, plotting",
              "statistics_unit": "one map; batches are repeated timing measurements, not independent maps",
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

        proof, expected, traces = [], {}, {}
        original_result = evaluate_treasure(matrix, values, DEPTH, use_alpha_beta=True, trace=False)
        for variant in (*VARIANTS, "production"):
            if variant == "production":
                result = original_result
            else:
                result = evaluate_gain_treasure(matrix, values, DEPTH, ordering=variant,
                                                use_alpha_beta=True, trace=False)
            _check_result(result, reference)
            row = {"variant": variant, "depth": DEPTH, "best_target": result.best_target,
                   "best_value": result.best_value, "reference_value": reference.value,
                   "correct": True, **{field: getattr(result.stats, field) for field in COUNTERS}}
            proof.append(row)
            expected[variant] = row
            if variant != "production":
                detailed = evaluate_gain_treasure(matrix, values, DEPTH, ordering=variant,
                                                  use_alpha_beta=True, trace=True)
                _check_result(detailed, reference, row, allow_trace=True)
                if (detailed.best_target, detailed.best_value) != (result.best_target, result.best_value):
                    raise AssertionError("轨迹审计改变了根决策")
                traces[variant] = asdict(detailed)
        _check_result(original_result, reference, expected["original"])
        if (original_result.best_value, original_result.best_target) != (
                proof[0]["best_value"], proof[0]["best_target"]):
            raise AssertionError("原顺序适配器与现有求解器不一致")
        _write_csv(out / "proof.csv", proof)
        _write_json(out / "search_traces.json", traces)
        print("Distance matrix and both search variants audited against independent reference", flush=True)

        estimates, calibration = {}, []
        for variant in VARIANTS:
            for _ in range(warmup):
                result, _ = _measure_once(matrix, values, variant)
                _check_result(result, reference, expected[variant])
            samples = []
            for _ in range(2):
                started = perf_counter_ns()
                result, _ = _measure_once(matrix, values, variant)
                samples.append((perf_counter_ns() - started) / 1_000_000)
                _check_result(result, reference, expected[variant])
            estimates[variant] = median(samples)
            calibration.append({"variant": variant, "estimate_ms": estimates[variant]})
        loops = choose_batch_loops(estimates, target_batch_ms, max_batch_ms, max_loops)
        _write_csv(out / "calibration.csv", calibration)
        config["batch_loops"] = loops
        _write_json(out / "config.json", config)
        raw = []
        for batch in range(batches):
            ordered = VARIANTS if batch % 2 == 0 else VARIANTS[::-1]
            for position, variant in enumerate(ordered):
                measured = _measure_batch(matrix, values, variant, loops, reference, expected[variant])
                raw.append({"run_id": run_id, "variant": variant, "depth": DEPTH,
                            "batch_index": batch, "batch_loops": loops,
                            "order_in_pair": position, **measured})
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
        from experiments.analyze_treasure_gain import analyze_results
        analyze_results(out)
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="题四：原顺序与即时收益动态排序对照")
    parser.add_argument("--out", type=Path, default=ROOT / "results/problem4_gain")
    parser.add_argument("--input", type=Path, default=ROOT / "cases/problem4.json")
    parser.add_argument("--batches", type=int, default=10)
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
