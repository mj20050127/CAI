"""题四第1、2项：官方距离核验与 H=1..4 的双算法、双计时范围对照。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
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
from experiments.treasure_reference import action_ledger, grid_weights, reference_distances, reference_game, validate_path
from src.grid import grid_search
from src.problem4_treasure import TreasureState, build_distance_matrix, evaluate_treasure


ROOT = Path(__file__).resolve().parent.parent
ALGORITHMS = ("minimax", "alpha_beta")
SCOPES = ("precomputed", "end_to_end")
TIMINGS = ("distance_ms", "search_ms", "query_ms", "total_ms")
RAW_FIELDS = ("run_id", "input_file", "input_sha256", "scope", "depth", "algorithm", "batch_index",
              "batch_loops", "order_in_pair", "scope_order_in_batch", "best_target", "best_value",
              "reference_value", "correct", *COUNTERS, *(f"{t}_mean" for t in TIMINGS), "batch_total_ms")


def load_input(path):
    payload = Path(path).read_bytes()
    case = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(case, dict) or not {"grid", "positions", "treasure_values"} <= set(case):
        raise ValueError("输入缺少 grid、positions 或 treasure_values")
    weights = grid_weights(case["grid"])
    values, positions = case["treasure_values"], case["positions"]
    if not isinstance(values, dict) or not 1 <= len(values) <= 8 or not isinstance(positions, dict):
        raise ValueError("本实验要求1至8个宝藏及位置字典")
    if set(positions) != {"A", "B"} | set(values):
        raise ValueError("位置标签必须恰好为 A、B 和所有宝藏")
    points = []
    for name, point in positions.items():
        if (not isinstance(point, list) or len(point) != 2 or any(type(x) is not int for x in point)
                or not 0 <= point[0] < len(weights) or not 0 <= point[1] < len(weights[0])):
            raise ValueError("位置坐标无效")
        if case["grid"][point[0]][point[1]] != name:
            raise ValueError("位置必须指向同名地图标记")
        points.append(tuple(point))
    if len(set(points)) != len(points):
        raise ValueError("各位置必须互异")
    return payload, case


def _check_result(result, reference, algorithm):
    if result.best_value != reference.value or result.best_target not in reference.best_targets:
        raise AssertionError("根值或首步与独立排列参考不一致")
    if algorithm == "minimax" and (result.stats.node_visits != reference.node_count
            or result.stats.leaf_evals != reference.leaf_count
            or result.stats.cutoff_events or result.stats.pruned_branches):
        raise AssertionError("Minimax 未完整访问或发生不应有的剪枝")
    if not (1 <= result.stats.leaf_evals <= reference.leaf_count
            and result.stats.leaf_evals <= result.stats.node_visits <= reference.node_count):
        raise AssertionError("博弈访问计数超出完整树范围")


def audit_case(case, output_dir, depths):
    """全部审计、单首步完整窗口查询和轨迹采集都在计时外。"""
    grid, positions, values = case["grid"], case["positions"], case["treasure_values"]
    matrix = build_distance_matrix(grid, positions)
    independent = reference_distances(grid, positions)
    audits, paths, steps = [], [], []
    weights = grid_weights(grid)
    for source, start in positions.items():
        for target, end in positions.items():
            cost = matrix[source][target]
            if cost != independent[source][target]:
                raise AssertionError("A* 与独立 Dijkstra 矩阵不一致")
            if source == target:
                path, counts = [tuple(start)], {}
            else:
                result = grid_search(tuple(start), tuple(end), grid, weighted=True, goal_free=False)
                path = result.path
                if result.cost != cost:
                    raise AssertionError("用于路径审计的 A* 与矩阵构建不一致")
                counts = asdict(result.stats)
            validate_path(grid, start, end, path, cost)
            audits.append({"source": source, "target": target, "astar_cost": cost,
                           "dijkstra_cost": independent[source][target], "path_cost": cost,
                           "correct": True, "diagonal": source == target})
            if source != target:
                paths.append({"source": source, "target": target, "cost": cost, "path": path, "stats": counts})
                cumulative = 0
                for index, (row, col) in enumerate(path):
                    entering = 0 if index == 0 else weights[row][col]
                    cumulative += entering
                    steps.append({"source": source, "target": target, "step": index, "row": row, "col": col,
                                  "cell": grid[row][col], "entering_cost": entering, "cumulative_cost": cumulative})
    _write_json(output_dir / "distance_matrix.json", matrix)
    _write_json(output_dir / "dijkstra_reference.json", independent)
    _write_csv(output_dir / "distance_audit.csv", audits)
    _write_json(output_dir / "paths.json", paths)
    _write_csv(output_dir / "path_steps.csv", steps)

    full = reference_game(independent, values, len(values))
    references, proof, actions, decisions, ledger, traces, variations = {}, [], [], [], [], {}, {}
    for depth in depths:
        ref = reference_game(independent, values, depth)
        references[depth] = ref
        for target, expected in ref.action_values.items():
            initial = TreasureState(target, "B", frozenset(set(values) - {target}),
                                    values[target] - matrix["A"][target], 0, "B")
            child = evaluate_treasure(matrix, values, depth - 1, use_alpha_beta=False, initial_state=initial)
            if child.best_value != expected:
                raise AssertionError("固定首步的完整 Minimax 值与独立参考不一致")
            actions.append({"depth": depth, "target": target, "exact_value": expected,
                            "minimax_child_value": child.best_value, "optimal_at_depth": target in ref.best_targets,
                            "full_horizon_value": full.action_values[target]})
        for algorithm in ALGORITHMS:
            detailed = evaluate_treasure(matrix, values, depth, use_alpha_beta=algorithm == "alpha_beta", trace=True)
            _check_result(detailed, ref, algorithm)
            proof.append({"depth": depth, "algorithm": algorithm, "best_target": detailed.best_target,
                          "best_value": detailed.best_value, "reference_value": ref.value, "correct": True,
                          **{field: getattr(detailed.stats, field) for field in COUNTERS}})
            traces[f"H{depth}_{algorithm}"] = asdict(detailed)
        chosen = ref.best_targets[0]
        decisions.append({"depth": depth, "chosen_target": chosen, "depth_value": ref.value,
                          "first_move_full_value": full.action_values[chosen], "full_optimal_value": full.value,
                          "first_move_regret": full.value - full.action_values[chosen],
                          "is_complete_game": depth >= len(values)})
        variations[str(depth)] = asdict(ref)
        for row in action_ledger(matrix, values, ref.principal_variation):
            ledger.append({"scenario": "depth_optimal", "depth": depth, **row})
    # 将各浅层首步放回完整博弈，保留完整最优响应路径以解释 regret。
    for target, sequence in full.action_variations.items():
        for row in action_ledger(matrix, values, sequence):
            ledger.append({"scenario": f"full_first_{target}", "depth": len(values), **row})
    _write_csv(output_dir / "proof.csv", proof)
    _write_csv(output_dir / "action_values.csv", actions)
    _write_csv(output_dir / "decisions.csv", decisions)
    _write_csv(output_dir / "action_ledger.csv", ledger)
    _write_json(output_dir / "principal_variations.json", variations)
    _write_json(output_dir / "search_traces.json", traces)
    return matrix, references, proof


@dataclass
class MeasuredRun:
    result: object
    distance_ms: float
    query_ms: float


def measure_once(case, matrix, depth, algorithm, scope):
    distance_ms = 0.0
    if scope == "end_to_end":
        started = perf_counter_ns()
        matrix = build_distance_matrix(case["grid"], case["positions"])
        distance_ms = (perf_counter_ns() - started) / 1_000_000
    started = perf_counter_ns()
    result = evaluate_treasure(matrix, case["treasure_values"], depth,
                               use_alpha_beta=algorithm == "alpha_beta", trace=False)
    return MeasuredRun(result, distance_ms, (perf_counter_ns() - started) / 1_000_000)


def measure_batch(case, matrix, depth, algorithm, scope, loops, ref, expected_counts):
    if scope not in SCOPES or algorithm not in ALGORITHMS or type(loops) is not int or loops < 1:
        raise ValueError("未知计时范围、算法或无效调用次数")
    started = perf_counter_ns()
    runs = [measure_once(case, matrix, depth, algorithm, scope) for _ in range(loops)]
    batch_ms = (perf_counter_ns() - started) / 1_000_000
    first = runs[0].result
    for run in runs:
        _check_result(run.result, ref, algorithm)
        if run.result.trace or any(getattr(run.result.stats, f) != expected_counts[f] for f in COUNTERS):
            raise AssertionError("正式调用的计数或轨迹开关与审计不一致")
    return {"best_target": first.best_target, "best_value": first.best_value, "reference_value": ref.value,
            "correct": True, **{f: getattr(first.stats, f) for f in COUNTERS},
            "distance_ms_mean": sum(r.distance_ms for r in runs) / loops,
            "search_ms_mean": sum(r.result.stats.elapsed_ms for r in runs) / loops,
            "query_ms_mean": sum(r.query_ms for r in runs) / loops,
            "total_ms_mean": batch_ms / loops, "batch_total_ms": batch_ms}


def run_experiment(output_dir, *, input_path=None, depths=(1, 2, 3, 4), batches=5, warmup=2,
                   target_batch_ms=25, max_batch_ms=200, max_loops=128, analyze=True):
    if (not isinstance(depths, (list, tuple)) or not depths or len(set(depths)) != len(depths)
            or any(type(d) is not int or d < 1 for d in depths)):
        raise ValueError("深度必须是互异的正整数列表")
    if any(type(n) is not int or n < 1 for n in (batches, warmup, max_loops)) or type(analyze) is not bool:
        raise ValueError("批次、预热、调用上限或分析开关无效")
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in (target_batch_ms, max_batch_ms)):
        raise ValueError("计时目标必须有限且为正")
    if analyze and importlib.util.find_spec("matplotlib") is None:
        raise ValueError("绘图需要 matplotlib")
    out = Path(output_dir).resolve()
    source = Path(input_path or ROOT / "cases/problem4.json").resolve()
    if source.is_relative_to(out) or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise ValueError("结果目录非空，或输入位于结果目录内；请指定新的空目录")
    payload, case = load_input(source)
    # 完整参考输入校验先于创建输出，坏输入不留下半个结果目录。
    independent = reference_distances(case["grid"], case["positions"])
    reference_game(independent, case["treasure_values"], max(depths))
    out.mkdir(parents=True, exist_ok=True)
    (out / "input").mkdir()
    (out / "input/problem4.json").write_bytes(payload)
    source_hashes, archive_hash = _snapshot_sources(out)
    run_id = "p4_depths_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    depth_order = list(depths)
    random.Random(20261026).shuffle(depth_order)
    config = {"experiment": "problem4_depths_v1", "run_id": run_id, "status": "running",
              "analysis_status": "pending" if analyze else "skipped", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_input": str(source), "input_file": "input/problem4.json", "input_sha256": sha256(payload).hexdigest(),
              "depths": list(depths), "measurement_depth_order": depth_order, "depth_order_seed": 20261026,
              "algorithms": list(ALGORITHMS), "scopes": list(SCOPES), "batches": batches, "warmup": warmup,
              "calibration_calls": 2, "target_fast_batch_ms": target_batch_ms,
              "max_estimated_slow_batch_ms": max_batch_ms, "max_loops": max_loops,
              "calibration_rule": "same depth and scope, same loops for both algorithms; fastest target with slowest duration and loop caps",
              "pair_order_rule": "(index in configured depths * batches + batch_index) % 2; 0 means minimax first",
              "scope_order_rule": "(index in configured depths + batch_index) % 2 reverses scope pair",
              "scope_definitions": {"precomputed": "reuse the same verified A* distance matrix; each call validates and searches the game",
                                    "end_to_end": "fresh 30 directed A* distances on the in-memory official map plus game validation and search; generic case uses n*(n-1) queries"},
              "time_scope": {"distance_ms": "fresh build_distance_matrix call, only end_to_end",
                             "search_ms": "original game search and counters, no trace, excludes input validation",
                             "query_ms": "game validation, matrix/value copying and search",
                             "total_ms": "actual batch wall time/loops including phases, wrapper, loop, collection and release"},
              "excluded": "disk I/O, JSON parsing, independent oracles, result checks, action-value and principal-variation audits, traces, plotting",
              "statistics_unit": "one official input; Q1-Q3 describes five repeated batch means, not independent map samples or confidence intervals",
              "action_order": list(case["treasure_values"]), "depth_unit": "one player action",
              "utility": "score_a-score_b", "gain": "treasure_value-entering_cost_distance",
              "goal_free": False, "sorting": False, "cache": False, "trace": False,
              "garbage_collection": "normal interpreter behavior", "source_sha256": source_hashes,
              "source_baseline_sha256": archive_hash,
              "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}}
    _write_json(out / "config.json", config)
    raw, calibration = [], []
    try:
        matrix, references, proof = audit_case(case, out, depths)
        expected = {(row["depth"], row["algorithm"]): row for row in proof}
        print(f"Distance and game audits passed; depths={list(depths)}, scopes={list(SCOPES)}, batches={batches}", flush=True)
        for depth in depth_order:
            index = list(depths).index(depth)
            loops_by_scope = {}
            for scope in SCOPES:
                estimates = {}
                for algorithm in ALGORITHMS[::1 if index % 2 == 0 else -1]:
                    for _ in range(warmup):
                        measured = measure_once(case, matrix, depth, algorithm, scope)
                        _check_result(measured.result, references[depth], algorithm)
                    samples = []
                    for _ in range(2):
                        started = perf_counter_ns()
                        measured = measure_once(case, matrix, depth, algorithm, scope)
                        samples.append((perf_counter_ns() - started) / 1_000_000)
                        _check_result(measured.result, references[depth], algorithm)
                    estimates[algorithm] = median(samples)
                loops = choose_batch_loops(estimates, target_batch_ms, max_batch_ms, max_loops)
                loops_by_scope[scope] = loops
                calibration.append({"depth": depth, "scope": scope, "batch_loops": loops,
                                    **{f"{a}_total_ms": estimates[a] for a in ALGORITHMS}})
            for batch in range(batches):
                scopes = SCOPES[::1 if (index + batch) % 2 == 0 else -1]
                algorithms = ALGORITHMS[::1 if (index * batches + batch) % 2 == 0 else -1]
                for scope_position, scope in enumerate(scopes):
                    for position, algorithm in enumerate(algorithms):
                        measured = measure_batch(case, matrix, depth, algorithm, scope, loops_by_scope[scope],
                                                 references[depth], expected[depth, algorithm])
                        raw.append({"run_id": run_id, "input_file": config["input_file"], "input_sha256": config["input_sha256"],
                                    "scope": scope, "depth": depth, "algorithm": algorithm, "batch_index": batch,
                                    "batch_loops": loops_by_scope[scope], "order_in_pair": position,
                                    "scope_order_in_batch": scope_position, **measured})
            print(f"Measured H={depth}; rows={len(raw)}", flush=True)
        _write_csv(out / "raw.csv", raw)
        _write_csv(out / "calibration.csv", calibration)
        config.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(), raw_rows=len(raw),
                      total_timed_calls=sum(r["batch_loops"] for r in raw), all_results_correct=True,
                      directed_queries_per_matrix=len(matrix) * (len(matrix) - 1))
        _write_json(out / "config.json", config)
    except Exception as error:
        config.update(status="failed", failure=f"{type(error).__name__}: {error}")
        _write_json(out / "config.json", config)
        raise
    if analyze:
        from experiments.analyze_treasure_depths import analyze_results
        analyze_results(out)
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="题四：距离矩阵验证与深度基础对照")
    parser.add_argument("--out", type=Path, default=ROOT / "results/problem4_depths")
    parser.add_argument("--input", type=Path, default=ROOT / "cases/problem4.json")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-loops", type=int, default=128)
    parser.add_argument("--target-batch-ms", type=float, default=25)
    parser.add_argument("--max-batch-ms", type=float, default=200)
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = run_experiment(args.out, input_path=args.input, depths=args.depths, batches=args.batches,
                                warmup=args.warmup, max_loops=args.max_loops, target_batch_ms=args.target_batch_ms,
                                max_batch_ms=args.max_batch_ms, analyze=not args.no_analysis)
    except (ValueError, OSError) as error:
        parser.exit(2, f"实验配置或输入错误：{error}\n")
    print(f"Completed: {args.out}; rows={config['raw_rows']}; calls={config['total_timed_calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
