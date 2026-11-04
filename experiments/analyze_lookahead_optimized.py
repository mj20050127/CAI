"""一步前瞻的等价实现对照：逐图计时配对、性质审计与图表。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from experiments.analyze_lookahead import (
    COUNTERS, TIMINGS, _aggregate_instances, _load_saved_case,
    _read_audits as _read_base_audits, _verify_sources as _verify_base_sources, _write_csv,
)
from experiments.analyze_weighted_heuristics import distribution, percentile, _save, _setup_plotting


ROOT = Path(__file__).resolve().parent.parent
HEURISTICS = ("corrected_manhattan", "lookahead_1", "lookahead_1_optimized")
BASELINE, ORIGINAL, OPTIMIZED = HEURISTICS
COLORS = {BASELINE: "#315d7a", ORIGINAL: "#a57b3a", OPTIMIZED: "#21847b"}
LABELS = {BASELINE: "Corrected\nManhattan", ORIGINAL: "Original\none-step", OPTIMIZED: "Optimized\none-step"}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    integer_fields = ("size", "batch_index", "batch_loops", "order_in_triplet", "path_cost", "reference_cost", "optimal_gap", *COUNTERS)
    required = {"run_id", "instance_id", "map_file", "map_sha256", "source", "seed", "heuristic", "correct", *integer_fields,
                *(f"{field}_mean" for field in TIMINGS)}
    rows, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("原始数据字段不完整")
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for field in integer_fields:
                row[field] = int(row[field])
            for field in TIMINGS:
                key = f"{field}_mean"
                row[key] = float(row[key])
                if not math.isfinite(row[key]) or row[key] <= 0:
                    raise ValueError("计时必须为有限正数")
            row["correct"] = str(row["correct"]).lower() in {"true", "1"}
            if row["heuristic"] not in HEURISTICS or row["source"] not in {"random", "official"}:
                raise ValueError("未知策略或地图来源")
            if not row["correct"] or row["optimal_gap"] != 0 or row["path_cost"] != row["reference_cost"]:
                raise ValueError("待汇总结果未通过独立最优代价校验")
            if row["size"] < 1 or row["batch_loops"] < 1 or row["batch_index"] < 0 or row["order_in_triplet"] not in (0, 1, 2):
                raise ValueError("地图规模、计时批次或三策略执行顺序无效")
            if any(row[field] < 0 for field in COUNTERS):
                raise ValueError("节点统计不能为负")
            key = row["instance_id"], row["heuristic"], row["batch_index"]
            if key in seen:
                raise ValueError("重复的实例、策略、批次记录")
            seen.add(key)
            rows.append(row)
    if not rows or len({row["run_id"] for row in rows}) != 1:
        raise ValueError("数据为空或混合了多个运行编号")
    return rows


def _paired_instances(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in instances:
        grouped[row["instance_id"]][row["heuristic"]] = row
    paired = []
    for instance_id, strategies in sorted(grouped.items()):
        if set(strategies) != set(HEURISTICS):
            raise ValueError("每张图必须具有完整三策略配对")
        base, old, fast = (strategies[name] for name in HEURISTICS)
        stable = ("source", "size", "map_file", "map_sha256", "seed", "reference_cost")
        if any(other[field] != base[field] for other in strategies.values() for field in stable):
            raise ValueError("三策略没有使用相同输入与独立参考代价")
        if any(old[field] != fast[field] for field in ("path_cost", *COUNTERS)):
            raise ValueError("优化前后搜索计数或代价不等价，不能作为纯实现优化")
        entry = {"instance_id": instance_id, **{field: base[field] for field in stable}}
        for name, current in strategies.items():
            entry.update({f"{name}_{field}": current[field] for field in ("path_cost", *COUNTERS)})
            entry.update({f"{name}_{prefix}": current[f"{prefix}_median"] for prefix in TIMINGS})
            if name == BASELINE:
                continue
            for field in ("valid_pops", "expansions", "heap_peak"):
                entry[f"{name}_{field}_reduction"] = 1 - current[field] / base[field] if base[field] else 0.0
            for prefix in TIMINGS:
                entry[f"{name}_{prefix.removesuffix('_ms')}_speedup"] = base[f"{prefix}_median"] / current[f"{prefix}_median"]
        for prefix in TIMINGS:
            short = prefix.removesuffix("_ms")
            entry[f"optimized_vs_original_{short}_speedup"] = old[f"{prefix}_median"] / fast[f"{prefix}_median"]
            entry[f"optimized_vs_original_{short}_time_reduction"] = 1 - fast[f"{prefix}_median"] / old[f"{prefix}_median"]
        for field in ("path_cost", *COUNTERS):
            entry[f"optimized_minus_original_{field}"] = fast[field] - old[field]
        paired.append(entry)
    return paired


def _read_audits(path: Path, instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audits = _read_base_audits(path, instances)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "same_formula_violations" not in set(reader.fieldnames or []):
            raise ValueError("缺少逐状态启发等价审计")
        equality = {(row["instance_id"], row["heuristic"]): int(row["same_formula_violations"]) for row in reader}
    for row in audits:
        value = equality[row["instance_id"], row["heuristic"]]
        if value < 0 or value > row["n_states"]:
            raise ValueError("启发等价审计计数越界")
        if value:
            raise ValueError("优化前后启发公式不等价")
        row["same_formula_violations"] = value
    return audits


def _read_equivalence(path: Path, paired: list[dict[str, Any]], audits: list[dict[str, Any]]):
    expected = {row["instance_id"]: row for row in paired}
    audit_counts = {row["instance_id"]: row["n_states"] for row in audits if row["heuristic"] == OPTIMIZED}
    flags = ("cost_equal", "path_equal", "expanded_order_equal", "counters_equal", "baseline_equal", "old_implementation_equal")
    records, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"instance_id", "source", "size", "map_sha256", "n_states", "heuristic_value_mismatches", *flags} <= set(reader.fieldnames or []):
            raise ValueError("等价核验字段不完整")
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            key = row["instance_id"]
            if key not in expected or key in seen:
                raise ValueError("等价核验含重复或未知地图")
            for field in ("size", "n_states", "heuristic_value_mismatches"):
                row[field] = int(row[field])
            if any(row[field] != expected[key][field] for field in ("source", "size", "map_sha256")):
                raise ValueError("等价核验与测量输入不一致")
            if row["n_states"] != audit_counts[key] or row["n_states"] < 1 or row["heuristic_value_mismatches"] != 0:
                raise ValueError("逐格启发值不等价或状态覆盖不符")
            for field in flags:
                row[field] = str(row[field]).lower() in {"true", "1"}
                if not row[field]:
                    raise ValueError(f"等价核验未通过：{field}")
            seen.add(key)
            records.append(row)
    if seen != set(expected):
        raise ValueError("等价核验未覆盖全部地图")
    return records


def _group_summary(instances, paired, audits, rows):
    summary_rows, groups, audit_groups = [], [], []
    for source, size in sorted({(row["source"], row["size"]) for row in instances}):
        pairs = [row for row in paired if (row["source"], row["size"]) == (source, size)]
        group = {"source": source, "size": size, "n_instances": len(pairs), "paired": {}}
        audit_group = {"source": source, "size": size, "n_instances": len(pairs)}
        for name in HEURISTICS:
            selected = [row for row in instances if (row["source"], row["size"], row["heuristic"]) == (source, size, name)]
            stats = {metric: distribution([row[f"{metric}_median"] if metric in TIMINGS else row[metric] for row in selected])
                     for metric in (*COUNTERS, "path_cost", "optimal_gap", *TIMINGS)}
            group[name] = {"n_instances": len(selected), **stats}
            flat = {"source": source, "size": size, "heuristic": name, "n_instances": len(selected)}
            for metric, values in stats.items():
                flat.update({f"{metric}_{part}": values[part] for part in ("median", "q1", "q3", "mean", "std")})
            summary_rows.append(flat)
            if name != BASELINE:
                group["paired"][name] = {metric: distribution([row[f"{name}_{metric}"] for row in pairs])
                                          for metric in ("valid_pops_reduction", "expansions_reduction", "heap_peak_reduction", "prepare_speedup", "search_speedup", "api_speedup")}
                group["paired"][name]["faster_api_maps"] = sum(row[f"{name}_api_ms"] < row[f"{BASELINE}_api_ms"] for row in pairs)
                group["paired"][name]["same_api_maps"] = sum(row[f"{name}_api_ms"] == row[f"{BASELINE}_api_ms"] for row in pairs)
                group["paired"][name]["slower_api_maps"] = sum(row[f"{name}_api_ms"] > row[f"{BASELINE}_api_ms"] for row in pairs)
                group["paired"][name]["fewer_valid_pops_maps"] = sum(row[f"{name}_valid_pops"] < row[f"{BASELINE}_valid_pops"] for row in pairs)
            selected_audits = [row for row in audits if (row["source"], row["size"], row["heuristic"]) == (source, size, name)]
            totals = ("n_states", "n_reachable_states", "n_directed_edges", "admissibility_violations", "consistency_violations", "dominance_violations", "strictly_stronger_states", "same_formula_violations")
            audit_group[name] = {f"{field}_total": sum(row[field] for row in selected_audits) for field in totals}
            audit_group[name].update(n_instances=len(selected_audits), max_overestimate=max(row["max_overestimate"] for row in selected_audits),
                                     maps_with_nonzero_goal_h=sum(row["goal_h"] != 0 for row in selected_audits))
        equivalent = {metric: distribution([row[f"optimized_vs_original_{metric}"] for row in pairs])
                      for metric in ("prepare_speedup", "search_speedup", "api_speedup", "prepare_time_reduction", "search_time_reduction", "api_time_reduction")}
        equivalent.update({f"{field}_difference": distribution([row[f"optimized_minus_original_{field}"] for row in pairs])
                           for field in ("path_cost", *COUNTERS)})
        equivalent.update(
            same_counts_maps=sum(all(row[f"optimized_minus_original_{field}"] == 0 for field in COUNTERS) for row in pairs),
            faster_api_maps=sum(row[f"{OPTIMIZED}_api_ms"] < row[f"{ORIGINAL}_api_ms"] for row in pairs),
            same_api_maps=sum(row[f"{OPTIMIZED}_api_ms"] == row[f"{ORIGINAL}_api_ms"] for row in pairs),
            slower_api_maps=sum(row[f"{OPTIMIZED}_api_ms"] > row[f"{ORIGINAL}_api_ms"] for row in pairs),
        )
        group["optimized_vs_original"] = equivalent
        groups.append(group)
        audit_groups.append(audit_group)
    return summary_rows, {
        "aggregation": {
            "within_map": "Median across batch per-call means; deterministic counters counted once per map.",
            "across_maps": "Independent maps; official excluded from random groups. IQR is dispersion, not a confidence interval.",
            "paired": "Effects within maps first: h0/new for baseline comparisons, original/optimized for implementation speedup. Then aggregate.",
            "time_scope": "Fresh query preparation and heuristic construction are charged; heuristic evaluation/cache work is inside search. API includes both plus collection and release.",
            "memory_scope": "Physical heap entries only; heuristic caches are excluded.",
        },
        "overall": {"n_unique_maps": len(paired), "n_random_maps": sum(row["source"] == "random" for row in paired),
                    "n_official_maps": sum(row["source"] == "official" for row in paired), "n_raw_batches": len(rows),
                    "n_strategy_results": len(instances), "all_correct": all(row["correct"] for row in rows),
                    "n_suboptimal_strategy_results": sum(row["optimal_gap"] != 0 for row in instances),
                    "all_optimized_counts_equal_original": all(all(row[f"optimized_minus_original_{field}"] == 0 for field in COUNTERS) for row in paired)},
        "groups": groups, "heuristic_audit": audit_groups,
    }


def _verify_sources(config):
    _verify_base_sources(config)
    if "experiments/optimized_lookahead.py" not in config.get("source_sha256", {}):
        raise ValueError("缺少优化启发实现的源码哈希")


def _distributions_plot(plt, instances, directory, metrics, name, title, note):
    selected = [row for row in instances if row["source"] == "random"]
    figure, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 5.0), squeeze=False)
    for axis, (metric, ylabel) in zip(axes[0], metrics):
        for index, name_key in enumerate(HEURISTICS):
            values = [row[metric] for row in selected if row["heuristic"] == name_key]
            box = axis.boxplot(values, positions=[index], widths=0.43, patch_artist=True, showfliers=False, manage_ticks=False)
            box["boxes"][0].set(facecolor=COLORS[name_key], alpha=0.23)
            box["medians"][0].set(color=COLORS[name_key], linewidth=2)
            axis.scatter([index + 0.13 * math.sin((i + 1) * 2.39996) for i in range(len(values))], values,
                         color=COLORS[name_key], alpha=0.55, s=20, edgecolors="none")
        axis.set_xticks(range(3), [LABELS[h] for h in HEURISTICS])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.17)
    figure.suptitle(title, fontsize=14, weight="bold")
    n = len({row["instance_id"] for row in selected})
    figure.text(0.5, 0.015, f"One dot per random map (n={n}); box Q1-Q3, line median. {note}", ha="center", fontsize=9, color="#465264")
    figure.tight_layout(rect=(0, 0.10, 1, 0.93))
    _save(figure, directory, name)
    plt.close(figure)


def _paired_plot(plt, paired, directory):
    selected = [row for row in paired if row["source"] == "random"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    comparisons = (
        ("optimized_vs_original_api_speedup", "Original / optimized API", "Implementation optimization"),
        (f"{OPTIMIZED}_api_speedup", "Corrected Manhattan / optimized API", "Overall result vs baseline"),
    )
    for axis, (metric, ylabel, title) in zip(axes, comparisons):
        values = [row[metric] for row in selected]
        axis.scatter([0.17 * math.sin((i + 1) * 2.39996) for i in range(len(values))], values,
                     color=COLORS[OPTIMIZED], alpha=0.60, s=28, edgecolors="none")
        center, q1, q3 = median(values), percentile(values, 0.25), percentile(values, 0.75)
        axis.errorbar(0, center, yerr=[[center - q1], [q3 - center]], fmt="D", color="#1b3945", capsize=5, zorder=4)
        axis.set_xticks([0], ["Optimized one-step"])
        axis.set_xlim(-0.6, 0.6)
        axis.axhline(1, color="#818890", linestyle="--", linewidth=1)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.16)
    figure.suptitle("Paired total-time effects of equivalent one-step implementations", fontsize=14, weight="bold")
    figure.text(0.5, 0.015, "Ratios within maps first; diamond/error bars: median and Q1-Q3. Speedup > 1 is faster. Preparation included.",
                ha="center", fontsize=9, color="#465264")
    figure.tight_layout(rect=(0, 0.10, 1, 0.93))
    _save(figure, directory, "03_paired_speedups")
    plt.close(figure)


def _trace_plot(plt, result_dir, strategies, directory, name, title):
    from experiments.optimized_lookahead import search_variant
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle
    case = _load_saved_case(result_dir, strategies[BASELINE])
    grid, start, goal = case["grid"], tuple(case["start"]), tuple(case["end"])
    background = [[float("nan") if cell == "X" else 0 if (r, c) == goal else 1 if cell in ("S", "E") else int(cell)
                   for c, cell in enumerate(row)] for r, row in enumerate(grid)]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    vmax = max((value for row in background for value in row if math.isfinite(value)), default=1)
    traces = {}
    for axis, key in zip(axes, HEURISTICS):
        result = search_variant(start, goal, grid, heuristic=key, trace=True).result
        traces[key] = result
        expected = strategies[key]
        if result.cost != expected["path_cost"] or any(getattr(result.stats, field) != expected[field] for field in COUNTERS):
            raise ValueError("轨迹计数与本轮实测不符")
        axis.imshow(background, cmap="Greys", vmin=0, vmax=max(1, vmax), alpha=0.32)
        for row, col in set(result.expanded_order):
            axis.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor=to_rgba(COLORS[key], 0.4), edgecolor="none"))
        axis.plot([p[1] for p in result.path], [p[0] for p in result.path], color="#a8323b", linewidth=1.6, alpha=0.85)
        for point, label in ((start, "S"), (goal, "E")):
            axis.text(point[1], point[0], label, ha="center", va="center", fontsize=9, weight="bold", color="#142d3c",
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.1})
        if len(grid) <= 7:
            for r, row in enumerate(background):
                for c, value in enumerate(row):
                    if (r, c) not in (start, goal):
                        axis.text(c, r, "X" if math.isnan(value) else str(int(value)), ha="center", va="center", fontsize=10)
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
        axis.set_title(f"{LABELS[key].replace(chr(10), ' ')}\nValid pops {result.stats.valid_pops}; cost {result.cost}")
    old, fast = traces[ORIGINAL], traces[OPTIMIZED]
    if old.path != fast.path or old.expanded_order != fast.expanded_order:
        raise ValueError("优化前后路径或扩展顺序不相同")
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.text(0.5, 0.025, f"{case['instance_id']}. Gray: terrain; tinted: expanded cells; red: optimal path. Original and optimized traces are identical.",
                ha="center", fontsize=9, color="#465264")
    figure.tight_layout(rect=(0, 0.09, 1, 0.91))
    _save(figure, directory, name)
    plt.close(figure)


def analyze_results(result_dir: Path) -> dict[str, Any]:
    result_dir = Path(result_dir).resolve()
    config = json.loads((result_dir / "config.json").read_text(encoding="utf-8-sig"))
    if config.get("status") != "completed":
        raise ValueError("仅可分析已完成的测量")
    _verify_sources(config)
    rows = _read_rows(result_dir / "raw.csv")
    if rows[0]["run_id"] != config.get("run_id") or len(rows) != config.get("raw_rows"):
        raise ValueError("原始运行编号或行数与配置不一致")
    batches = config["batches"]
    if type(batches) is not int or batches < 1:
        raise ValueError("配置批次数无效")
    instances = _aggregate_instances(rows)
    for instance in instances:
        actual = {row["batch_index"] for row in rows if row["instance_id"] == instance["instance_id"] and row["heuristic"] == instance["heuristic"]}
        if actual != set(range(batches)):
            raise ValueError("某实例的测量批次不完整")
    paired = _paired_instances(instances)
    if len(paired) != config.get("measured_instances"):
        raise ValueError("实例数与配置不符")
    for pair in paired:
        _load_saved_case(result_dir, pair)
        for batch in range(batches):
            triplet = [row for row in rows if row["instance_id"] == pair["instance_id"] and row["batch_index"] == batch]
            if len(triplet) != 3 or len({row["batch_loops"] for row in triplet}) != 1 or {row["order_in_triplet"] for row in triplet} != {0, 1, 2}:
                raise ValueError("三策略批次的循环次数或执行顺序不匹配")
    audits = _read_audits(result_dir / "heuristic_audit.csv", instances)
    equivalence = _read_equivalence(result_dir / "equivalence.csv", paired, audits)
    summary_rows, report = _group_summary(instances, paired, audits, rows)
    report.update(run_id=rows[0]["run_id"], config_file="config.json", dataset_dir=config.get("dataset_dir"), baseline=BASELINE,
                  implementation_comparison={"original": ORIGINAL, "optimized": OPTIMIZED})
    report["equivalence"] = {"n_maps": len(equivalence), "n_states_total": sum(row["n_states"] for row in equivalence),
                             "heuristic_value_mismatches": sum(row["heuristic_value_mismatches"] for row in equivalence),
                             "all_paths_equal": all(row["path_equal"] for row in equivalence),
                             "all_expanded_orders_equal": all(row["expanded_order_equal"] for row in equivalence),
                             "all_counters_equal": all(row["counters_equal"] for row in equivalence),
                             "all_previous_implementations_equal": all(row["baseline_equal"] and row["old_implementation_equal"] for row in equivalence)}
    report["heuristic_strength_reference"] = "Both one-step variants are compared with h0 for dominance and strict improvement."
    _write_csv(result_dir / "per_instance.csv", instances)
    _write_csv(result_dir / "paired.csv", paired)
    _write_csv(result_dir / "summary.csv", summary_rows)
    directory = result_dir / "figures"
    directory.mkdir(exist_ok=True)
    plt = _setup_plotting()
    _distributions_plot(plt, instances, directory,
                        [("valid_pops", "Valid pops (including goal)"), ("heap_peak", "Physical heap peak (entries)")],
                        "01_workload", "Equivalent one-step implementations preserve search work", "Original and optimized counts match; heap excludes caches.")
    _distributions_plot(plt, instances, directory,
                        [("prepare_ms_median", "Preparation per call (ms)"), ("search_ms_median", "Search including heuristic (ms)"), ("api_ms_median", "Complete API per call (ms)")],
                        "02_runtime", "One-step implementation costs with all preparation charged", "Fresh query every call; official excluded.")
    _paired_plot(plt, paired, directory)
    bases = [row for row in instances if row["source"] == "random" and row["heuristic"] == BASELINE]
    if bases:
        center = median(row["valid_pops"] for row in bases)
        representative = min(bases, key=lambda row: (abs(row["valid_pops"] - center), row["instance_id"]))
        strategies = {row["heuristic"]: row for row in instances if row["instance_id"] == representative["instance_id"]}
        report["representative_trace"] = {"instance_id": representative["instance_id"], "map_sha256": representative["map_sha256"],
                                            "selection_rule": "Baseline valid pops closest to random-map median; ties by instance_id.",
                                            "figure": "figures/04_representative_search.png"}
        _trace_plot(plt, result_dir, strategies, directory, "04_representative_search", "Representative random map: unchanged one-step search")
    official_ids = sorted({row["instance_id"] for row in instances if row["source"] == "official"})
    if official_ids:
        official = {row["heuristic"]: row for row in instances if row["instance_id"] == official_ids[0]}
        _trace_plot(plt, result_dir, official, directory, "05_official_search", "Official map: equivalent one-step implementations")
    (result_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    config.update(analysis_status="completed", analysis_completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  analysis_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (result_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="一步前瞻等价实现的配对统计与图表")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "problem2_lookahead_optimized")
    args = parser.parse_args(argv)
    try:
        result = analyze_results(args.results)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f"分析失败：{error}\n")
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
