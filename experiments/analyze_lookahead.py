"""题二前瞻修正的逐图配对统计、性质审计和可复核图表。"""

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

from experiments.analyze_weighted_heuristics import distribution, percentile, _save, _setup_plotting
from experiments.weighted_dataset import input_sha256


ROOT = Path(__file__).resolve().parent.parent
HEURISTICS = ("corrected_manhattan", "lookahead_1", "lookahead_2")
DEPTHS = dict(zip(HEURISTICS, (0, 1, 2)))
BASELINE = HEURISTICS[0]
COUNTERS = ("valid_pops", "expansions", "generated_successors", "pushes", "stale_pops", "heap_peak")
TIMINGS = ("prepare_ms", "search_ms", "api_ms")
AUDIT_FIELDS = (
    "n_states", "n_reachable_states", "n_directed_edges", "admissibility_violations",
    "consistency_violations", "max_overestimate", "goal_h", "dominance_violations", "strictly_stronger_states",
)
COLORS = {BASELINE: "#315d7a", "lookahead_1": "#21847b", "lookahead_2": "#c87932"}
LABELS = {BASELINE: "Corrected\nManhattan", "lookahead_1": "One-step\nlookahead", "lookahead_2": "Two-step\nlookahead"}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("不能保存空的汇总表")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def _aggregate_instances(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["instance_id"], row["heuristic"]].append(row)
    instances = []
    stable = ("source", "size", "map_file", "map_sha256", "seed", "path_cost", "reference_cost", "optimal_gap", *COUNTERS)
    for (instance_id, heuristic), batches in sorted(grouped.items()):
        first = batches[0]
        if any(other[field] != first[field] for other in batches for field in stable):
            raise ValueError("同图同策略的输入、结果或确定性计数发生变化")
        entry = {"instance_id": instance_id, "heuristic": heuristic, **{field: first[field] for field in stable},
                 "n_batches": len(batches), "batch_loops_min": min(row["batch_loops"] for row in batches),
                 "batch_loops_max": max(row["batch_loops"] for row in batches), "correct": True}
        for prefix in TIMINGS:
            stats = distribution([row[f"{prefix}_mean"] for row in batches])
            entry.update({f"{prefix}_{part}": stats[part] for part in ("median", "q1", "q3")})
        instances.append(entry)
    return instances


def _paired_instances(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in instances:
        grouped[row["instance_id"]][row["heuristic"]] = row
    paired = []
    for instance_id, strategies in sorted(grouped.items()):
        if set(strategies) != set(HEURISTICS):
            raise ValueError("每张图必须具有完整三策略配对")
        base = strategies[BASELINE]
        stable = ("source", "size", "map_file", "map_sha256", "seed", "reference_cost")
        if any(other[field] != base[field] for other in strategies.values() for field in stable):
            raise ValueError("三策略没有使用相同输入与独立参考代价")
        entry = {"instance_id": instance_id, **{field: base[field] for field in stable}}
        for heuristic, current in strategies.items():
            entry.update({f"{heuristic}_{field}": current[field] for field in ("path_cost", *COUNTERS)})
            entry.update({f"{heuristic}_{prefix}": current[f"{prefix}_median"] for prefix in TIMINGS})
            if heuristic == BASELINE:
                continue
            for field in ("valid_pops", "expansions", "heap_peak"):
                entry[f"{heuristic}_{field}_reduction"] = 1 - current[field] / base[field] if base[field] else 0.0
            for prefix in TIMINGS:
                entry[f"{heuristic}_{prefix.removesuffix('_ms')}_speedup"] = base[f"{prefix}_median"] / current[f"{prefix}_median"]
        paired.append(entry)
    return paired


def _read_audits(path: Path, instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {(row["instance_id"], row["heuristic"]): row for row in instances}
    audits, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"instance_id", "source", "size", "heuristic", *AUDIT_FIELDS} <= set(reader.fieldnames or []):
            raise ValueError("启发性质审计字段不完整")
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for field in ("size", *AUDIT_FIELDS):
                row[field] = int(row[field])
            key = row["instance_id"], row["heuristic"]
            if key not in expected or key in seen:
                raise ValueError("启发审计缺少对应测量或含重复记录")
            if any(row[field] != expected[key][field] for field in ("source", "size")):
                raise ValueError("启发审计与测量元数据不一致")
            if any(row[field] < 0 for field in AUDIT_FIELDS) or not 1 <= row["n_reachable_states"] <= row["n_states"]:
                raise ValueError("启发性质审计计数无效")
            if (row["admissibility_violations"] > row["n_reachable_states"]
                    or row["consistency_violations"] > row["n_directed_edges"]
                    or row["dominance_violations"] > row["n_states"]
                    or row["strictly_stronger_states"] > row["n_states"]):
                raise ValueError("启发性质审计计数越界")
            if bool(row["admissibility_violations"]) != bool(row["max_overestimate"]):
                raise ValueError("高估状态数与最大高估值不一致")
            seen.add(key)
            audits.append(row)
    if seen != set(expected):
        raise ValueError("启发性质审计未覆盖全部实例与策略")
    return audits


def _group_summary(instances, paired, audits, rows):
    summary_rows, groups, audit_groups = [], [], []
    for source, size in sorted({(row["source"], row["size"]) for row in instances}):
        pairs = [row for row in paired if (row["source"], row["size"]) == (source, size)]
        group = {"source": source, "size": size, "n_instances": len(pairs), "paired": {}}
        audit_group = {"source": source, "size": size, "n_instances": len(pairs)}
        for heuristic in HEURISTICS:
            selected = [row for row in instances if (row["source"], row["size"], row["heuristic"]) == (source, size, heuristic)]
            stats = {metric: distribution([row[f"{metric}_median"] if metric in TIMINGS else row[metric] for row in selected])
                     for metric in (*COUNTERS, "path_cost", "optimal_gap", *TIMINGS)}
            group[heuristic] = {"n_instances": len(selected), **stats}
            summary_row = {"source": source, "size": size, "heuristic": heuristic, "n_instances": len(selected)}
            for metric, values in stats.items():
                summary_row.update({f"{metric}_{part}": values[part] for part in ("median", "q1", "q3", "mean", "std")})
            summary_rows.append(summary_row)
            if heuristic != BASELINE:
                pair_stats = {metric: distribution([row[f"{heuristic}_{metric}"] for row in pairs])
                              for metric in ("valid_pops_reduction", "expansions_reduction", "heap_peak_reduction", "prepare_speedup", "search_speedup", "api_speedup")}
                for relation, operator in (("fewer", lambda x, y: x < y), ("same", lambda x, y: x == y), ("more", lambda x, y: x > y)):
                    pair_stats[f"{relation}_valid_pops_maps"] = sum(operator(row[f"{heuristic}_valid_pops"], row[f"{BASELINE}_valid_pops"]) for row in pairs)
                for relation, operator in (("faster", lambda x, y: x < y), ("same", lambda x, y: x == y), ("slower", lambda x, y: x > y)):
                    pair_stats[f"{relation}_api_maps"] = sum(operator(row[f"{heuristic}_api_ms"], row[f"{BASELINE}_api_ms"]) for row in pairs)
                group["paired"][heuristic] = pair_stats
            selected_audits = [row for row in audits if (row["source"], row["size"], row["heuristic"]) == (source, size, heuristic)]
            totals = ("n_states", "n_reachable_states", "n_directed_edges", "admissibility_violations", "consistency_violations", "dominance_violations", "strictly_stronger_states")
            audit_group[heuristic] = {f"{field}_total": sum(row[field] for row in selected_audits) for field in totals}
            audit_group[heuristic].update(
                n_instances=len(selected_audits), max_overestimate=max(row["max_overestimate"] for row in selected_audits),
                maps_with_admissibility_violations=sum(row["admissibility_violations"] > 0 for row in selected_audits),
                maps_with_consistency_violations=sum(row["consistency_violations"] > 0 for row in selected_audits),
                maps_with_dominance_violations=sum(row["dominance_violations"] > 0 for row in selected_audits),
                maps_with_nonzero_goal_h=sum(row["goal_h"] != 0 for row in selected_audits),
            )
        groups.append(group)
        audit_groups.append(audit_group)
    return summary_rows, {
        "aggregation": {
            "within_map": "Median across batch per-call means; deterministic counters counted once per map.",
            "across_maps": "Independent maps; median and linearly interpolated Q1/Q3. Official excluded from random groups.",
            "paired": "Per-map reduction=1-new/baseline; speedup=baseline/new. Then aggregate, not ratios of group medians.",
            "time_scope": "Preparation includes validation, minimum weight and new cache construction. Lazy heuristic computation is inside search. API includes both plus loop/result collection.",
            "memory_scope": "heap_peak counts physical heap entries only; excludes heuristic memoization and other memory.",
            "uncertainty": "IQR describes dispersion across maps, not a confidence interval; repeated batches are not new maps.",
        },
        "overall": {"n_unique_maps": len(paired), "n_random_maps": sum(row["source"] == "random" for row in paired),
                    "n_official_maps": sum(row["source"] == "official" for row in paired), "n_raw_batches": len(rows),
                    "n_strategy_results": len(instances), "all_correct": all(row["correct"] for row in rows),
                    "n_suboptimal_strategy_results": sum(row["optimal_gap"] != 0 for row in instances)},
        "groups": groups, "heuristic_audit": audit_groups,
    }


def _verify_sources(config):
    hashes = config.get("source_sha256", {})
    required = {"src/astar.py", "src/grid.py", "src/heuristics.py", "experiments/lookahead_heuristics.py"}
    if not required <= set(hashes):
        raise ValueError("缺少复核轨迹所需的源码哈希")
    for relative, expected in hashes.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"源码已变化：{relative}；不能混用旧统计与新轨迹")


def _load_saved_case(result_dir: Path, expected):
    path = (result_dir / expected["map_file"]).resolve()
    if not path.is_relative_to(result_dir.resolve()):
        raise ValueError("地图路径越过本轮结果目录")
    case = json.loads(path.read_text(encoding="utf-8-sig"))
    if input_sha256(case) != expected["map_sha256"]:
        raise ValueError("保存地图与测量输入哈希不一致")
    return case


def _distributions_plot(plt, instances, directory, metrics, name, title, note):
    selected = [row for row in instances if row["source"] == "random"]
    figure, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 5.0), squeeze=False)
    for axis, (metric, ylabel) in zip(axes[0], metrics):
        for index, heuristic in enumerate(HEURISTICS):
            values = [row[metric] for row in selected if row["heuristic"] == heuristic]
            box = axis.boxplot(values, positions=[index], widths=0.43, patch_artist=True, showfliers=False, manage_ticks=False)
            box["boxes"][0].set(facecolor=COLORS[heuristic], alpha=0.23)
            box["medians"][0].set(color=COLORS[heuristic], linewidth=2)
            axis.scatter([index + 0.13 * math.sin((i + 1) * 2.39996) for i in range(len(values))], values,
                         color=COLORS[heuristic], alpha=0.55, s=20, edgecolors="none")
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
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))
    for axis, metric, label, multiplier, baseline in (
        (axes[0], "valid_pops_reduction", "Valid-pop reduction vs baseline (%)", 100, 0),
        (axes[1], "api_speedup", "Total API speedup (baseline / lookahead)", 1, 1),
    ):
        for index, heuristic in enumerate(HEURISTICS[1:]):
            values = [row[f"{heuristic}_{metric}"] * multiplier for row in selected]
            axis.scatter([index + 0.13 * math.sin((i + 1) * 2.39996) for i in range(len(values))], values,
                         s=26, color=COLORS[heuristic], alpha=0.55, edgecolors="none")
            center, q1, q3 = median(values), percentile(values, 0.25), percentile(values, 0.75)
            axis.errorbar(index, center, yerr=[[center - q1], [q3 - center]], fmt="D", color="#1b3945", capsize=5, zorder=4)
        axis.set_xticks([0, 1], [LABELS[h] for h in HEURISTICS[1:]])
        axis.set_xlim(-0.5, 1.5)
        axis.axhline(baseline, color="#818890", linestyle="--", linewidth=1)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.16)
    figure.suptitle("Paired effects of terrain-aware lookahead", fontsize=14, weight="bold")
    figure.text(0.5, 0.015, "Baseline: corrected Manhattan. Ratios within maps first; diamond/error bars: median and Q1-Q3. Speedup > 1 is faster.",
                ha="center", fontsize=9, color="#465264")
    figure.tight_layout(rect=(0, 0.10, 1, 0.93))
    _save(figure, directory, "03_paired_effects")
    plt.close(figure)


def _trace_plot(plt, result_dir, strategies, directory, name, title):
    from experiments.lookahead_heuristics import search_lookahead
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle
    case = _load_saved_case(result_dir, strategies[BASELINE])
    grid, start, goal = case["grid"], tuple(case["start"]), tuple(case["end"])
    background = [[float("nan") if cell == "X" else 0 if (r, c) == goal else 1 if cell in ("S", "E") else int(cell)
                   for c, cell in enumerate(row)] for r, row in enumerate(grid)]
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.8))
    vmax = max((value for row in background for value in row if math.isfinite(value)), default=1)
    for axis, heuristic in zip(axes, HEURISTICS):
        result = search_lookahead(start, goal, grid, depth=DEPTHS[heuristic], trace=True).result
        expected = strategies[heuristic]
        if result.cost != expected["path_cost"] or any(getattr(result.stats, field) != expected[field] for field in COUNTERS):
            raise ValueError("轨迹计数与本轮实测不符，不能用新实现解释旧结果")
        axis.imshow(background, cmap="Greys", vmin=0, vmax=max(1, vmax), alpha=0.32)
        for row, col in set(result.expanded_order):
            axis.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor=to_rgba(COLORS[heuristic], 0.40), edgecolor="none"))
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
        axis.set_title(f"{LABELS[heuristic].replace(chr(10), ' ')}\nValid pops {result.stats.valid_pops}; cost {result.cost}")
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.text(0.5, 0.025, f"{case['instance_id']}. Gray: terrain costs; tinted: expanded cells; red: optimal path. Same neighbor and queue rules.",
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
                raise ValueError("三策略批次的输入或循环次数不匹配")
    audits = _read_audits(result_dir / "heuristic_audit.csv", instances)
    summary_rows, report = _group_summary(instances, paired, audits, rows)
    report.update(run_id=rows[0]["run_id"], config_file="config.json", dataset_dir=config.get("dataset_dir"), baseline=BASELINE)
    _write_csv(result_dir / "per_instance.csv", instances)
    _write_csv(result_dir / "paired.csv", paired)
    _write_csv(result_dir / "summary.csv", summary_rows)
    directory = result_dir / "figures"
    directory.mkdir(exist_ok=True)
    plt = _setup_plotting()
    _distributions_plot(plt, instances, directory,
                        [("valid_pops", "Valid pops (including goal)"), ("heap_peak", "Physical heap peak (entries)")],
                        "01_workload", "Lookahead: search work and frontier pressure", "Heap excludes heuristic caches; official excluded.")
    _distributions_plot(plt, instances, directory,
                        [("prepare_ms_median", "Preparation per call (ms)"), ("search_ms_median", "Search incl. lazy heuristic (ms)"), ("api_ms_median", "Total API per call (ms)")],
                        "02_runtime", "Lookahead: all preparation and heuristic costs are charged", "Lazy heuristic work is inside search; API includes both.")
    _paired_plot(plt, paired, directory)
    random_bases = [row for row in instances if row["source"] == "random" and row["heuristic"] == BASELINE]
    if random_bases:
        center = median(row["valid_pops"] for row in random_bases)
        representative = min(random_bases, key=lambda row: (abs(row["valid_pops"] - center), row["instance_id"]))
        strategies = {row["heuristic"]: row for row in instances if row["instance_id"] == representative["instance_id"]}
        report["representative_trace"] = {"instance_id": representative["instance_id"], "map_sha256": representative["map_sha256"],
                                            "selection_rule": "Baseline valid pops closest to the random-map median; ties by instance_id.",
                                            "figure": "figures/04_representative_search.png"}
        _trace_plot(plt, result_dir, strategies, directory, "04_representative_search", "Representative random map: expanded cells and optimal paths")
    official_ids = sorted({row["instance_id"] for row in instances if row["source"] == "official"})
    if official_ids:
        official = {row["heuristic"]: row for row in instances if row["instance_id"] == official_ids[0]}
        _trace_plot(plt, result_dir, official, directory, "05_official_search", "Official weighted map: three corrected heuristics")
    (result_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    config.update(analysis_status="completed", analysis_completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  analysis_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (result_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="题二前瞻修正曼哈顿的配对统计与图表")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "problem2_lookahead")
    args = parser.parse_args(argv)
    try:
        result = analyze_results(args.results)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f"分析失败：{error}\n")
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
