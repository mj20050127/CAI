"""题二的三策略配对汇总、启发性质审计和图表；不重新采集计时。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
HEURISTICS = ("zero", "manhattan", "corrected_manhattan")
COUNTERS = ("valid_pops", "expansions", "generated_successors", "pushes", "stale_pops", "heap_peak")
COLORS = {"zero": "#244e76", "manhattan": "#df832e", "corrected_manhattan": "#21847b"}
LABELS = {"zero": "Zero", "manhattan": "Raw Manhattan", "corrected_manhattan": "Corrected Manhattan"}
AUDIT_FIELDS = (
    "n_reachable_states", "admissibility_violations", "consistency_violations", "max_overestimate", "goal_h",
)


def percentile(values: list[float], proportion: float) -> float:
    if not values or not 0 <= proportion <= 1:
        raise ValueError("分位数需要非空数据及合法比例")
    ordered = sorted(values)
    index = (len(ordered) - 1) * proportion
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("分布统计需要非空数据")
    return {
        "median": median(values), "q1": percentile(values, 0.25), "q3": percentile(values, 0.75),
        "mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values), "max": max(values),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("不能保存空的汇总表")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    integer_fields = ("size", "batch_index", "batch_loops", "order_in_triplet", "path_cost", "reference_cost", "optimal_gap", *COUNTERS)
    required = {
        "run_id", "instance_id", "map_file", "map_sha256", "source", "seed", "heuristic", "correct",
        "search_ms_mean", "api_ms_mean", *integer_fields,
    }
    rows, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("原始数据字段不完整")
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for field in integer_fields:
                row[field] = int(row[field])
            for field in ("search_ms_mean", "api_ms_mean"):
                row[field] = float(row[field])
                if not math.isfinite(row[field]) or row[field] <= 0:
                    raise ValueError("计时必须为有限正数")
            row["correct"] = str(row["correct"]).lower() in {"true", "1"}
            if row["heuristic"] not in HEURISTICS or row["source"] not in {"random", "official"}:
                raise ValueError("未知策略或地图来源")
            if not row["correct"] or row["optimal_gap"] != 0 or row["path_cost"] != row["reference_cost"]:
                raise ValueError("待汇总结果未通过独立最优代价校验")
            if row["batch_loops"] < 1 or row["batch_index"] < 0 or row["order_in_triplet"] not in (0, 1, 2):
                raise ValueError("计时批次或三策略执行顺序无效")
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


def _read_audits(path: Path, instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {(row["instance_id"], row["heuristic"]): row for row in instances}
    audits, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"instance_id", "source", "size", "heuristic", *AUDIT_FIELDS}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("启发性质审计字段不完整")
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for field in ("size", "n_reachable_states", "admissibility_violations", "consistency_violations", "goal_h"):
                row[field] = int(row[field])
            row["max_overestimate"] = float(row["max_overestimate"])
            key = row["instance_id"], row["heuristic"]
            if key not in expected or key in seen:
                raise ValueError("启发审计缺少对应测量或含重复记录")
            if any(row[field] != expected[key][field] for field in ("source", "size")):
                raise ValueError("启发审计与测量实例元数据不一致")
            if row["n_reachable_states"] < 1 or any(row[field] < 0 for field in AUDIT_FIELDS):
                raise ValueError("启发性质审计计数无效")
            if not math.isfinite(row["max_overestimate"]) or row["admissibility_violations"] > row["n_reachable_states"]:
                raise ValueError("启发性质审计范围无效")
            if bool(row["admissibility_violations"]) != bool(row["max_overestimate"]):
                raise ValueError("高估状态数与最大高估值不一致")
            seen.add(key)
            audits.append(row)
    if seen != set(expected):
        raise ValueError("启发性质审计未覆盖全部实例与策略")
    return audits


def _aggregate_instances(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["instance_id"], row["heuristic"]].append(row)
    instances = []
    stable = ("source", "size", "map_file", "map_sha256", "seed", "path_cost", "reference_cost", "optimal_gap", *COUNTERS)
    for (instance_id, heuristic), batches in sorted(grouped.items()):
        first = batches[0]
        if any(other[field] != first[field] for other in batches for field in stable):
            raise ValueError("同图同策略的批次输入、结果或节点计数发生变化")
        entry = {
            "instance_id": instance_id, "heuristic": heuristic,
            **{field: first[field] for field in stable}, "n_batches": len(batches),
            "batch_loops_min": min(row["batch_loops"] for row in batches),
            "batch_loops_max": max(row["batch_loops"] for row in batches), "correct": True,
        }
        for source, prefix in (("search_ms_mean", "search_ms"), ("api_ms_mean", "api_ms")):
            stats = distribution([row[source] for row in batches])
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
        zero = strategies["zero"]
        stable = ("source", "size", "map_file", "map_sha256", "seed", "reference_cost")
        if any(other[field] != zero[field] for other in strategies.values() for field in stable):
            raise ValueError("三策略没有使用相同输入与独立参考代价")
        entry = {"instance_id": instance_id, **{field: zero[field] for field in stable}}
        for heuristic in HEURISTICS:
            current = strategies[heuristic]
            entry.update({f"{heuristic}_{field}": current[field] for field in ("path_cost", "valid_pops", "expansions", "heap_peak")})
            for prefix in ("search_ms", "api_ms"):
                entry[f"{heuristic}_{prefix}"] = current[f"{prefix}_median"]
            if heuristic != "zero":
                for field in ("valid_pops", "expansions", "heap_peak"):
                    entry[f"{heuristic}_{field}_reduction"] = 1 - current[field] / zero[field] if zero[field] else 0.0
                entry[f"{heuristic}_search_speedup"] = zero["search_ms_median"] / current["search_ms_median"]
                entry[f"{heuristic}_api_speedup"] = zero["api_ms_median"] / current["api_ms_median"]
        for field in ("path_cost", "valid_pops", "expansions", "heap_peak"):
            entry[f"raw_minus_corrected_{field}"] = strategies["manhattan"][field] - strategies["corrected_manhattan"][field]
        paired.append(entry)
    return paired


def _group_summary(instances, paired, audits, rows):
    summary_rows, groups, audit_groups = [], [], []
    metrics = (*COUNTERS, "path_cost", "optimal_gap", "search_ms", "api_ms")
    for source, size in sorted({(row["source"], row["size"]) for row in instances}):
        selected_pairs = [row for row in paired if (row["source"], row["size"]) == (source, size)]
        group: dict[str, Any] = {"source": source, "size": size, "n_instances": len(selected_pairs)}
        audit_group: dict[str, Any] = {"source": source, "size": size, "n_instances": len(selected_pairs)}
        group["paired"] = {}
        for heuristic in HEURISTICS:
            selected = [row for row in instances if (row["source"], row["size"], row["heuristic"]) == (source, size, heuristic)]
            stats = {
                metric: distribution([row[f"{metric}_median"] if metric.endswith("_ms") else row[metric] for row in selected])
                for metric in metrics
            }
            group[heuristic] = {"n_instances": len(selected), **stats}
            summary_row = {"source": source, "size": size, "heuristic": heuristic, "n_instances": len(selected)}
            for metric, values in stats.items():
                summary_row.update({f"{metric}_{part}": values[part] for part in ("median", "q1", "q3", "mean", "std")})
            summary_rows.append(summary_row)
            if heuristic != "zero":
                pair_metrics = ("valid_pops_reduction", "expansions_reduction", "heap_peak_reduction", "search_speedup", "api_speedup")
                group["paired"][heuristic] = {
                    metric: distribution([row[f"{heuristic}_{metric}"] for row in selected_pairs]) for metric in pair_metrics
                }
            selected_audits = [row for row in audits if (row["source"], row["size"], row["heuristic"]) == (source, size, heuristic)]
            audit_group[heuristic] = {
                "n_instances": len(selected_audits),
                "maps_with_admissibility_violations": sum(row["admissibility_violations"] > 0 for row in selected_audits),
                "maps_with_consistency_violations": sum(row["consistency_violations"] > 0 for row in selected_audits),
                "admissibility_violations_total": sum(row["admissibility_violations"] for row in selected_audits),
                "consistency_violations_total": sum(row["consistency_violations"] for row in selected_audits),
                "max_overestimate": max(row["max_overestimate"] for row in selected_audits),
                "n_reachable_states_total": sum(row["n_reachable_states"] for row in selected_audits),
                "violating_states_per_map": distribution([row["admissibility_violations"] for row in selected_audits]),
            }
        group["raw_vs_corrected"] = {
            f"{field}_difference": distribution([row[f"raw_minus_corrected_{field}"] for row in selected_pairs])
            for field in ("path_cost", "valid_pops", "expansions", "heap_peak")
        }
        group["raw_vs_corrected"]["same_valid_pops_maps"] = sum(row["raw_minus_corrected_valid_pops"] == 0 for row in selected_pairs)
        group["raw_vs_corrected"]["same_cost_maps"] = sum(row["raw_minus_corrected_path_cost"] == 0 for row in selected_pairs)
        groups.append(group)
        audit_groups.append(audit_group)
    return summary_rows, {
        "aggregation": {
            "within_map": "Median across per-call batch means; deterministic counts counted once per map.",
            "across_maps": "Median and linearly interpolated Q1/Q3 across independent maps; official excluded from random groups.",
            "paired": "Per-map reduction=1-strategy/zero and speedup=zero/strategy, then aggregate; not ratios of group medians.",
            "warning": "IQR is dispersion, not a confidence interval. Inadmissibility does not by itself imply suboptimal output in this domain.",
        },
        "overall": {
            "n_unique_maps": len(paired), "n_random_maps": sum(row["source"] == "random" for row in paired),
            "n_official_maps": sum(row["source"] == "official" for row in paired), "n_raw_batches": len(rows),
            "n_strategy_results": len(instances), "all_correct": all(row["correct"] for row in rows),
            "n_suboptimal_strategy_results": sum(row["optimal_gap"] != 0 for row in instances),
        },
        "groups": groups, "heuristic_audit": audit_groups,
    }


def _setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold", "figure.facecolor": "white", "svg.fonttype": "none",
    })
    return plt


def _save(figure, directory: Path, name: str):
    figure.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
    figure.savefig(directory / f"{name}.svg", bbox_inches="tight")


def _distributions_plot(plt, instances, directory, metrics, name, title):
    figure, axes = plt.subplots(1, len(metrics), figsize=(12.4, 4.9))
    selected = [row for row in instances if row["source"] == "random"]
    for axis, (metric, ylabel) in zip(axes, metrics):
        for index, heuristic in enumerate(HEURISTICS):
            values = [row[metric] for row in selected if row["heuristic"] == heuristic]
            box = axis.boxplot(values, positions=[index], widths=0.42, patch_artist=True, showfliers=False, manage_ticks=False)
            box["boxes"][0].set(facecolor=COLORS[heuristic], alpha=0.22)
            for line in box["medians"]:
                line.set(color=COLORS[heuristic], linewidth=2)
            axis.scatter([index + 0.13 * math.sin((i + 1) * 2.39996) for i in range(len(values))], values,
                         color=COLORS[heuristic], alpha=0.55, s=18, edgecolors="none")
        axis.set_xticks(range(3), ["Zero", "Raw\nManhattan", "Corrected\nManhattan"])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.17)
    n = len({row["instance_id"] for row in selected})
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.text(0.5, 0.01, f"Each dot: one random map (n={n}). Box: Q1-Q3; line: median. Official map is excluded.", ha="center", fontsize=9, color="#465264")
    figure.tight_layout(rect=(0, 0.085, 1, 0.92))
    _save(figure, directory, name)
    plt.close(figure)


def _paired_plot(plt, paired, directory):
    selected = [row for row in paired if row["source"] == "random"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.9))
    for axis, metric, label, multiplier, baseline in (
        (axes[0], "valid_pops_reduction", "Valid-pop reduction vs zero (%)", 100, 0),
        (axes[1], "api_speedup", "API speedup (zero / strategy)", 1, 1),
    ):
        for index, heuristic in enumerate(HEURISTICS[1:]):
            values = [row[f"{heuristic}_{metric}"] * multiplier for row in selected]
            axis.scatter([index + 0.13 * math.sin((i + 1) * 2.39996) for i in range(len(values))], values,
                         s=24, color=COLORS[heuristic], alpha=0.55, edgecolors="none")
            center, q1, q3 = median(values), percentile(values, 0.25), percentile(values, 0.75)
            axis.errorbar(index, center, yerr=[[center - q1], [q3 - center]], fmt="D", color="#1b3945", capsize=5, zorder=4)
        axis.set_xticks([0, 1], ["Raw\nManhattan", "Corrected\nManhattan"])
        axis.axhline(baseline, color="#818890", linestyle="--", linewidth=1)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.16)
    differences = [row["raw_minus_corrected_valid_pops"] for row in selected]
    axes[2].scatter(range(1, len(differences) + 1), differences, color="#4a6a79", s=28, alpha=0.7)
    axes[2].axhline(0, color="#818890", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Random map index")
    axes[2].set_ylabel("Valid pops: raw minus corrected")
    axes[2].grid(axis="y", alpha=0.16)
    figure.suptitle("Paired effects and raw-versus-corrected search work", fontsize=14, weight="bold")
    figure.text(0.5, 0.01, "Ratios are computed within each map before aggregation. Diamond/error bars: median and Q1-Q3.", ha="center", fontsize=9, color="#465264")
    figure.tight_layout(rect=(0, 0.09, 1, 0.92))
    _save(figure, directory, "03_paired_effects")
    plt.close(figure)


def _verify_trace(case, result, expected):
    from experiments.weighted_dataset import input_sha256
    if input_sha256(case) != expected["map_sha256"]:
        raise ValueError("轨迹地图与本轮输入哈希不一致")
    if result.cost != expected["path_cost"] or any(getattr(result.stats, field) != expected[field] for field in COUNTERS):
        raise ValueError("轨迹计数与本轮实测不符，不能用新实现解释旧结果")


def _official_plot(plt, result_dir, instances, directory):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Rectangle
    from src.problem2_terrain import search_weighted
    official = {row["heuristic"]: row for row in instances if row["source"] == "official"}
    path = result_dir / official["zero"]["map_file"]
    case = json.loads(path.read_text(encoding="utf-8-sig"))
    grid, start, end = case["grid"], tuple(case["start"]), tuple(case["end"])
    background = [[0 if (r, c) == end else 1 if cell in ("S", "E") else int(cell) for c, cell in enumerate(row)] for r, row in enumerate(grid)]
    cmap = ListedColormap(["#fffdf7", "#e7f0f7", "#d5e1ed", "#abbfd1"])
    norm = BoundaryNorm([-0.5, 0.5, 2, 5, 9], cmap.N)
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 5.4))
    for axis, heuristic in zip(axes, HEURISTICS):
        result = search_weighted(start, end, grid, heuristic=heuristic, trace=True)
        _verify_trace(case, result, official[heuristic])
        axis.imshow(background, cmap=cmap, norm=norm)
        for row, col in result.expanded_order:
            axis.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, color=COLORS[heuristic], alpha=0.22, linewidth=0))
        for r, row in enumerate(background):
            for c, value in enumerate(row):
                label = "S" if (r, c) == start else "E:0" if (r, c) == end else str(value)
                axis.text(c, r, label, ha="center", va="center", fontsize=10, color="#183143", weight="bold" if (r, c) in (start, end) else "normal")
        axis.plot([p[1] for p in result.path], [p[0] for p in result.path], color="#087450", linewidth=2.6, alpha=0.8)
        axis.set_xticks(range(len(grid[0])))
        axis.set_yticks(range(len(grid)))
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
        axis.set_xticks([c - 0.5 for c in range(len(grid[0]) + 1)], minor=True)
        axis.set_yticks([r - 0.5 for r in range(len(grid) + 1)], minor=True)
        axis.grid(which="minor", color="white", linewidth=1)
        axis.tick_params(which="minor", length=0)
        axis.set_title(f"{LABELS[heuristic]}\nPops {result.stats.valid_pops}; expansions {result.stats.expansions}; cost {result.cost}")
    figure.suptitle("Official weighted terrain: entering the goal costs zero", fontsize=14, weight="bold")
    figure.text(0.5, 0.015, "Numbers: entering costs; colored overlays: expanded cells; green: returned minimum-cost path. Start is not charged.", ha="center", fontsize=9, color="#465264")
    figure.tight_layout(rect=(0, 0.08, 1, 0.91))
    _save(figure, directory, "04_official_terrain")
    plt.close(figure)


def _admissibility_plot(plt, audits, directory):
    from matplotlib.patches import FancyArrowPatch, Rectangle
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    for index, heuristic in enumerate(HEURISTICS):
        values = [row["admissibility_violations"] for row in audits if row["source"] == "random" and row["heuristic"] == heuristic]
        axes[0].scatter([index + 0.12 * math.sin((i + 1) * 2.39996) for i in range(len(values))], values,
                        color=COLORS[heuristic], alpha=0.55, s=22)
        axes[0].plot(index, median(values), marker="D", color="#1b3945", markersize=7)
    axes[0].set_xticks(range(3), ["Zero", "Raw\nManhattan", "Corrected\nManhattan"])
    axes[0].set_ylabel("Overestimating states per random map")
    axes[0].grid(axis="y", alpha=0.16)
    axes[0].set_title("Admissibility checked against exact distances")
    axis = axes[1]
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    for x, label, fill in ((0.1, "Neighbor n", "#f4dec2"), (0.64, "Goal E", "#d8ede7")):
        axis.add_patch(Rectangle((x, 0.48), 0.26, 0.28, facecolor=fill, edgecolor="#33505d", linewidth=1.5))
        axis.text(x + 0.13, 0.62, label, ha="center", va="center", weight="bold", fontsize=12)
    axis.add_patch(FancyArrowPatch((0.37, 0.62), (0.63, 0.62), arrowstyle="->", mutation_scale=18, linewidth=1.6, color="#33505d"))
    axis.text(0.5, 0.76, "Entering E costs 0", ha="center", fontsize=11)
    axis.text(0.5, 0.34, "True remaining cost h*(n) = 0", ha="center", fontsize=12)
    axis.text(0.5, 0.22, "Raw Manhattan: 1 > 0  (overestimate)", ha="center", fontsize=11, color="#b56a20")
    axis.text(0.5, 0.11, "Corrected Manhattan: max(0, 1 - 1) = 0", ha="center", fontsize=11, color="#176c63")
    axis.set_title("A one-step counterexample to admissibility")
    figure.suptitle("Heuristic validity and returned-path optimality are separate checks", fontsize=14, weight="bold")
    figure.text(0.5, 0.015, "An overestimate disproves admissibility. It does not alone prove suboptimal output in this particular integer-cost domain.", ha="center", fontsize=9, color="#465264")
    figure.tight_layout(rect=(0, 0.09, 1, 0.92))
    _save(figure, directory, "05_admissibility")
    plt.close(figure)


def analyze_results(result_dir: Path) -> dict[str, Any]:
    result_dir = Path(result_dir)
    config = json.loads((result_dir / "config.json").read_text(encoding="utf-8-sig"))
    if config.get("status") != "completed":
        raise ValueError("仅可分析已完成的测量")
    for relative, expected in config.get("source_sha256", {}).items():
        if relative.startswith("src/") and sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"核心源码已变化：{relative}；不能混用旧统计与新轨迹")
    rows = _read_rows(result_dir / "raw.csv")
    if rows[0]["run_id"] != config.get("run_id") or len(rows) != config.get("raw_rows"):
        raise ValueError("原始运行编号或行数与配置不一致")
    instances = _aggregate_instances(rows)
    for instance in instances:
        actual = {row["batch_index"] for row in rows if row["instance_id"] == instance["instance_id"] and row["heuristic"] == instance["heuristic"]}
        if actual != set(range(config["batches"])):
            raise ValueError("某实例的测量批次不完整")
    paired = _paired_instances(instances)
    if len(paired) != config.get("measured_instances"):
        raise ValueError("实例数与配置不符")
    for instance_id in {row["instance_id"] for row in rows}:
        for batch in range(config["batches"]):
            triplet = [row for row in rows if row["instance_id"] == instance_id and row["batch_index"] == batch]
            if len(triplet) != 3 or len({row["batch_loops"] for row in triplet}) != 1 or {row["order_in_triplet"] for row in triplet} != {0, 1, 2}:
                raise ValueError("三策略批次的输入或循环次数不匹配")
    audits = _read_audits(result_dir / "heuristic_audit.csv", instances)
    summary_rows, report = _group_summary(instances, paired, audits, rows)
    report.update(run_id=rows[0]["run_id"], config_file="config.json", dataset_dir=config.get("dataset_dir"))
    _write_csv(result_dir / "per_instance.csv", instances)
    _write_csv(result_dir / "paired.csv", paired)
    _write_csv(result_dir / "summary.csv", summary_rows)
    directory = result_dir / "figures"
    directory.mkdir(exist_ok=True)
    plt = _setup_plotting()
    _distributions_plot(plt, instances, directory, [("valid_pops", "Valid pops (including goal)"), ("heap_peak", "Physical heap peak (entries)")], "01_workload", "Weighted maps: search work and frontier pressure")
    _distributions_plot(plt, instances, directory, [("search_ms_median", "Search-only time per call (ms)"), ("api_ms_median", "API batch-amortized time (ms)")], "02_runtime", "Weighted maps: search and preparation costs")
    _paired_plot(plt, paired, directory)
    _official_plot(plt, result_dir, instances, directory)
    _admissibility_plot(plt, audits, directory)
    (result_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    config.update(analysis_status="completed", analysis_completed_at_utc=datetime.now(timezone.utc).isoformat(), analysis_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (result_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="题二三种启发的配对统计与性质审计")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "problem2_heuristics")
    args = parser.parse_args(argv)
    try:
        result = analyze_results(args.results)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f"分析失败：{error}\n")
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
