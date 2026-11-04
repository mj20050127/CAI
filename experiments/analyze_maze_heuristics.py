"""汇总题一的配对实验，先按实例聚合计时批次，再跨地图统计。

运行：python -m experiments.analyze_maze_heuristics --results results/problem1_heuristics
原始 CSV 为数据来源；该脚本不重复测量算法，也不修改基础实现。
"""

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
HEURISTICS = ("zero", "manhattan")
COUNTERS = (
    "valid_pops", "expansions", "generated_successors", "pushes", "stale_pops", "heap_peak",
)
PAIR_METRICS = (
    "valid_pops_reduction", "expansions_reduction", "heap_peak_reduction",
    "search_speedup", "api_speedup",
)
COLORS = {"zero": "#244e76", "manhattan": "#df832e"}
LABELS = {"zero": "Zero heuristic", "manhattan": "Manhattan"}


def percentile(values: list[float], proportion: float) -> float:
    """排序后作线性插值；单样本的所有分位数均为该样本。"""
    if not values or not 0 <= proportion <= 1:
        raise ValueError("分位数需要非空数据和 [0,1] 范围内的比例")
    ordered = sorted(values)
    index = (len(ordered) - 1) * proportion
    low = int(math.floor(index))
    high = int(math.ceil(index))
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "median": median(values), "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75), "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values), "max": max(values),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"没有可写入 {path.name} 的记录")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "run_id", "instance_id", "map_file", "map_sha256", "source", "size", "density", "seed",
            "heuristic", "batch_index", "batch_loops", "order_in_pair", "path_cost", "bfs_cost", "correct",
            "search_ms_mean", "api_ms_mean", *COUNTERS,
        }
        if not required <= set(reader.fieldnames or []):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"原始数据缺少字段：{missing}")
        rows = []
        seen = set()
        for line_number, raw in enumerate(reader, start=2):
            row: dict[str, Any] = dict(raw)
            for field in ("size", "batch_index", "batch_loops", "order_in_pair", "path_cost", "bfs_cost", *COUNTERS):
                row[field] = int(row[field])
            if row["source"] == "official" and row["density"] == "":
                row["density"] = None
            else:
                row["density"] = float(row["density"])
                if not math.isfinite(row["density"]) or not 0 <= row["density"] <= 1:
                    raise ValueError(f"第 {line_number} 行的障碍率无效")
            for field in ("search_ms_mean", "api_ms_mean"):
                row[field] = float(row[field])
                if not math.isfinite(row[field]):
                    raise ValueError(f"第 {line_number} 行的 {field} 不是有限数")
            row["correct"] = str(row["correct"]).lower() in {"true", "1"}
            if row["heuristic"] not in HEURISTICS or row["source"] not in {"random", "official"}:
                raise ValueError(f"第 {line_number} 行存在未知算法或来源")
            if not row["correct"] or row["path_cost"] != row["bfs_cost"]:
                raise ValueError(f"第 {line_number} 行没有通过最短路径校验")
            if row["batch_loops"] < 1 or row["search_ms_mean"] <= 0 or row["api_ms_mean"] <= 0:
                raise ValueError(f"第 {line_number} 行的计时或批次循环数无效")
            if row["batch_index"] < 0 or row["order_in_pair"] not in (0, 1):
                raise ValueError(f"第 {line_number} 行的批次编号或配对执行顺序无效")
            if any(row[field] < 0 for field in COUNTERS):
                raise ValueError(f"第 {line_number} 行出现负计数")
            key = row["instance_id"], row["heuristic"], row["batch_index"]
            if key in seen:
                raise ValueError(f"重复的实例/算法/批次记录：{key}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("原始数据为空")
    if len({row["run_id"] for row in rows}) != 1:
        raise ValueError("同一个结果目录不能混合多个 run_id")
    return rows


def _aggregate_instances(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["instance_id"], row["heuristic"]].append(row)
    instances = []
    stable = ("source", "size", "map_file", "map_sha256", "density", "seed", "path_cost", "bfs_cost", *COUNTERS)
    for (instance_id, heuristic), batches in sorted(groups.items()):
        first = batches[0]
        for other in batches[1:]:
            if any(other[field] != first[field] for field in stable):
                raise ValueError(f"{instance_id}/{heuristic} 的批次之间输入或计数发生变化")
        entry = {
            "instance_id": instance_id, "source": first["source"], "size": first["size"],
            "map_file": first["map_file"], "map_sha256": first["map_sha256"],
            "density": first["density"], "seed": first["seed"], "heuristic": heuristic,
            "n_batches": len(batches), "batch_loops_min": min(row["batch_loops"] for row in batches),
            "batch_loops_max": max(row["batch_loops"] for row in batches),
            "correct": True, "path_cost": first["path_cost"], "bfs_cost": first["bfs_cost"],
        }
        entry.update({field: first[field] for field in COUNTERS})
        for source_field, prefix in (("search_ms_mean", "search_ms"), ("api_ms_mean", "api_ms")):
            stats = distribution([row[source_field] for row in batches])
            entry.update({f"{prefix}_{part}": stats[part] for part in ("median", "q1", "q3")})
        instances.append(entry)
    return instances


def _paired_instances(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in instances:
        groups[row["instance_id"]][row["heuristic"]] = row
    paired = []
    for instance_id, algorithms in sorted(groups.items()):
        if set(algorithms) != set(HEURISTICS):
            raise ValueError(f"实例 {instance_id} 缺少配对算法")
        zero, manhattan = algorithms["zero"], algorithms["manhattan"]
        stable = ("source", "size", "map_file", "map_sha256", "path_cost", "bfs_cost", "density", "seed")
        if any(zero[field] != manhattan[field] for field in stable):
            raise ValueError(f"实例 {instance_id} 的两种算法没有使用相同输入或最优路径代价")
        entry = {"instance_id": instance_id, **{field: zero[field] for field in stable}}
        entry["n_batches_zero"] = zero["n_batches"]
        entry["n_batches_manhattan"] = manhattan["n_batches"]
        for field in ("valid_pops", "expansions", "heap_peak"):
            entry[f"{field}_zero"] = zero[field]
            entry[f"{field}_manhattan"] = manhattan[field]
            entry[f"{field}_reduction"] = 1 - manhattan[field] / zero[field] if zero[field] else 0.0
        for prefix in ("search_ms", "api_ms"):
            entry[f"{prefix}_zero"] = zero[f"{prefix}_median"]
            entry[f"{prefix}_manhattan"] = manhattan[f"{prefix}_median"]
            entry[f"{prefix.removesuffix('_ms')}_speedup"] = zero[f"{prefix}_median"] / manhattan[f"{prefix}_median"]
        paired.append(entry)
    return paired


def _group_summary(
    instances: list[dict[str, Any]], paired: list[dict[str, Any]], rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries = []
    keys = sorted({(row["source"], row["size"]) for row in instances})
    json_groups = []
    for source, size in keys:
        json_group: dict[str, Any] = {"source": source, "size": size}
        for heuristic in HEURISTICS:
            selected = [row for row in instances if (row["source"], row["size"], row["heuristic"]) == (source, size, heuristic)]
            entry: dict[str, Any] = {"source": source, "size": size, "heuristic": heuristic, "n_instances": len(selected)}
            metrics = {}
            for column, prefix in (
                ("valid_pops", "valid_pops"), ("expansions", "expansions"),
                ("generated_successors", "generated_successors"), ("pushes", "pushes"),
                ("heap_peak", "heap_peak"), ("stale_pops", "stale_pops"),
                ("search_ms_median", "search_ms"), ("api_ms_median", "api_ms"),
                ("path_cost", "path_cost"),
            ):
                stats = distribution([row[column] for row in selected])
                metrics[prefix] = stats
                entry.update({f"{prefix}_{part}": stats[part] for part in ("median", "q1", "q3", "mean", "std")})
            summaries.append(entry)
            json_group[heuristic] = {"n_instances": len(selected), **metrics}
        selected_pairs = [row for row in paired if (row["source"], row["size"]) == (source, size)]
        json_group["n_instances"] = len(selected_pairs)
        json_group["paired"] = {
            metric: distribution([row[metric] for row in selected_pairs]) for metric in PAIR_METRICS
        }
        json_group["counts"] = {
            "valid_pops_better": sum(row["valid_pops_manhattan"] < row["valid_pops_zero"] for row in selected_pairs),
            "valid_pops_equal": sum(row["valid_pops_manhattan"] == row["valid_pops_zero"] for row in selected_pairs),
            "valid_pops_worse": sum(row["valid_pops_manhattan"] > row["valid_pops_zero"] for row in selected_pairs),
            "heap_peak_better": sum(row["heap_peak_manhattan"] < row["heap_peak_zero"] for row in selected_pairs),
            "heap_peak_equal": sum(row["heap_peak_manhattan"] == row["heap_peak_zero"] for row in selected_pairs),
            "heap_peak_worse": sum(row["heap_peak_manhattan"] > row["heap_peak_zero"] for row in selected_pairs),
            "search_time_better": sum(row["search_speedup"] > 1 for row in selected_pairs),
            "api_time_better": sum(row["api_speedup"] > 1 for row in selected_pairs),
        }
        json_groups.append(json_group)
    return summaries, {
        "aggregation": {
            "stage1": "Within each map/heuristic: median of per-call batch means; counters counted once per map.",
            "stage2": "Across independent maps: median and Q1/Q3; quartiles use linear interpolation.",
            "paired": "Reduction = 1 - Manhattan/zero; speedup = zero/Manhattan; computed per map before aggregation.",
            "warning": "Median of paired ratios is not the ratio of group medians. IQR is dispersion, not a confidence interval.",
        },
        "overall": {
            "n_unique_maps": len(paired),
            "n_random_maps": sum(row["source"] == "random" for row in paired),
            "n_official_maps": sum(row["source"] == "official" for row in paired),
            "n_raw_batches": len(rows), "all_correct": all(row["correct"] for row in rows),
        },
        "groups": json_groups,
    }


def _setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "svg.fonttype": "none",
    })
    return plt


def _save_figure(figure, directory: Path, name: str) -> None:
    figure.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
    figure.savefig(directory / f"{name}.svg", bbox_inches="tight")


def _comparison_plot(plt, groups: list[dict[str, Any]], columns: list[tuple[str, str]], directory: Path, name: str, title: str, footer: str) -> None:
    figure, axes = plt.subplots(1, len(columns), figsize=(12.2, 4.5))
    axes = list(axes) if len(columns) > 1 else [axes]
    positions = list(range(len(groups)))
    for axis, (metric, ylabel) in zip(axes, columns):
        for offset, heuristic in ((-0.15, "zero"), (0.15, "manhattan")):
            medians = [group[heuristic][metric]["median"] for group in groups]
            lower = [value - group[heuristic][metric]["q1"] for value, group in zip(medians, groups)]
            upper = [group[heuristic][metric]["q3"] - value for value, group in zip(medians, groups)]
            axis.errorbar(
                [position + offset for position in positions], medians, yerr=[lower, upper],
                color=COLORS[heuristic], marker="o", markersize=6, linewidth=1.8,
                capsize=4, label=LABELS[heuristic],
            )
        axis.set_xticks(positions, [f"{group['size']} x {group['size']}" for group in groups])
        axis.set_xlabel("Map size")
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.18, which="both")
        axis.legend(frameon=False)
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=9, color="#465264")
    figure.tight_layout(rect=(0, 0.075, 1, 0.92))
    _save_figure(figure, directory, name)
    plt.close(figure)


def _paired_plot(plt, groups: list[dict[str, Any]], pairs: list[dict[str, Any]], directory: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.3, 4.6))
    specifications = (
        ("valid_pops_reduction", "Valid-pop reduction (%)", 100, 0),
        ("heap_peak_reduction", "Heap-peak reduction (%)", 100, 0),
        ("search_speedup", "Search speedup (zero / Manhattan)", 1, 1),
    )
    for axis, (metric, ylabel, scale, baseline) in zip(axes, specifications):
        for index, group in enumerate(groups):
            samples = [row[metric] * scale for row in pairs if row["source"] == group["source"] and row["size"] == group["size"]]
            x = [index + 0.12 * math.sin((j + 1) * 2.39996) for j in range(len(samples))]
            axis.scatter(x, samples, s=22, color="#48848b", alpha=0.62, edgecolors="none", zorder=2)
            mid, low, high = median(samples), percentile(samples, 0.25), percentile(samples, 0.75)
            axis.errorbar(index, mid, yerr=[[mid - low], [high - mid]], fmt="D", color="#183b49", markersize=6, capsize=6, linewidth=2, zorder=3)
        axis.axhline(baseline, color="#7a7d83", linestyle="--", linewidth=1)
        axis.set_xticks(range(len(groups)), [str(group["size"]) for group in groups])
        axis.set_xlabel("Map side length")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.16)
    figure.suptitle("Paired effects: each dot is one independent map", fontsize=14, weight="bold")
    figure.text(0.5, 0.01, "Diamond: median; error bar: Q1-Q3 across maps. Positive reduction / speedup > 1 favors Manhattan.", ha="center", va="bottom", fontsize=9, color="#465264")
    figure.tight_layout(rect=(0, 0.075, 1, 0.92))
    _save_figure(figure, directory, "03_paired_improvements")
    plt.close(figure)


def _verify_trace(case: dict[str, Any], result: Any, expected: dict[str, Any]) -> None:
    """轨迹只作解释，仍须与该轮保存输入和实测计数一致，不能混用新代码。"""
    from experiments.maze_dataset import map_sha256
    if map_sha256(case["start"], case["end"], case["grid"]) != expected["map_sha256"]:
        raise ValueError("轨迹地图哈希与原始记录不一致")
    if result.cost != expected["path_cost"] or any(
        getattr(result.stats, field) != expected[field] for field in COUNTERS
    ):
        raise ValueError("轨迹搜索与保存的原始结果不一致；不能用已修改的算法解释旧数据")


def _official_search_plot(plt, result_dir: Path, directory: Path, instances: list[dict[str, Any]]) -> None:
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    from src.problem1_maze import search_maze

    official_rows = {row["heuristic"]: row for row in instances if row["source"] == "official"}
    with (result_dir / official_rows["zero"]["map_file"]).open(encoding="utf-8-sig") as stream:
        case = json.load(stream)
    grid = case["grid"]
    start, end = tuple(case["start"]), tuple(case["end"])
    background = [[int(cell == "X") for cell in row] for row in grid]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 5.5))
    for axis, heuristic in zip(axes, HEURISTICS):
        result = search_maze(start, end, grid, heuristic=heuristic, trace=True)
        _verify_trace(case, result, official_rows[heuristic])
        axis.imshow(background, cmap=ListedColormap(["#faf9f4", "#283b4d"]), vmin=0, vmax=1)
        for step, (row, col) in enumerate(result.expanded_order, start=1):
            axis.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, color=COLORS[heuristic], alpha=0.24, linewidth=0))
            if (row, col) not in (start, end):
                axis.text(col, row, str(step), ha="center", va="center", fontsize=10, color="#27333d", zorder=5)
        axis.plot([point[1] for point in result.path], [point[0] for point in result.path], color="#197b61", linewidth=2.6, marker="o", markersize=4, alpha=0.86)
        for point, label in ((start, "S"), (end, "E")):
            axis.text(point[1], point[0], label, ha="center", va="center", weight="bold", fontsize=12,
                      bbox={"boxstyle": "circle,pad=0.2", "facecolor": "white", "edgecolor": "#197b61"}, zorder=6)
        axis.set_xticks(range(len(grid[0])))
        axis.set_yticks(range(len(grid)))
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
        axis.set_xticks([x - 0.5 for x in range(len(grid[0]) + 1)], minor=True)
        axis.set_yticks([y - 0.5 for y in range(len(grid) + 1)], minor=True)
        axis.grid(which="minor", color="white", linewidth=1.2)
        axis.tick_params(which="minor", length=0)
        axis.set_title(f"{LABELS[heuristic]}\nValid pops {result.stats.valid_pops}; expansions {result.stats.expansions}; cost {result.cost}")
    figure.suptitle("Official maze: equal optimum, different search footprint", fontsize=14, weight="bold")
    figure.text(0.5, 0.015, "Colored cells: expanded; numbers: expansion order; green line: shortest path. Goal is not an expansion.", ha="center", va="bottom", fontsize=9, color="#465264")
    figure.tight_layout(rect=(0, 0.065, 1, 0.92))
    _save_figure(figure, directory, "04_official_search")
    plt.close(figure)


def _representative_search_plot(plt, result_dir: Path, directory: Path, paired: list[dict[str, Any]], instances: list[dict[str, Any]]) -> dict[str, Any] | None:
    """选择 51 格组中节省率最接近组中位数的地图，而非挑最好看的个例。"""
    from matplotlib.colors import ListedColormap
    from src.problem1_maze import search_maze

    candidates = [row for row in paired if row["source"] == "random" and row["size"] == 51]
    if not candidates:
        return None
    center = median(row["valid_pops_reduction"] for row in candidates)
    selected = min(candidates, key=lambda row: (abs(row["valid_pops_reduction"] - center), row["instance_id"]))
    path = (result_dir / selected["map_file"]).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        case = json.load(stream)
    grid = case["grid"]
    start, end = tuple(case["start"]), tuple(case["end"])
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 6.8))
    for axis, heuristic in zip(axes, HEURISTICS):
        result = search_maze(start, end, grid, heuristic=heuristic, trace=True)
        expected = next(row for row in instances if row["instance_id"] == selected["instance_id"] and row["heuristic"] == heuristic)
        _verify_trace(case, result, expected)
        expanded = set(result.expanded_order)
        cells = [[1 if cell == "X" else 2 if (row, col) in expanded else 0 for col, cell in enumerate(line)] for row, line in enumerate(grid)]
        fill = "#b9ccde" if heuristic == "zero" else "#f0c99d"
        axis.imshow(cells, cmap=ListedColormap(["#faf9f4", "#283b4d", fill]), vmin=0, vmax=2, interpolation="nearest")
        axis.plot([point[1] for point in result.path], [point[0] for point in result.path], color="#087450", linewidth=1.35)
        axis.scatter([start[1], end[1]], [start[0], end[0]], color="#087450", s=24, edgecolor="white", linewidth=0.7, zorder=5)
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
        axis.set_title(f"{LABELS[heuristic]}\nValid pops {result.stats.valid_pops}; expansions {result.stats.expansions}; cost {result.cost}")
    figure.suptitle("Representative 51 x 51 maze: explored cells and shortest paths", fontsize=14, weight="bold")
    figure.text(0.5, 0.017,
                f"{selected['instance_id']}: nearest to the group's median valid-pop reduction ({100 * selected['valid_pops_reduction']:.1f}%).\n"
                "Dark: walls; pale blue/orange: expanded; off-white: not expanded; green: returned shortest path.",
                ha="center", va="bottom", fontsize=9.5, color="#465264")
    figure.tight_layout(rect=(0, 0.14, 1, 0.92))
    _save_figure(figure, directory, "05_representative_search")
    plt.close(figure)
    return {
        "instance_id": selected["instance_id"], "map_file": selected["map_file"],
        "map_sha256": selected["map_sha256"], "valid_pops_reduction": selected["valid_pops_reduction"],
        "selection_rule": "Among 51x51 maps, minimum absolute deviation from median paired valid-pop reduction; tie by instance_id.",
    }


def analyze_results(result_dir: Path) -> dict[str, Any]:
    """校验并生成明细、配对表、分组汇总 JSON/CSV 与静态图。"""
    result_dir = Path(result_dir)
    with (result_dir / "config.json").open(encoding="utf-8-sig") as stream:
        config = json.load(stream)
    if config.get("status") != "completed":
        raise ValueError("仅可分析已完成采集的结果；请等待测量结束")
    for relative, expected_hash in config.get("source_sha256", {}).items():
        if relative.startswith("src/") and sha256((ROOT / relative).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"核心源代码已变化：{relative}；请用原版本重绘，避免混用新轨迹和旧数据")
    rows = _read_rows(result_dir / "raw.csv")
    if rows[0]["run_id"] != config.get("run_id"):
        raise ValueError("raw.csv 的 run_id 与配置不一致")
    if "raw_rows" in config and len(rows) != config["raw_rows"]:
        raise ValueError("raw.csv 行数与 config.json 不符，结果可能不完整")
    instances = _aggregate_instances(rows)
    if "batches" in config:
        expected = set(range(config["batches"]))
        for instance in instances:
            actual = {row["batch_index"] for row in rows if row["instance_id"] == instance["instance_id"] and row["heuristic"] == instance["heuristic"]}
            if actual != expected:
                raise ValueError(f"{instance['instance_id']}/{instance['heuristic']} 的计时批次不完整")
    paired = _paired_instances(instances)
    if len(paired) != config.get("measured_instances"):
        raise ValueError("测量实例数与配置不一致，数据可能不完整")
    for instance_id in {row["instance_id"] for row in rows}:
        for batch_index in range(config["batches"]):
            pair = [row for row in rows if row["instance_id"] == instance_id and row["batch_index"] == batch_index]
            if len({row["batch_loops"] for row in pair}) != 1 or {row["order_in_pair"] for row in pair} != {0, 1}:
                raise ValueError("配对批次的循环数或运行顺序不符合实验约定")
    summaries, report = _group_summary(instances, paired, rows)
    report["run_id"] = rows[0]["run_id"]
    report["config_file"] = "config.json"
    report["dataset_dir"] = config.get("dataset_dir")
    _write_csv(result_dir / "per_instance.csv", instances)
    _write_csv(result_dir / "paired.csv", paired)
    _write_csv(result_dir / "summary.csv", summaries)
    directory = result_dir / "figures"
    directory.mkdir(exist_ok=True)
    plt = _setup_plotting()
    random_groups = [group for group in report["groups"] if group["source"] == "random"]
    plot_groups = random_groups or report["groups"]
    samples = ", ".join(f"{group['size']}: n={group['n_instances']}" for group in plot_groups)
    _comparison_plot(plt, plot_groups, [("valid_pops", "Valid pops (including start and goal)"), ("heap_peak", "Physical heap peak (entries)")], directory, "01_workload", "Search work and frontier pressure", f"Median and Q1-Q3 across independent maps ({samples}); log y-axes.")
    _comparison_plot(plt, plot_groups, [("search_ms", "Search-only time per call (ms)"), ("api_ms", "Whole API time per call (ms)")], directory, "02_runtime", "Runtime: search-only and whole-call costs", "First aggregate repeated timing batches within each map; then median and Q1-Q3 across maps. Log y-axes.")
    _paired_plot(plt, plot_groups, paired, directory)
    _official_search_plot(plt, result_dir, directory, instances)
    report["representative_map"] = _representative_search_plot(plt, result_dir, directory, paired, instances)
    (result_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    config["analysis_status"] = "completed"
    config["analysis_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    config["analysis_source_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    (result_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="汇总题一零启发与曼哈顿配对实验并绘图")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "problem1_heuristics")
    args = parser.parse_args(argv)
    try:
        report = analyze_results(args.results)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f"分析失败：{error}\n")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
