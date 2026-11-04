"""题三：同树同顺序的 Minimax / Alpha-Beta 配对汇总，不重新采集计时。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from experiments.analyze_weighted_heuristics import distribution, _save, _setup_plotting


ROOT = Path(__file__).resolve().parent.parent
ALGORITHMS = ("minimax", "alpha_beta")
COUNTERS = ("node_visits", "leaf_evals", "cutoff_events", "pruned_branches")
TIMINGS = ("search_ms", "api_ms")
META = ("source", "tree_file", "tree_sha256", "seed", "depth", "branch_factor",
        "node_count", "leaf_count", "reference_value")
COLORS = {"minimax": "#315d7a", "alpha_beta": "#21847b"}
LABELS = {"minimax": "Minimax", "alpha_beta": "Alpha-Beta"}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("不能保存空的汇总表")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    integers = ("depth", "branch_factor", "node_count", "leaf_count", "batch_index",
                "batch_loops", "order_in_pair", "root_value", "reference_value", *COUNTERS)
    required = {"run_id", "instance_id", "tree_file", "tree_sha256", "source", "seed",
                "algorithm", "correct", *integers, "search_ms_mean", "api_ms_mean", "batch_api_ms"}
    rows, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("原始数据字段不完整")
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for field in integers:
                row[field] = int(row[field])
            for field in ("search_ms_mean", "api_ms_mean", "batch_api_ms"):
                row[field] = float(row[field])
                if not math.isfinite(row[field]) or row[field] <= 0:
                    raise ValueError("计时必须为有限正数")
            seed = row["seed"]
            row["seed"] = None if seed in ("", "None", "null") else int(seed)
            if str(row["correct"]).lower() not in ("true", "1"):
                raise ValueError("结果未通过正确性校验")
            row["correct"] = True
            if row["algorithm"] not in ALGORITHMS or row["source"] not in ("random", "official"):
                raise ValueError("未知算法或输入来源")
            if row["root_value"] != row["reference_value"]:
                raise ValueError("根值与独立参考不一致")
            if (row["batch_index"] < 0 or row["batch_loops"] < 1 or row["order_in_pair"] not in (0, 1)
                    or row["depth"] < 0 or row["branch_factor"] < 0
                    or row["node_count"] < 1 or not 1 <= row["leaf_count"] <= row["node_count"]):
                raise ValueError("规模、批次或执行顺序无效")
            if any(row[field] < 0 for field in COUNTERS):
                raise ValueError("节点计数不能为负")
            if not (1 <= row["leaf_evals"] <= row["leaf_count"]
                    and row["leaf_evals"] <= row["node_visits"] <= row["node_count"]):
                raise ValueError("访问或叶评计数超出输入规模")
            if row["algorithm"] == "minimax" and (
                row["node_visits"] != row["node_count"] or row["leaf_evals"] != row["leaf_count"]
                or row["cutoff_events"] or row["pruned_branches"]
            ):
                raise ValueError("Minimax 完整树计数不符")
            if (row["cutoff_events"] > row["pruned_branches"]
                    or row["cutoff_events"] > row["node_visits"] - row["leaf_evals"]
                    or row["pruned_branches"] > row["node_count"] - row["node_visits"]):
                raise ValueError("剪枝事件或直接兄弟计数不符")
            if not math.isclose(row["batch_api_ms"], row["api_ms_mean"] * row["batch_loops"], rel_tol=1e-8):
                raise ValueError("完整批时与每次平均耗时不符")
            if row["api_ms_mean"] < row["search_ms_mean"]:
                raise ValueError("接口时间不能小于内部搜索时间")
            key = row["instance_id"], row["algorithm"], row["batch_index"]
            if key in seen:
                raise ValueError("重复的实例、算法、批次记录")
            seen.add(key)
            rows.append(row)
    if not rows or len({row["run_id"] for row in rows}) != 1:
        raise ValueError("数据为空或混合多个运行编号")
    return rows


def _validate_batches(rows: list[dict[str, Any]], batches: int) -> None:
    if type(batches) is not int or batches < 1:
        raise ValueError("配置批次数无效")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["instance_id"]].append(row)
    for entries in grouped.values():
        if {(r["algorithm"], r["batch_index"]) for r in entries} != {
            (algorithm, batch) for algorithm in ALGORITHMS for batch in range(batches)
        } or len(entries) != len(ALGORITHMS) * batches:
            raise ValueError("实例的两算法批次不完整")
        for batch in range(batches):
            pair = [r for r in entries if r["batch_index"] == batch]
            if {r["order_in_pair"] for r in pair} != {0, 1} or len({r["batch_loops"] for r in pair}) != 1:
                raise ValueError("同树两算法的批次循环或顺序不匹配")


def _aggregate_instances(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["instance_id"], row["algorithm"]].append(row)
    instances = []
    stable = (*META, "root_value", *COUNTERS)
    for (instance_id, algorithm), entries in sorted(grouped.items()):
        first = entries[0]
        if any(row[field] != first[field] for row in entries for field in stable):
            raise ValueError("同树同算法的输入、结果或确定性计数发生变化")
        entry = {"instance_id": instance_id, "algorithm": algorithm,
                 **{field: first[field] for field in stable}, "n_batches": len(entries), "correct": True,
                 "batch_loops_min": min(row["batch_loops"] for row in entries),
                 "batch_loops_max": max(row["batch_loops"] for row in entries)}
        for timing in TIMINGS:
            stats = distribution([row[f"{timing}_mean"] for row in entries])
            entry.update({f"{timing}_{part}": stats[part] for part in ("median", "q1", "q3")})
        instances.append(entry)
    return instances


def _paired_instances(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(dict)
    for row in instances:
        if row["algorithm"] in grouped[row["instance_id"]]:
            raise ValueError("同实例含重复算法汇总")
        grouped[row["instance_id"]][row["algorithm"]] = row
    paired = []
    for instance_id, algorithms in sorted(grouped.items()):
        if set(algorithms) != set(ALGORITHMS):
            raise ValueError("每棵树必须具有完整两算法配对")
        mm, ab = (algorithms[name] for name in ALGORITHMS)
        if any(mm[field] != ab[field] for field in (*META, "root_value")):
            raise ValueError("两算法未使用相同输入或根值")
        if ab["node_visits"] > mm["node_visits"] or ab["leaf_evals"] > mm["leaf_evals"]:
            raise ValueError("Alpha-Beta 访问次数不能超过完整 Minimax")
        row = {"instance_id": instance_id, **{field: mm[field] for field in META}, "root_value": mm["root_value"]}
        for algorithm, current in algorithms.items():
            row.update({f"{algorithm}_{field}": current[field] for field in COUNTERS})
            row.update({f"{algorithm}_{timing}": current[f"{timing}_median"] for timing in TIMINGS})
        row["node_reduction"] = 1 - ab["node_visits"] / mm["node_visits"]
        row["leaf_reduction"] = 1 - ab["leaf_evals"] / mm["leaf_evals"]
        for timing in TIMINGS:
            name = timing.removesuffix("_ms")
            row[f"{name}_speedup"] = mm[f"{timing}_median"] / ab[f"{timing}_median"]
            row[f"{name}_time_reduction"] = 1 - ab[f"{timing}_median"] / mm[f"{timing}_median"]
        paired.append(row)
    return paired


def _group_summary(instances, paired, rows):
    groups, summary_rows = [], []
    metrics = (*COUNTERS, *TIMINGS)
    paired_metrics = ("node_reduction", "leaf_reduction", "search_speedup", "api_speedup",
                      "search_time_reduction", "api_time_reduction")
    for source, depth in sorted({(row["source"], row["depth"]) for row in paired}):
        selected_pairs = [r for r in paired if (r["source"], r["depth"]) == (source, depth)]
        group = {"source": source, "depth": depth, "n_instances": len(selected_pairs)}
        for algorithm in ALGORITHMS:
            selected = [r for r in instances if (r["source"], r["depth"], r["algorithm"]) == (source, depth, algorithm)]
            stats = {metric: distribution([r[f"{metric}_median"] if metric in TIMINGS else r[metric] for r in selected])
                     for metric in metrics}
            group[algorithm] = {"n_instances": len(selected), **stats}
            summary_row = {"source": source, "depth": depth, "algorithm": algorithm, "n_instances": len(selected)}
            for metric, values in stats.items():
                summary_row.update({f"{metric}_{part}": value for part, value in values.items()})
            summary_rows.append(summary_row)
        pair_stats = {metric: distribution([r[metric] for r in selected_pairs]) for metric in paired_metrics}
        for prefix in ("search", "api"):
            pair_stats[f"faster_{prefix}_trees"] = sum(r[f"{prefix}_speedup"] > 1 for r in selected_pairs)
            pair_stats[f"same_{prefix}_trees"] = sum(r[f"{prefix}_speedup"] == 1 for r in selected_pairs)
            pair_stats[f"slower_{prefix}_trees"] = sum(r[f"{prefix}_speedup"] < 1 for r in selected_pairs)
        pair_stats["fewer_node_trees"] = sum(r["node_reduction"] > 0 for r in selected_pairs)
        pair_stats["same_node_trees"] = sum(r["node_reduction"] == 0 for r in selected_pairs)
        pair_stats["fewer_leaf_trees"] = sum(r["leaf_reduction"] > 0 for r in selected_pairs)
        pair_stats["same_leaf_trees"] = sum(r["leaf_reduction"] == 0 for r in selected_pairs)
        group["paired"] = pair_stats
        groups.append(group)
    return summary_rows, {
        "aggregation": {
            "within_tree": "Median of per-call batch means; deterministic counters counted once per tree.",
            "across_trees": "Separate source/depth groups; independent trees, not repeated batches. Median and linearly interpolated Q1/Q3.",
            "paired": "Per tree: reduction=1-AB/Minimax; speedup=Minimax/AB. Aggregate these ratios, not ratios of group medians.",
            "search_scope": "Recursive search including counters and path-tuple bookkeeping; trace disabled. Excludes full input validation.",
            "api_scope": "Input validation plus search, result objects, loop and collection; excludes tree loading/generation, oracle, trace and I/O.",
            "uncertainty": "IQR describes tree-to-tree dispersion, not a confidence interval.",
            "pruning_scope": "cutoff_events counts actual cutoffs; pruned_branches counts skipped direct siblings, not all omitted descendants.",
        },
        "overall": {"n_unique_trees": len(paired), "n_random_trees": sum(r["source"] == "random" for r in paired),
                    "n_official_trees": sum(r["source"] == "official" for r in paired), "n_raw_batches": len(rows),
                    "n_algorithm_results": len(instances), "total_timed_calls": sum(r["batch_loops"] for r in rows),
                    "all_correct": all(r["correct"] for r in rows)},
        "groups": groups,
    }


def _verify_sources(result_dir: Path, config: dict[str, Any]) -> None:
    hashes = config.get("source_sha256", {})
    required = {"src/game_tree.py", "experiments/game_tree_dataset.py", "experiments/exp3_minimax_alpha_beta.py",
                "experiments/analyze_game_tree.py"}
    if not required <= set(hashes):
        raise ValueError("缺少题三复核所需的源码哈希")
    for relative, expected in hashes.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"源码已变化：{relative}；请使用本轮归档源码重现")
    archive_path = result_dir / "source_baseline.zip"
    if sha256(archive_path.read_bytes()).hexdigest() != config.get("source_baseline_sha256"):
        raise ValueError("源码归档哈希不符")
    with ZipFile(archive_path) as archive:
        if set(archive.namelist()) != set(hashes) or len(archive.namelist()) != len(hashes):
            raise ValueError("源码归档成员不完整或重复")
        if any(sha256(archive.read(name)).hexdigest() != expected for name, expected in hashes.items()):
            raise ValueError("源码归档内容哈希不符")


def _verify_dataset(result_dir: Path, paired: list[dict[str, Any]], config: dict[str, Any]):
    from experiments.game_tree_dataset import load_dataset
    manifest_path = result_dir / "dataset" / "manifest.json"
    if config.get("dataset_manifest_sha256") and sha256(manifest_path.read_bytes()).hexdigest() != config["dataset_manifest_sha256"]:
        raise ValueError("数据集 manifest 哈希不符")
    manifest, cases = load_dataset(result_dir / "dataset")
    lookup = {case["instance_id"]: case for case in cases}
    if set(lookup) != {r["instance_id"] for r in paired}:
        raise ValueError("实际输入与测量实例覆盖不一致")
    for row in paired:
        case = lookup[row["instance_id"]]
        if any(row[field] != case[field] for field in META):
            raise ValueError("测量元数据与保存的实际树不一致")
    return manifest, lookup


def _verify_correctness(path: Path, instances: list[dict[str, Any]]) -> None:
    expected = {(r["instance_id"], r["algorithm"]): r for r in instances}
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"instance_id", "algorithm", *META, "root_value", "correct", *COUNTERS}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("独立正确性记录字段不完整")
        for raw in reader:
            key = raw["instance_id"], raw["algorithm"]
            if key not in expected or key in seen:
                raise ValueError("正确性记录含重复或未知实例算法")
            row = expected[key]
            if str(raw["correct"]).lower() not in ("true", "1"):
                raise ValueError("存在失败的正确性记录")
            for field in (*META, "root_value", *COUNTERS):
                actual = raw[field]
                if field == "seed":
                    actual = None if actual in ("", "None", "null") else int(actual)
                elif isinstance(row[field], int):
                    actual = int(actual)
                if actual != row[field]:
                    raise ValueError("正确性记录与正式测量不一致")
            seen.add(key)
    if seen != set(expected):
        raise ValueError("正确性记录未覆盖全部实例算法")


def _tree_paths(data):
    root = data.get("tree_structure", data)
    paths = {}
    def walk(node, path):
        paths[path] = node
        for index, child in enumerate(node.get("children", [])):
            walk(child, path + (index,))
    walk(root, ())
    return paths


def _verify_official_trace(result_dir, instances, lookup):
    official_ids = [key for key, case in lookup.items() if case["source"] == "official"]
    if len(official_ids) != 1:
        raise ValueError("题三基准要求恰好一个官方用例")
    instance_id = official_ids[0]
    paths = _tree_paths(lookup[instance_id]["tree_data"])
    traces = json.loads((result_dir / "official_trace.json").read_text(encoding="utf-8-sig"))
    if not set(ALGORITHMS) <= set(traces):
        raise ValueError("官方轨迹缺少两算法")
    expected = {r["algorithm"]: r for r in instances if r["instance_id"] == instance_id}
    for algorithm in ALGORITHMS:
        trace = traces[algorithm]
        if trace["value"] != expected[algorithm]["root_value"] or any(
            trace["stats"][field] != expected[algorithm][field] for field in COUNTERS
        ):
            raise ValueError("官方轨迹与实测根值或计数不一致")
        entered = [tuple(e["path"]) for e in trace["trace"] if e["event"] == "enter"]
        leaves = [e for e in trace["trace"] if e["event"] == "leaf"]
        cuts = [e for e in trace["trace"] if e["event"] == "cutoff"]
        if len(entered) != len(set(entered)) or not set(entered) <= set(paths) or len(entered) != expected[algorithm]["node_visits"]:
            raise ValueError("官方访问轨迹不完整或重复")
        expected_leaves = {path for path in entered if not paths[path].get("children")}
        if ({tuple(e["path"]) for e in leaves} != expected_leaves
                or len(leaves) != expected[algorithm]["leaf_evals"]
                or any(e["value"] != paths[tuple(e["path"])]["value"] for e in leaves)):
            raise ValueError("官方叶评轨迹不符")
        skipped = [tuple(path) for event in cuts for path in event["skipped_paths"]]
        if len(cuts) != expected[algorithm]["cutoff_events"] or len(skipped) != expected[algorithm]["pruned_branches"]:
            raise ValueError("官方截断轨迹计数不符")
        if not set(skipped) <= set(paths) or any(
            path[:len(skip)] == skip for skip in skipped for path in entered
        ):
            raise ValueError("官方剪枝路径与已访问路径矛盾")
        omitted = {path for path in paths if any(path[:len(skip)] == skip for skip in skipped)}
        if set(entered) | omitted != set(paths):
            raise ValueError("官方轨迹未解释全部未访问节点")
    return lookup[instance_id], traces


def _depth_plot(plt, groups, directory, metrics, name, title):
    random_groups = sorted((g for g in groups if g["source"] == "random"), key=lambda g: g["depth"])
    if not random_groups:
        return
    depths = [g["depth"] for g in random_groups]
    figure, axes = plt.subplots(1, len(metrics), figsize=(12.6, 4.9), squeeze=False)
    for axis, (metric, label) in zip(axes[0], metrics):
        for algorithm in ALGORITHMS:
            medians = [g[algorithm][metric]["median"] for g in random_groups]
            lower = [g[algorithm][metric]["q1"] for g in random_groups]
            upper = [g[algorithm][metric]["q3"] for g in random_groups]
            axis.plot(depths, medians, "o-", color=COLORS[algorithm], label=LABELS[algorithm], linewidth=2)
            axis.fill_between(depths, lower, upper, color=COLORS[algorithm], alpha=0.15)
        axis.set_xticks(depths)
        axis.set_xlabel("Depth (root-to-leaf edges)")
        axis.set_ylabel(label)
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.18)
        axis.legend(frameon=False)
    sizes = ", ".join(f"d={g['depth']}: n={g['n_instances']}" for g in random_groups)
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.text(0.5, 0.015, f"Random trees only ({sizes}). Lines: median; bands: Q1-Q3, not confidence intervals.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.075, 1, 0.92))
    _save(figure, directory, name)
    plt.close(figure)


def _paired_plot(plt, paired, directory):
    selected = [r for r in paired if r["source"] == "random"]
    if not selected:
        return
    depths = sorted({r["depth"] for r in selected})
    metrics = (("node_reduction", "Node-visit reduction (%)", 100, 0),
               ("search_speedup", "Search speedup (Minimax / AB)", 1, 1),
               ("api_speedup", "Full API speedup (Minimax / AB)", 1, 1))
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.9))
    for axis, (metric, label, scale, baseline) in zip(axes, metrics):
        for index, depth in enumerate(depths):
            values = [r[metric] * scale for r in selected if r["depth"] == depth]
            axis.scatter([index + 0.16 * math.sin((i + 1) * 2.39996) for i in range(len(values))], values,
                         s=22, alpha=0.55, color=COLORS["alpha_beta"], edgecolors="none")
            stats = distribution(values)
            axis.errorbar(index, stats["median"], yerr=[[stats["median"] - stats["q1"]], [stats["q3"] - stats["median"]]],
                          color="#193c4a", fmt="D", capsize=5, zorder=4)
        axis.set_xticks(range(len(depths)), [str(depth) for depth in depths])
        axis.set_xlabel("Depth")
        axis.set_ylabel(label)
        axis.axhline(baseline, linestyle="--", color="#8b929a", linewidth=1)
        axis.grid(axis="y", alpha=0.15)
    figure.suptitle("Paired pruning benefit on the same tree and child order", fontsize=14, weight="bold")
    figure.text(0.5, 0.015, "Each dot: one tree after batch aggregation. Diamonds: medians; whiskers: Q1-Q3. Speedup > 1 means faster.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.075, 1, 0.92))
    _save(figure, directory, "03_paired_savings")
    plt.close(figure)


def _official_plot(plt, case, traces, directory):
    nodes = _tree_paths(case["tree_data"])
    leaf_paths = [path for path, node in nodes.items() if not node.get("children")]
    x_positions = {path: float(i) for i, path in enumerate(leaf_paths)}
    for path in sorted(nodes, key=len, reverse=True):
        if path not in x_positions:
            children = [path + (i,) for i in range(len(nodes[path]["children"]))]
            x_positions[path] = sum(x_positions[child] for child in children) / len(children)
    figure, axes = plt.subplots(1, 2, figsize=(14.6, 5.9))
    for axis, algorithm in zip(axes, ALGORITHMS):
        trace = traces[algorithm]
        entered = {tuple(e["path"]) for e in trace["trace"] if e["event"] == "enter"}
        returns = {tuple(e["path"]): e for e in trace["trace"] if e["event"] in ("leaf", "return")}
        for path in nodes:
            x, y = x_positions[path], -len(path)
            visited = path in entered
            color = COLORS[algorithm] if visited else "#b8bdc3"
            if path:
                parent = path[:-1]
                axis.plot([x_positions[parent], x], [-len(parent), y], color=color,
                          linestyle="-" if visited else "--", linewidth=1.4, zorder=0)
            is_leaf = not nodes[path].get("children")
            marker = "s" if not is_leaf and len(path) % 2 == 0 else "o"
            axis.scatter([x], [y], s=740 if not is_leaf else 580, marker=marker,
                         color="white" if visited else "#eef0f2", edgecolors=color, linewidths=1.5, zorder=2)
            if is_leaf:
                label = str(nodes[path]["value"])
            else:
                event = returns.get(path)
                label = "?" if event is None else ({"LOWER": ">=", "UPPER": "<="}.get(event.get("bound"), "") + str(event["value"]))
            axis.text(x, y, label, ha="center", va="center", fontsize=10, color=color, zorder=3)
        axis.set_title(f"{LABELS[algorithm]}: root={trace['value']}\nVisits={trace['stats']['node_visits']}; leaves={trace['stats']['leaf_evals']}")
        axis.set_xlim(-0.65, max(x_positions.values()) + 0.65)
        axis.set_ylim(-case["depth"] - 0.5, 0.55)
        axis.axis("off")
    figure.suptitle("Official tree: identical root value with one leaf skipped", fontsize=14, weight="bold")
    figure.text(0.5, 0.035, "Square: MAX; circle: MIN or leaf. Gray/dashed: skipped. >= / <= marks a returned bound, not an exact subtree value.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.09, 1, 0.92))
    _save(figure, directory, "04_official_pruning")
    plt.close(figure)


def analyze_results(result_dir: Path) -> dict[str, Any]:
    result_dir = Path(result_dir).resolve()
    config = json.loads((result_dir / "config.json").read_text(encoding="utf-8-sig"))
    if config.get("status") != "completed":
        raise ValueError("仅可分析已完成的测量")
    _verify_sources(result_dir, config)
    rows = _read_rows(result_dir / "raw.csv")
    if rows[0]["run_id"] != config.get("run_id") or len(rows) != config.get("raw_rows"):
        raise ValueError("运行编号或原始行数与配置不符")
    _validate_batches(rows, config["batches"])
    instances = _aggregate_instances(rows)
    paired = _paired_instances(instances)
    if len(paired) != config.get("measured_instances"):
        raise ValueError("实例数与配置不符")
    if sum(r["batch_loops"] for r in rows) != config.get("total_timed_calls"):
        raise ValueError("正式计时调用数与配置不符")
    _, lookup = _verify_dataset(result_dir, paired, config)
    _verify_correctness(result_dir / "correctness.csv", instances)
    official, traces = _verify_official_trace(result_dir, instances, lookup)
    summary_rows, report = _group_summary(instances, paired, rows)
    report.update(run_id=config["run_id"], config_file="config.json", dataset_dir="dataset",
                  official_trace_figure="figures/04_official_pruning.png")
    _write_csv(result_dir / "instances.csv", instances)
    _write_csv(result_dir / "paired.csv", paired)
    _write_csv(result_dir / "summary.csv", summary_rows)
    directory = result_dir / "figures"
    directory.mkdir(exist_ok=True)
    plt = _setup_plotting()
    _depth_plot(plt, report["groups"], directory,
                (("node_visits", "Node visits (including root and leaves)"), ("leaf_evals", "Leaf evaluations")),
                "01_workload", "Minimax vs Alpha-Beta: search work across depths")
    _depth_plot(plt, report["groups"], directory,
                (("search_ms", "Recursive search per call (ms)"), ("api_ms", "Full API per call (ms, includes validation)")),
                "02_runtime", "Minimax vs Alpha-Beta: search and full interface costs")
    _paired_plot(plt, paired, directory)
    _official_plot(plt, official, traces, directory)
    (result_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    config.update(analysis_status="completed", analysis_completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  analysis_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (result_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="题三 Minimax 与 Alpha-Beta 的配对统计和图表")
    parser.add_argument("--results", type=Path, default=ROOT / "results/problem3_minimax_alpha_beta")
    args = parser.parse_args(argv)
    try:
        report = analyze_results(args.results)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2, f"分析失败：{error}\n")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
