"""题三访问顺序：按树嵌套汇总随机重排，并核对实际排列与完整成本。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from statistics import median
from typing import Any
from zipfile import ZipFile

from experiments.analyze_weighted_heuristics import distribution, _save, _setup_plotting


ROOT = Path(__file__).resolve().parent.parent
COUNTERS = ("node_visits", "leaf_evals", "cutoff_events", "pruned_branches")
TIMINGS = ("copy_ms", "oracle_ms", "order_ms", "prepare_ms", "search_ms", "query_ms", "total_ms")
META = ("source", "tree_file", "tree_sha256", "seed", "depth", "branch_factor", "node_count", "leaf_count", "reference_value")
FAMILIES = ("original", "random", "ideal")
PAIRED_METRICS = ("node_reduction", "leaf_reduction", "search_speedup", "query_speedup", "total_speedup",
                  "search_time_reduction", "query_time_reduction", "total_time_reduction")
COLORS = {"original": "#315d7a", "random": "#a97636", "ideal": "#21847b"}
LABELS = {"original": "Original order", "random": "Random order (tree median)", "ideal": "Exact-value order"}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("不能保存空的汇总表")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _optional_int(value):
    return None if value in (None, "", "None", "null") else int(value)


def _spec(variant: str):
    if variant in ("original", "ideal"):
        return variant, None
    if variant.startswith("random_") and len(variant) == 9 and variant[7:].isdigit() and 0 <= int(variant[7:]) < 10:
        return "random", int(variant[7:])
    raise ValueError("未知访问顺序变体")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    integers = ("depth", "branch_factor", "node_count", "leaf_count", "batch_index", "batch_loops", "order_in_group",
                "schedule_row", "root_value", "reference_value", "oracle_node_evals", *COUNTERS)
    required = {"run_id", "instance_id", "tree_file", "tree_sha256", "source", "seed", "variant", "ordering",
                "order_seed", "recipe_sha256", "correct", "batch_total_ms", *integers, *(f"{t}_mean" for t in TIMINGS)}
    rows, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("原始数据字段不完整")
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for field in integers:
                row[field] = int(row[field])
            row["seed"], row["order_seed"] = _optional_int(row["seed"]), _optional_int(row["order_seed"])
            if (row["ordering"], row["order_seed"]) != _spec(row["variant"]):
                raise ValueError("变体与排序类型或种子不一致")
            for field in (*(f"{t}_mean" for t in TIMINGS), "batch_total_ms"):
                row[field] = float(row[field])
                if not math.isfinite(row[field]) or row[field] < 0:
                    raise ValueError("计时必须为有限非负数")
            if any(row[f"{t}_mean"] <= 0 for t in ("search_ms", "query_ms", "total_ms")) or row["batch_total_ms"] <= 0:
                raise ValueError("搜索、查询和完整批时必须为正数")
            if str(row["correct"]).lower() not in ("true", "1") or row["root_value"] != row["reference_value"]:
                raise ValueError("根值未通过独立参考校验")
            row["correct"] = True
            if row["source"] not in ("random", "official"):
                raise ValueError("未知树来源")
            if (row["depth"] < 0 or row["branch_factor"] < 0 or row["node_count"] < 1 or row["leaf_count"] < 1
                    or row["batch_index"] < 0 or row["batch_loops"] < 1 or row["order_in_group"] < 0 or row["schedule_row"] < 0):
                raise ValueError("规模、批次或执行顺序无效")
            if not (1 <= row["leaf_evals"] <= row["leaf_count"] <= row["node_count"]
                    and row["leaf_evals"] <= row["node_visits"] <= row["node_count"]):
                raise ValueError("访问与叶评计数超出完整树规模")
            if (any(row[f] < 0 for f in COUNTERS) or row["cutoff_events"] > row["pruned_branches"]
                    or row["cutoff_events"] > row["node_visits"] - row["leaf_evals"]
                    or row["pruned_branches"] > row["node_count"] - row["node_visits"]):
                raise ValueError("剪枝计数不符")
            expected_evals = row["node_count"] if row["ordering"] == "ideal" else 0
            if row["oracle_node_evals"] != expected_evals:
                raise ValueError("真实子树值预计算未完整计入或错误地用于其他策略")
            if row["ordering"] == "original" and any(row[f"{t}_mean"] != 0 for t in ("copy_ms", "oracle_ms", "order_ms", "prepare_ms")):
                raise ValueError("原顺序不应有复制或排序准备成本")
            if row["ordering"] == "random" and row["oracle_ms_mean"] != 0:
                raise ValueError("随机顺序不应使用真实子树值")
            tolerance = 1e-8 * max(1.0, row["total_ms_mean"])
            if (row["search_ms_mean"] > row["query_ms_mean"] + tolerance
                    or row["prepare_ms_mean"] + row["query_ms_mean"] > row["total_ms_mean"] + tolerance
                    or sum(row[f"{t}_mean"] for t in ("copy_ms", "oracle_ms", "order_ms")) > row["prepare_ms_mean"] + tolerance):
                raise ValueError("准备、搜索、查询和完整时间范围矛盾")
            if not math.isclose(row["batch_total_ms"], row["total_ms_mean"] * row["batch_loops"], rel_tol=1e-8):
                raise ValueError("完整批时与每次平均时间不符")
            key = row["instance_id"], row["variant"], row["batch_index"]
            if key in seen:
                raise ValueError("重复的树、顺序和批次记录")
            seen.add(key)
            rows.append(row)
    if not rows or len({r["run_id"] for r in rows}) != 1:
        raise ValueError("数据为空或混合多个运行编号")
    return rows


def _validate_batches(rows, batches, variants, group_orders=None):
    if type(batches) is not int or batches < 1 or len(set(variants)) != len(variants):
        raise ValueError("批次或顺序列表无效")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["instance_id"]].append(row)
    for entries in grouped.values():
        if {(r["variant"], r["batch_index"]) for r in entries} != {(v, b) for v in variants for b in range(batches)}:
            raise ValueError("每棵树的顺序与批次不完整")
        if len(entries) != len(variants) * batches or len({r["batch_loops"] for r in entries}) != 1:
            raise ValueError("同树各顺序必须具有相同批内调用次数")
        for b in range(batches):
            block = [r for r in entries if r["batch_index"] == b]
            if {r["order_in_group"] for r in block} != set(range(len(variants))) or len({r["schedule_row"] for r in block}) != 1:
                raise ValueError("执行位置或顺序组编号不匹配")
            if group_orders is not None:
                index = block[0]["schedule_row"]
                actual = [r["variant"] for r in sorted(block, key=lambda r: r["order_in_group"])]
                if not 0 <= index < len(group_orders) or actual != list(group_orders[index]):
                    raise ValueError("实际执行顺序与配置顺序表不符")


def _aggregate_instances(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["instance_id"], r["variant"]].append(r)
    instances = []
    stable = (*META, "ordering", "order_seed", "recipe_sha256", "root_value", "oracle_node_evals", *COUNTERS)
    for (instance_id, variant), entries in sorted(grouped.items()):
        first = entries[0]
        if any(r[f] != first[f] for r in entries for f in stable):
            raise ValueError("同树同顺序的输入、配方、结果或确定性计数变化")
        row = {"instance_id": instance_id, "variant": variant, **{f: first[f] for f in stable}, "n_batches": len(entries), "correct": True}
        for timing in TIMINGS:
            stats = distribution([r[f"{timing}_mean"] for r in entries])
            row.update({f"{timing}_{part}": stats[part] for part in ("median", "q1", "q3")})
        instances.append(row)
    return instances


def _paired_instances(instances, variants):
    grouped = defaultdict(dict)
    for row in instances:
        if row["variant"] in grouped[row["instance_id"]]:
            raise ValueError("同树具有重复顺序汇总")
        grouped[row["instance_id"]][row["variant"]] = row
    paired = []
    for instance_id, current in sorted(grouped.items()):
        if set(current) != set(variants) or "original" not in current:
            raise ValueError("每棵树缺少完整顺序配对")
        base = current["original"]
        if any(r[f] != base[f] for r in current.values() for f in (*META, "root_value")):
            raise ValueError("不同顺序未使用相同输入或根值")
        for variant in variants:
            if variant == "original":
                continue
            r = current[variant]
            entry = {"instance_id": instance_id, "variant": variant, "ordering": r["ordering"], "order_seed": r["order_seed"],
                     **{f: r[f] for f in META}, "recipe_sha256": r["recipe_sha256"], "root_value": r["root_value"]}
            for f in COUNTERS:
                entry[f], entry[f"original_{f}"] = r[f], base[f]
            for t in TIMINGS:
                entry[t], entry[f"original_{t}"] = r[f"{t}_median"], base[f"{t}_median"]
            entry["node_reduction"] = 1 - r["node_visits"] / base["node_visits"]
            entry["leaf_reduction"] = 1 - r["leaf_evals"] / base["leaf_evals"]
            for prefix in ("search", "query", "total"):
                entry[f"{prefix}_speedup"] = base[f"{prefix}_ms_median"] / r[f"{prefix}_ms_median"]
                entry[f"{prefix}_time_reduction"] = 1 - r[f"{prefix}_ms_median"] / base[f"{prefix}_ms_median"]
            paired.append(entry)
    return paired


def _tree_families(instances, paired):
    by_tree = defaultdict(list)
    by_pair = defaultdict(list)
    for r in instances:
        by_tree[r["instance_id"]].append(r)
    for r in paired:
        by_pair[r["instance_id"], r["ordering"]].append(r)
    result = []
    for key, all_orders in sorted(by_tree.items()):
        for family in FAMILIES:
            selected = [r for r in all_orders if r["ordering"] == family]
            if not selected:
                raise ValueError("缺少某类顺序")
            first = selected[0]
            entry = {"instance_id": key, "family": family, **{f: first[f] for f in META}, "root_value": first["root_value"], "n_orders": len(selected)}
            for f in (*COUNTERS, "oracle_node_evals"):
                entry[f] = median(r[f] for r in selected)
            for t in TIMINGS:
                entry[t] = median(r[f"{t}_median"] for r in selected)
            ps = by_pair[key, family]
            for metric in PAIRED_METRICS:
                entry[metric] = median(r[metric] for r in ps) if ps else (1.0 if metric.endswith("speedup") else 0.0)
            entry["node_visits_min"] = min(r["node_visits"] for r in selected)
            entry["node_visits_max"] = max(r["node_visits"] for r in selected)
            entry["node_reduction_min"] = min((r["node_reduction"] for r in ps), default=0.0)
            entry["node_reduction_max"] = max((r["node_reduction"] for r in ps), default=0.0)
            for metric, better in (("node_reduction", 0), ("query_speedup", 1), ("total_speedup", 1)):
                values = [r[metric] for r in ps] or [better]
                for outcome, predicate in (("better", lambda x: x > better), ("same", lambda x: x == better), ("worse", lambda x: x < better)):
                    entry[f"{metric}_{outcome}_orders"] = sum(predicate(x) for x in values)
            result.append(entry)
    return result


def _group_summary(tree_rows, paired, raw):
    groups, flat = [], []
    metrics = (*COUNTERS, "oracle_node_evals", *TIMINGS, "node_visits_min", "node_visits_max", "node_reduction_min", "node_reduction_max")
    for source, depth in sorted({(r["source"], r["depth"]) for r in tree_rows}):
        chosen = [r for r in tree_rows if (r["source"], r["depth"]) == (source, depth)]
        group = {"source": source, "depth": depth, "n_trees": len({r["instance_id"] for r in chosen})}
        for family in FAMILIES:
            selected = [r for r in chosen if r["family"] == family]
            stats = {m: distribution([r[m] for r in selected]) for m in metrics}
            current = {"n_trees": len(selected), "n_orders_per_tree": selected[0]["n_orders"], **stats}
            flat_row = {"source": source, "depth": depth, "family": family, "n_trees": len(selected), "n_orders_per_tree": selected[0]["n_orders"]}
            for m, values in stats.items():
                flat_row.update({f"{m}_{part}": v for part, v in values.items()})
            if family != "original":
                pair_stats = {m: distribution([r[m] for r in selected]) for m in PAIRED_METRICS}
                for prefix in ("search", "query", "total"):
                    pair_stats[f"faster_{prefix}_trees"] = sum(r[f"{prefix}_speedup"] > 1 for r in selected)
                    pair_stats[f"same_{prefix}_trees"] = sum(r[f"{prefix}_speedup"] == 1 for r in selected)
                    pair_stats[f"slower_{prefix}_trees"] = sum(r[f"{prefix}_speedup"] < 1 for r in selected)
                pair_stats["fewer_node_trees"] = sum(r["node_reduction"] > 0 for r in selected)
                pair_stats["same_node_trees"] = sum(r["node_reduction"] == 0 for r in selected)
                pair_stats["more_node_trees"] = sum(r["node_reduction"] < 0 for r in selected)
                seeds = [r for r in paired if (r["source"], r["depth"], r["ordering"]) == (source, depth, family)]
                pair_stats["n_order_comparisons"] = len(seeds)
                pair_stats["order_comparisons_note"] = "Nested orders on the same trees; these are not independent tree samples."
                for metric, boundary in (("node_reduction", 0), ("search_speedup", 1), ("query_speedup", 1), ("total_speedup", 1)):
                    pair_stats[f"better_{metric}_orders"] = sum(r[metric] > boundary for r in seeds)
                    pair_stats[f"same_{metric}_orders"] = sum(r[metric] == boundary for r in seeds)
                    pair_stats[f"worse_{metric}_orders"] = sum(r[metric] < boundary for r in seeds)
                current["paired"] = pair_stats
            group[family] = current
            flat.append(flat_row)
        groups.append(group)
    ids = {r["instance_id"] for r in tree_rows}
    random_ids = {r["instance_id"] for r in tree_rows if r["source"] == "random"}
    return flat, {
        "aggregation": {
            "batch": "For each tree and variant: median of per-call batch means; deterministic counters counted once.",
            "random_family": "For each tree: median across random orders; then aggregate independent trees within source and depth.",
            "paired": "Calculate each random-order/original ratio per tree first, median these ratios within the tree, then aggregate trees. Not ratios of family or group medians.",
            "sample_size": "Random permutations are nested repeats; 30 trees x 10 orders remains 30 independent input trees per depth.",
            "time_scope": "search excludes full validation; query includes validation; total includes fresh copying, full exact-value preprocessing if ideal, ordering, query and loop/result handling.",
            "uncertainty": "Q1-Q3 are linearly interpolated across-tree dispersion, not confidence intervals.",
            "oracle": "Exact-value order is an oracle reference; full true-value preparation is charged on every timed call.",
        },
        "overall": {"n_unique_trees": len(ids), "n_random_trees": len(random_ids), "n_official_trees": len(ids - random_ids),
                    "n_raw_batches": len(raw), "n_variant_results": len({(r['instance_id'], r['variant']) for r in raw}),
                    "n_random_order_comparisons": sum(r["source"] == "random" and r["ordering"] == "random" for r in paired),
                    "total_timed_calls": sum(r["batch_loops"] for r in raw), "all_correct": all(r["correct"] for r in raw)},
        "groups": groups,
    }


def _verify_sources(result_dir, config):
    required = {"src/game_tree.py", "experiments/game_tree_dataset.py", "experiments/game_tree_ordering.py",
                "experiments/exp3_visit_order.py", "experiments/analyze_visit_order.py"}
    hashes = config.get("source_sha256", {})
    if not required <= set(hashes):
        raise ValueError("缺少本轮源码哈希")
    for relative, expected in hashes.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"源码已变化：{relative}；请使用本轮归档源码")
    path = result_dir / "source_baseline.zip"
    if sha256(path.read_bytes()).hexdigest() != config.get("source_baseline_sha256"):
        raise ValueError("源码归档哈希不符")
    with ZipFile(path) as archive:
        if set(archive.namelist()) != set(hashes) or len(archive.namelist()) != len(hashes):
            raise ValueError("归档成员不完整或重复")
        if any(sha256(archive.read(n)).hexdigest() != h for n, h in hashes.items()):
            raise ValueError("归档源码内容不符")


def _verify_dataset(result_dir, instances, config):
    from experiments.game_tree_dataset import load_dataset
    manifest_path = result_dir / "dataset/manifest.json"
    if sha256(manifest_path.read_bytes()).hexdigest() != config.get("dataset_manifest_sha256"):
        raise ValueError("保存的数据集manifest哈希不符")
    _, cases = load_dataset(result_dir / "dataset")
    lookup = {c["instance_id"]: c for c in cases}
    if set(lookup) != {r["instance_id"] for r in instances}:
        raise ValueError("保存的树与测量实例覆盖不一致")
    for row in instances:
        if any(row[f] != lookup[row["instance_id"]][f] for f in META):
            raise ValueError("实际输入与测量元数据不一致")
    return lookup


def _verify_proof(path, instances):
    expected = {(r["instance_id"], r["variant"]): r for r in instances}
    fields = (*META, "ordering", "order_seed", "recipe_sha256", "root_value", "oracle_node_evals", *COUNTERS)
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"instance_id", "variant", "correct", "minimax_value", "minimax_node_visits", "minimax_leaf_evals", *fields}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("顺序正确性证明字段不完整")
        for raw in reader:
            key = raw["instance_id"], raw["variant"]
            if key not in expected or key in seen or str(raw["correct"]).lower() not in ("true", "1"):
                raise ValueError("顺序证明重复、未知或校验失败")
            row = expected[key]
            for f in fields:
                actual = _optional_int(raw[f]) if f in ("seed", "order_seed") else int(raw[f]) if isinstance(row[f], int) else raw[f]
                if actual != row[f]:
                    raise ValueError("顺序证明与测量结果不一致")
            if (int(raw["minimax_value"]), int(raw["minimax_node_visits"]), int(raw["minimax_leaf_evals"])) != (row["reference_value"], row["node_count"], row["leaf_count"]):
                raise ValueError("重排后的Minimax未完整访问或根值变化")
            seen.add(key)
    if seen != set(expected):
        raise ValueError("顺序证明未覆盖全部变体")


def _recipe_tree(data, permutations, variant):
    """从保存的原树与实际排列独立重建字典树，同时验证排序规则。"""
    from experiments.game_tree_dataset import reference_tree
    source = data.get("tree_structure", data)
    nodes, pending = [], [(source, 0)]
    while pending:
        node, depth = pending.pop()
        nodes.append((node, depth))
        pending.extend((c, depth + 1) for c in reversed(node.get("children", [])))
    inner = [(n, d) for n, d in nodes if n.get("children")]
    if not isinstance(permutations, (list, tuple)) or len(permutations) != len(inner):
        raise ValueError("实际排列未覆盖每个内部节点")
    values, clones = {}, {}
    for node, depth in reversed(nodes):
        cs = node.get("children", [])
        values[id(node)] = (max if depth % 2 == 0 else min)(values[id(c)] for c in cs) if cs else node["value"]
        clones[id(node)] = {k: v for k, v in node.items() if k != "children"}
        if cs:
            clones[id(node)]["children"] = [clones[id(c)] for c in cs]
    ordering, seed = _spec(variant)
    rng = random.Random(seed)
    for (node, depth), order in zip(inner, permutations):
        n = len(node["children"])
        if not isinstance(order, (list, tuple)) or any(type(i) is not int for i in order) or sorted(order) != list(range(n)):
            raise ValueError("保存的孩子索引不是合法排列")
        want = list(range(n))
        if ordering == "random":
            rng.shuffle(want)
        elif ordering == "ideal":
            want.sort(key=lambda i: values[id(node["children"][i])], reverse=depth % 2 == 0)
        if list(order) != want:
            raise ValueError("保存排列不符合固定随机种子或准确子树值规则")
        clones[id(node)]["children"] = [clones[id(c)] for i in order for c in [node["children"][i]]]
    reconstructed = {"tree_structure": clones[id(source)]}
    original, ordered = reference_tree(data), reference_tree(reconstructed)
    if original != ordered:
        raise ValueError("实际重排改变了根值、节点或结构规模")
    return reconstructed


def _verify_recipes(path, instances, lookup):
    expected = {(r["instance_id"], r["variant"]): r for r in instances}
    official, seen = {}, set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            key = raw["instance_id"], raw["variant"]
            if key not in expected or key in seen:
                raise ValueError("实际排列存在重复或未知变体")
            row = expected[key]
            if (raw["ordering"], raw["order_seed"]) != (row["ordering"], row["order_seed"]):
                raise ValueError("排列配方的类型或种子不符")
            digest = sha256(json.dumps(raw["permutations"], separators=(",", ":")).encode("utf-8")).hexdigest()
            if digest != raw["recipe_sha256"] or digest != row["recipe_sha256"]:
                raise ValueError("实际排列哈希不符")
            for field in META:
                if field in raw and raw[field] != row[field]:
                    raise ValueError("排列配方的输入元数据不符")
            data = _recipe_tree(lookup[key[0]]["tree_data"], raw["permutations"], key[1])
            if row["source"] == "official" and key[1] in ("original", "random_00", "ideal"):
                official[key[1]] = data
            seen.add(key)
    if seen != set(expected):
        raise ValueError("实际排列未覆盖全部测量变体")
    return official


def _tree_paths(data):
    result, pending = {}, [(data.get("tree_structure", data), ())]
    while pending:
        node, path = pending.pop()
        result[path] = node
        pending.extend((c, path + (i,)) for i, c in reversed(list(enumerate(node.get("children", [])))))
    return result


def _verify_traces(result_dir, instances, official):
    traces = json.loads((result_dir / "official_trace.json").read_text(encoding="utf-8-sig"))
    required = {"original", "random_00", "ideal"}
    if set(traces) != required or set(official) != required:
        raise ValueError("官方轨迹缺少原始、随机或理想顺序")
    expected = {r["variant"]: r for r in instances if r["source"] == "official"}
    for variant in required:
        result, row = traces[variant], expected[variant]
        if result["value"] != row["root_value"] or any(result["stats"][f] != row[f] for f in COUNTERS):
            raise ValueError("官方轨迹与实测根值或计数不一致")
        paths = _tree_paths(official[variant])
        entered = [tuple(e["path"]) for e in result["trace"] if e["event"] == "enter"]
        leaves = [e for e in result["trace"] if e["event"] == "leaf"]
        cuts = [e for e in result["trace"] if e["event"] == "cutoff"]
        if len(entered) != len(set(entered)) or len(entered) != row["node_visits"] or not set(entered) <= set(paths):
            raise ValueError("官方访问轨迹不完整或重复")
        if (len(leaves) != row["leaf_evals"] or {tuple(e["path"]) for e in leaves} != {p for p in entered if not paths[p].get("children")}
                or any(e["value"] != paths[tuple(e["path"])]["value"] for e in leaves)):
            raise ValueError("官方叶值或叶评轨迹不符")
        skipped = [tuple(p) for e in cuts for p in e["skipped_paths"]]
        if len(cuts) != row["cutoff_events"] or len(skipped) != row["pruned_branches"] or not set(skipped) <= set(paths):
            raise ValueError("官方剪枝轨迹计数不符")
        if any(p[:len(skip)] == skip for p in entered for skip in skipped):
            raise ValueError("官方已剪子树与已访问路径矛盾")
    return traces


def _depth_plot(plt, groups, directory, metrics, name, title):
    selected = sorted((g for g in groups if g["source"] == "random"), key=lambda g: g["depth"])
    if not selected:
        return
    depths = [g["depth"] for g in selected]
    figure, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 4.9), squeeze=False)
    for axis, (metric, label) in zip(axes[0], metrics):
        for family in FAMILIES:
            vals = [g[family][metric] for g in selected]
            axis.plot(depths, [v["median"] for v in vals], "o-", color=COLORS[family], label=LABELS[family], linewidth=2)
            axis.fill_between(depths, [v["q1"] for v in vals], [v["q3"] for v in vals], color=COLORS[family], alpha=.14)
        axis.set_xticks(depths)
        axis.set_xlabel("Depth (root-to-leaf edges)")
        axis.set_ylabel(label)
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=.18)
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.text(.5, .015, "Random orders: median within each tree first. Lines: across-tree medians; bands: Q1-Q3, not confidence intervals.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .075, 1, .92))
    _save(figure, directory, name)
    plt.close(figure)


def _paired_plot(plt, tree_rows, directory):
    selected = [r for r in tree_rows if r["source"] == "random" and r["family"] != "original"]
    depths = sorted({r["depth"] for r in selected})
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.1))
    for axis, metric, label in zip(axes, ("node_reduction", "search_speedup", "total_speedup"), ("Node reduction vs original (%)", "Search speedup (original / ordered)", "Complete speedup incl. preparation")):
        scale, boundary = (100, 0) if metric == "node_reduction" else (1, 1)
        for family, shift in (("random", -.14), ("ideal", .14)):
            for i, depth in enumerate(depths):
                vals = [r[metric] * scale for r in selected if r["family"] == family and r["depth"] == depth]
                axis.scatter([i + shift + .07 * math.sin((j + 1) * 2.39996) for j in range(len(vals))], vals, s=17, alpha=.4, color=COLORS[family])
                s = distribution(vals)
                axis.errorbar(i + shift, s["median"], yerr=[[s["median"] - s["q1"]], [s["q3"] - s["median"]]], fmt="D", color=COLORS[family], capsize=4, label=LABELS[family] if i == 0 else None)
        axis.axhline(boundary, color="#89919b", linestyle="--", linewidth=1)
        axis.set_xticks(range(len(depths)), [str(d) for d in depths])
        axis.set_xlabel("Depth")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=.15)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Visit order: search savings and complete single-query cost", fontsize=14, weight="bold")
    figure.text(.5, .015, "Each dot: one tree; random ratios are medians across its fixed random orders. Speedup > 1 means faster.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .075, 1, .92))
    _save(figure, directory, "03_paired_speedups")
    plt.close(figure)


def _variability_plot(plt, tree_rows, directory):
    selected = [r for r in tree_rows if r["source"] == "random" and r["family"] == "random"]
    depths = sorted({r["depth"] for r in selected})
    figure, axes = plt.subplots(1, len(depths), figsize=(6 * len(depths), 4.9), squeeze=False)
    for axis, depth in zip(axes[0], depths):
        chosen = sorted((r for r in selected if r["depth"] == depth), key=lambda r: r["node_reduction"])
        for i, r in enumerate(chosen):
            axis.plot([i, i], [100 * r["node_reduction_min"], 100 * r["node_reduction_max"]], color=COLORS["random"], alpha=.5)
            axis.scatter(i, 100 * r["node_reduction"], color=COLORS["random"], s=18)
        axis.axhline(0, color="#89919b", linestyle="--", linewidth=1)
        axis.set_title(f"Depth {depth}; {len(chosen)} trees")
        axis.set_xlabel("Tree (sorted by median benefit)")
        axis.set_ylabel("Node reduction vs original (%)")
        axis.grid(axis="y", alpha=.15)
    figure.suptitle("The same tree can benefit or lose from a random visit order", fontsize=14, weight="bold")
    figure.text(.5, .015, "One line per tree: min-max over fixed random orders. Dot: within-tree median. Positive means fewer visits than original.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .075, 1, .92))
    _save(figure, directory, "04_order_variability")
    plt.close(figure)


def _official_plot(plt, official, traces, directory):
    variants = ("original", "random_00", "ideal")
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.3))
    for axis, variant in zip(axes, variants):
        nodes, result = _tree_paths(official[variant]), traces[variant]
        leaves = [p for p, node in nodes.items() if not node.get("children")]
        x = {p: float(i) for i, p in enumerate(leaves)}
        for path in sorted(nodes, key=len, reverse=True):
            if path not in x:
                cs = [path + (i,) for i in range(len(nodes[path]["children"]))]
                x[path] = sum(x[c] for c in cs) / len(cs)
        entered = {tuple(e["path"]) for e in result["trace"] if e["event"] == "enter"}
        returns = {tuple(e["path"]): e for e in result["trace"] if e["event"] in ("leaf", "return")}
        color = COLORS[_spec(variant)[0]]
        for path, node in nodes.items():
            visited = path in entered
            c = color if visited else "#b8bdc3"
            if path:
                axis.plot([x[path[:-1]], x[path]], [-len(path[:-1]), -len(path)], color=c, linestyle="-" if visited else "--", linewidth=1.2, zorder=0)
            leaf = not node.get("children")
            axis.scatter(x[path], -len(path), marker="s" if not leaf and len(path) % 2 == 0 else "o", s=480 if leaf else 620,
                         color="white" if visited else "#eef0f2", edgecolors=c, zorder=2)
            event = returns.get(path)
            label = str(node["value"]) if leaf else "?" if event is None else {"LOWER": ">=", "UPPER": "<="}.get(event.get("bound"), "") + str(event["value"])
            axis.text(x[path], -len(path), label, ha="center", va="center", color=c, fontsize=9, zorder=3)
        axis.set_title(f"{variant}: root={result['value']}\nVisits={result['stats']['node_visits']}; leaves={result['stats']['leaf_evals']}")
        axis.set_xlim(-.65, max(x.values()) + .65)
        axis.set_ylim(-max(map(len, nodes)) - .5, .5)
        axis.axis("off")
    figure.suptitle("Official tree: identical root value under different visit orders", fontsize=14, weight="bold")
    figure.text(.5, .025, "Gray/dashed: skipped. Square: MAX; circle: MIN or leaf. >= / <= indicates a returned bound. Exact-value order requires full preprocessing.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .09, 1, .91))
    _save(figure, directory, "05_official_pruning")
    plt.close(figure)


def analyze_results(result_dir: Path) -> dict[str, Any]:
    result_dir = Path(result_dir).resolve()
    config = json.loads((result_dir / "config.json").read_text(encoding="utf-8-sig"))
    if config.get("status") != "completed":
        raise ValueError("仅可分析已完成的测量")
    variants = config["variants"]
    random_count = config["random_count"]
    if variants != ["original", *(f"random_{i:02d}" for i in range(random_count)), "ideal"]:
        raise ValueError("配置顺序列表与随机种子数不符")
    _verify_sources(result_dir, config)
    rows = _read_rows(result_dir / "raw.csv")
    if len(rows) != config.get("raw_rows") or rows[0]["run_id"] != config.get("run_id"):
        raise ValueError("运行编号或原始行数与配置不符")
    _validate_batches(rows, config["batches"], variants, config.get("group_orders"))
    instances = _aggregate_instances(rows)
    paired = _paired_instances(instances, variants)
    trees = _tree_families(instances, paired)
    if len({r["instance_id"] for r in rows}) != config.get("measured_instances") or sum(r["batch_loops"] for r in rows) != config.get("total_timed_calls"):
        raise ValueError("实例数或正式调用次数与配置不符")
    lookup = _verify_dataset(result_dir, instances, config)
    _verify_proof(result_dir / "proof.csv", instances)
    if sha256((result_dir / "recipes.jsonl.gz").read_bytes()).hexdigest() != config.get("recipes_sha256"):
        raise ValueError("实际排列归档文件哈希与配置不符")
    official = _verify_recipes(result_dir / "recipes.jsonl.gz", instances, lookup)
    traces = _verify_traces(result_dir, instances, official)
    flat, report = _group_summary(trees, paired, rows)
    report.update(run_id=config["run_id"], variants=variants, random_count=random_count, dataset_dir="dataset")
    _write_csv(result_dir / "instances.csv", instances)
    _write_csv(result_dir / "paired.csv", paired)
    _write_csv(result_dir / "tree_summary.csv", trees)
    _write_csv(result_dir / "summary.csv", flat)
    directory = result_dir / "figures"
    directory.mkdir(exist_ok=True)
    plt = _setup_plotting()
    _depth_plot(plt, report["groups"], directory, (("node_visits", "Node visits, including root and leaves"), ("leaf_evals", "Leaf evaluations")),
                "01_nodes", "Alpha-Beta work under different visit orders")
    _depth_plot(plt, report["groups"], directory, (("search_ms", "Search per call (ms)"), ("query_ms", "Query incl. validation (ms)"), ("total_ms", "Complete incl. preparation (ms)")),
                "02_costs", "Visit order: preparation must be charged to each single query")
    _paired_plot(plt, trees, directory)
    _variability_plot(plt, trees, directory)
    _official_plot(plt, official, traces, directory)
    (result_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    config.update(analysis_status="completed", analysis_completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  analysis_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (result_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="题三访问顺序的嵌套配对统计与图表")
    parser.add_argument("--results", type=Path, default=ROOT / "results/problem3_visit_order")
    args = parser.parse_args(argv)
    try:
        report = analyze_results(args.results)
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(2, f"分析失败：{error}\n")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
