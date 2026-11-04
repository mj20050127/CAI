"""核验并汇总题四固定全局优先级实验；分析已有记录，不重新计时。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
from itertools import permutations
import json
import math
from pathlib import Path
import random
from statistics import mean, median
from zipfile import ZipFile

from experiments.analyze_weighted_heuristics import _save, _setup_plotting, distribution
from experiments.exp3_minimax_alpha_beta import COUNTERS, _write_csv, _write_json
from experiments.exp4_depths import load_input
from experiments.treasure_reference import reference_distances, reference_game


ROOT = Path(__file__).resolve().parent.parent
TIMES = ("search_ms", "query_ms", "total_ms")


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _file_in(folder, name):
    path = (folder / name).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise ValueError("输入文件路径无效")
    return path


def _verify_sources(folder, config):
    hashes = config.get("source_sha256", {})
    if not {"src/problem4_treasure.py", "experiments/exp4_order.py",
            "experiments/analyze_treasure_order.py", "experiments/treasure_reference.py"} <= set(hashes):
        raise ValueError("本次源码快照不完整")
    archive = folder / "source_baseline.zip"
    if sha256(archive.read_bytes()).hexdigest() != config.get("source_baseline_sha256"):
        raise ValueError("源码归档的哈希不匹配")
    with ZipFile(archive) as stream:
        if len(stream.namelist()) != len(hashes) or set(stream.namelist()) != set(hashes):
            raise ValueError("源码归档成员不匹配")
        for name, digest in hashes.items():
            local = (ROOT / name).resolve()
            if (not local.is_relative_to(ROOT.resolve()) or not local.is_file()
                    or sha256(local.read_bytes()).hexdigest() != digest
                    or sha256(stream.read(name)).hexdigest() != digest):
                raise ValueError(f"源码与快照不匹配：{name}")


def _validate_result(row, ref, algorithm):
    for name in ("depth", "best_value", "reference_value", *COUNTERS):
        row[name] = int(row[name])
    if (row["depth"] != 4 or row["best_value"] != ref.value
            or row["reference_value"] != ref.value
            or row["best_target"] != ref.best_targets[0]
            or str(row["correct"]).lower() not in {"true", "1"}):
        raise ValueError("求解结果未通过独立参考校验")
    if (any(row[name] < 0 for name in COUNTERS)
            or not 1 <= row["leaf_evals"] <= min(row["node_visits"], ref.leaf_count)
            or row["node_visits"] > ref.node_count
            or row["cutoff_events"] > row["pruned_branches"]
            or (row["cutoff_events"] == 0) != (row["pruned_branches"] == 0)):
        raise ValueError("搜索计数不合理")
    if algorithm == "minimax" and (row["node_visits"] != ref.node_count
            or row["leaf_evals"] != ref.leaf_count or row["cutoff_events"] or row["pruned_branches"]):
        raise ValueError("Minimax 的搜索计数与完整树不符")


def _verify_data(folder, config):
    if config.get("experiment") != "p4_order_v1" or config.get("status") != "completed":
        raise ValueError("只能分析已完成的题四优先级实验")
    if config.get("depth") != 4 or type(config.get("batches")) is not int or config["batches"] < 1:
        raise ValueError("深度或批次数不符合实验配置")
    payload, case = load_input(_file_in(folder, config["input_file"]))
    if sha256(payload).hexdigest() != config.get("input_sha256"):
        raise ValueError("输入哈希不匹配")
    values = case["treasure_values"]
    if len(values) != 4 or any(">" in name for name in values):
        raise ValueError("实验须使用四个宝藏")
    order_ids = [">".join(p) for p in permutations(values)]
    original = ">".join(values)
    if config.get("orders") != order_ids or config.get("original_priority") != original:
        raise ValueError("固定优先级列表与输入不一致")
    matrix = reference_distances(case["grid"], case["positions"])
    for name in ("distance_matrix.json", "dijkstra_reference.json"):
        if json.loads((folder / name).read_text(encoding="utf8")) != matrix:
            raise ValueError("距离矩阵与独立参考不符")
    refs = {order: reference_game(matrix, {k: values[k] for k in order.split(">")}, 4)
            for order in order_ids}
    proof = {}
    for row in _read_csv(folder / "proof.csv"):
        key = row["priority_id"], row["algorithm"]
        if key in proof or key[0] not in refs or key[1] not in {"minimax", "alpha_beta"}:
            raise ValueError("审计记录重复或包含未知配置")
        _validate_result(row, refs[key[0]], key[1])
        proof[key] = row
    expected_proof = {(p, a) for p in order_ids for a in ("minimax", "alpha_beta")}
    if set(proof) != expected_proof:
        raise ValueError("审计记录不完整")

    variants = [(p, "alpha_beta") for p in order_ids] + [(original, "minimax")]
    expected = {(p, a, b) for p, a in variants for b in range(config["batches"])}
    grouped, seen, positions = defaultdict(list), set(), defaultdict(dict)
    for row in _read_csv(folder / "raw.csv"):
        p, algorithm = row["priority_id"], row["algorithm"]
        for field in ("batch_index", "batch_loops", "execution_position"):
            row[field] = int(row[field])
        key = p, algorithm, row["batch_index"]
        if key not in expected or key in seen or row["run_id"] != config["run_id"]:
            raise ValueError("原始批次重复、缺少或混入其他运行")
        seen.add(key)
        _validate_result(row, refs[p], algorithm)
        audit = proof[p, algorithm]
        if any(row[field] != audit[field] for field in ("best_value", "best_target", *COUNTERS)):
            raise ValueError("原始搜索计数与审计记录不一致")
        if row["batch_loops"] != config["batch_loops"] or row["batch_loops"] < 1:
            raise ValueError("各配置需要相同的批内调用次数")
        position = row["execution_position"]
        if position in positions[row["batch_index"]]:
            raise ValueError("批次中存在重复的执行位置")
        positions[row["batch_index"]][position] = (p, algorithm)
        for field in (*[f"{t}_mean" for t in TIMES], "batch_total_ms"):
            row[field] = float(row[field])
            if not math.isfinite(row[field]) or row[field] <= 0:
                raise ValueError("时间必须为有限正数")
        if (row["search_ms_mean"] > row["query_ms_mean"] + 1e-8
                or row["query_ms_mean"] > row["total_ms_mean"] + 1e-8
                or not math.isclose(row["batch_total_ms"], row["total_ms_mean"] * row["batch_loops"], rel_tol=1e-8)):
            raise ValueError("计时范围或批时与均值矛盾")
        grouped[p, algorithm].append(row)
    if seen != expected or config.get("raw_rows") != len(seen) or config.get("total_timed_calls") != len(seen) * config["batch_loops"]:
        raise ValueError("原始批次或计时调用数不完整")
    rng = random.Random(config["order_seed"])
    calibration_order = variants.copy()
    rng.shuffle(calibration_order)
    for batch in range(config["batches"]):
        shuffled = variants.copy()
        rng.shuffle(shuffled)
        if [positions[batch].get(i) for i in range(len(variants))] != shuffled:
            raise ValueError("批次执行顺序与随机种子不匹配")
    return order_ids, original, refs, proof, grouped


def _summarize(order_ids, original, refs, proof, grouped):
    variants = [(p, "alpha_beta") for p in order_ids] + [(original, "minimax")]
    mm_nodes = proof[original, "minimax"]["node_visits"]
    mm_leaves = proof[original, "minimax"]["leaf_evals"]
    base_nodes = proof[original, "alpha_beta"]["node_visits"]
    base_total = median(r["total_ms_mean"] for r in grouped[original, "alpha_beta"])
    rows = []
    for priority, algorithm in variants:
        runs, audit = grouped[priority, algorithm], proof[priority, algorithm]
        row = {"priority_id": priority, "algorithm": algorithm, "is_original_priority": priority == original,
               "batches": len(runs), "batch_loops": runs[0]["batch_loops"],
               "best_target": audit["best_target"], "best_value": audit["best_value"],
               **{name: audit[name] for name in COUNTERS},
               "node_reduction_vs_minimax": 1 - audit["node_visits"] / mm_nodes,
               "leaf_reduction_vs_minimax": 1 - audit["leaf_evals"] / mm_leaves,
               "node_reduction_vs_original_ab": 1 - audit["node_visits"] / base_nodes}
        for field in TIMES:
            stats = distribution([r[f"{field}_mean"] for r in runs])
            row.update({f"{field}_{name}": value for name, value in stats.items()})
        row["total_speedup_vs_original_ab"] = base_total / row["total_ms_median"]
        rows.append(row)
    alpha_rows = [r for r in rows if r["algorithm"] == "alpha_beta"]
    minimum = min(r["node_visits"] for r in alpha_rows)
    maximum = max(r["node_visits"] for r in alpha_rows)
    histogram = dict(sorted(Counter(r["node_visits"] for r in alpha_rows).items()))
    analysis = {"experiment": "p4_order_v1", "orders": len(order_ids),
                "all_root_results_validated": True, "reference_value": refs[original].value,
                "reference_best_targets": refs[original].best_targets,
                "original_priority": original, "original_alpha_beta": next(r for r in alpha_rows if r["priority_id"] == original),
                "minimax_baseline": next(r for r in rows if r["algorithm"] == "minimax"),
                "nodes": {"minimum": minimum, "maximum": maximum,
                          "mean": mean(r["node_visits"] for r in alpha_rows),
                          "median": median(r["node_visits"] for r in alpha_rows),
                          "histogram": histogram,
                          "best_priorities": [r["priority_id"] for r in alpha_rows if r["node_visits"] == minimum],
                          "worst_priorities": [r["priority_id"] for r in alpha_rows if r["node_visits"] == maximum]},
                "time_note": "Batch means summarized separately by configuration; ratios are ratios of median times, not medians of paired ratios.",
                "scope_note": "One official map; 24 fixed global priorities, not all per-node orders. Priority dictionaries and distance matrix are prepared outside timing."}
    return rows, analysis


def _plot(rows, analysis, folder):
    plt = _setup_plotting()
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False
    directory = folder / "figures"
    directory.mkdir(exist_ok=True)
    items = sorted((r for r in rows if r["algorithm"] == "alpha_beta"),
                   key=lambda r: (r["node_visits"], r["priority_id"]))
    ys = list(range(len(items)))
    labels = [r["priority_id"].replace(">", " → ") for r in items]
    colors = ["#c85c42" if r["is_original_priority"] else "#268276" for r in items]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 11), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 1.1]})
    ax1.barh(ys, [r["node_visits"] for r in items], color=colors, alpha=.84)
    ax1.axvline(analysis["minimax_baseline"]["node_visits"], color="#5d6470", linestyle="--", label="Minimax")
    ax1.axvline(analysis["original_alpha_beta"]["node_visits"], color="#c85c42", linestyle=":", label="原优先级")
    ax1.set_yticks(ys, labels); ax1.invert_yaxis(); ax1.set_xlabel("访问节点数（含根和叶）"); ax1.legend(frameon=False)
    medians = [r["total_ms_median"] for r in items]
    errors = [[r["total_ms_median"] - r["total_ms_q1"] for r in items],
              [r["total_ms_q3"] - r["total_ms_median"] for r in items]]
    ax2.errorbar(medians, ys, xerr=errors, fmt="o", color="#315d7a", ecolor="#9babb4", capsize=2, markersize=3)
    ax2.axvline(analysis["original_alpha_beta"]["total_ms_median"], color="#c85c42", linestyle=":")
    ax2.set_xlabel("每次总时间（毫秒；中位数及 Q1–Q3）")
    for ax in (ax1, ax2): ax.grid(axis="x", alpha=.18); ax.set_axisbelow(True)
    fig.suptitle("题四：24种固定全局优先级，H=4", fontsize=14, weight="bold")
    fig.text(.5, .015, "时间不含建距离矩阵和准备优先级；单张地图的重复批次并非独立地图或置信区间。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96)); _save(fig, directory, "01_order_nodes_time"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    histogram = analysis["nodes"]["histogram"]
    axes[0].bar(list(histogram), list(histogram.values()), color="#268276", width=.8)
    axes[0].set_xlabel("Alpha-Beta 访问节点数"); axes[0].set_ylabel("优先级数量")
    axes[1].scatter([r["node_visits"] for r in items], [r["leaf_evals"] for r in items], c=colors, s=32, alpha=.8)
    baseline = analysis["original_alpha_beta"]
    axes[1].scatter([baseline["node_visits"]], [baseline["leaf_evals"]], color="#c85c42", s=65,
                    edgecolors="white", linewidths=.8, zorder=5)
    axes[1].set_xlabel("访问节点数"); axes[1].set_ylabel("叶子评估次数")
    for ax in axes: ax.grid(alpha=.18); ax.set_axisbelow(True)
    fig.suptitle("访问顺序与搜索工作量", fontsize=14, weight="bold")
    fig.text(.5, .015, "每点对应一种固定全局优先级；相同计数的点重叠，红色表示原优先级。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95)); _save(fig, directory, "02_order_distribution"); plt.close(fig)


def analyze_results(result_dir, *, make_plots=True):
    """验证已完成的原始实验、生成汇总，按需绘制图表；可重复执行。"""
    folder = Path(result_dir).resolve()
    config = json.loads((folder / "config.json").read_text(encoding="utf8"))
    _verify_sources(folder, config)
    order_ids, original, refs, proof, grouped = _verify_data(folder, config)
    rows, analysis = _summarize(order_ids, original, refs, proof, grouped)
    analysis.update(run_id=config["run_id"], input_sha256=config["input_sha256"],
                    raw_sha256=sha256((folder / "raw.csv").read_bytes()).hexdigest(),
                    batches=config["batches"], total_timed_calls=config["total_timed_calls"],
                    time_scope=config["time_scope"])
    _write_csv(folder / "order_summary.csv", rows)
    _write_json(folder / "analysis.json", analysis)
    if make_plots:
        _plot(rows, analysis, folder)
    config.update(analysis_status="completed", analysis_completed_at_utc=datetime.now(timezone.utc).isoformat())
    _write_json(folder / "config.json", config)
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description="分析题四24种固定全局优先级的已保存结果")
    parser.add_argument("--results", type=Path, default=ROOT / "results/problem4_order")
    args = parser.parse_args(argv)
    result = analyze_results(args.results)
    print(json.dumps({key: result[key] for key in ("orders", "all_root_results_validated", "nodes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
