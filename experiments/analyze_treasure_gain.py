"""核验并分析题四即时收益排序实验保存的数据，不重新采集计时。"""

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
from zipfile import ZipFile

from experiments.analyze_weighted_heuristics import _save, _setup_plotting, distribution
from experiments.exp3_minimax_alpha_beta import COUNTERS, _write_csv, _write_json
from experiments.exp4_depths import load_input
from experiments.treasure_reference import reference_distances, reference_game


ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ("original", "gain")
TIMES = ("search_ms", "query_ms", "total_ms")


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _verify_sources(folder, config):
    hashes = config.get("source_sha256", {})
    required = {"src/problem4_treasure.py", "experiments/treasure_reference.py",
                "experiments/treasure_gain.py", "experiments/exp4_gain.py",
                "experiments/analyze_treasure_gain.py"}
    if not isinstance(hashes, dict) or not required <= hashes.keys():
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


def _validate_result(row, reference, depth):
    for name in ("depth", "best_value", "reference_value", *COUNTERS):
        row[name] = int(row[name])
    if (row["depth"] != depth or row["best_value"] != reference.value
            or row["reference_value"] != reference.value
            or row["best_target"] not in reference.best_targets
            or str(row["correct"]).lower() not in {"true", "1"}):
        raise ValueError("根值或首步未通过独立参考校验")
    if (any(row[name] < 0 for name in COUNTERS)
            or not 1 <= row["leaf_evals"] <= min(row["node_visits"], reference.leaf_count)
            or row["node_visits"] > reference.node_count
            or row["cutoff_events"] > row["pruned_branches"]
            or (row["cutoff_events"] == 0) != (row["pruned_branches"] == 0)):
        raise ValueError("搜索计数不合理")


def _verify_data(folder, config):
    if config.get("experiment") != "p4_gain_v1" or config.get("status") != "completed":
        raise ValueError("只能分析已完成的即时收益排序实验")
    if (config.get("depth") != 4 or config.get("variants") != list(VARIANTS)
            or type(config.get("batches")) is not int or config["batches"] < 1
            or type(config.get("batch_loops")) is not int or config["batch_loops"] < 1):
        raise ValueError("实验配置无效")
    input_file = (folder / config["input_file"]).resolve()
    if not input_file.is_relative_to(folder.resolve()) or not input_file.is_file():
        raise ValueError("输入文件路径无效")
    payload, case = load_input(input_file)
    if sha256(payload).hexdigest() != config.get("input_sha256"):
        raise ValueError("输入哈希不匹配")
    matrix = reference_distances(case["grid"], case["positions"])
    for name in ("distance_matrix.json", "dijkstra_reference.json"):
        if json.loads((folder / name).read_text(encoding="utf8")) != matrix:
            raise ValueError("距离矩阵与独立参考不符")
    reference = reference_game(matrix, case["treasure_values"], config["depth"])
    if (config.get("reference_value") != reference.value
            or config.get("reference_best_targets") != reference.best_targets):
        raise ValueError("独立参考与配置不符")
    proof = {}
    for row in _read_csv(folder / "proof.csv"):
        variant = row["variant"]
        if variant not in (*VARIANTS, "production") or variant in proof:
            raise ValueError("审计记录重复或含未知配置")
        _validate_result(row, reference, config["depth"])
        proof[variant] = row
    if set(proof) != {*VARIANTS, "production"}:
        raise ValueError("审计记录不完整")
    if any(proof["original"][f] != proof["production"][f]
           for f in ("best_target", "best_value", *COUNTERS)):
        raise ValueError("原顺序实验实现与生产实现的结果或搜索计数不一致")

    expected = {(v, b) for v in VARIANTS for b in range(config["batches"])}
    grouped, seen, positions = defaultdict(list), set(), defaultdict(dict)
    for row in _read_csv(folder / "raw.csv"):
        variant = row["variant"]
        for name in ("batch_index", "batch_loops", "order_in_pair"):
            row[name] = int(row[name])
        key = variant, row["batch_index"]
        if key not in expected or key in seen or row["run_id"] != config["run_id"]:
            raise ValueError("原始批次重复、缺少或混入其他运行")
        seen.add(key)
        _validate_result(row, reference, config["depth"])
        if any(row[f] != proof[variant][f] for f in ("best_target", "best_value", *COUNTERS)):
            raise ValueError("正式调用结果或搜索计数与审计不一致")
        if row["batch_loops"] != config["batch_loops"]:
            raise ValueError("配对配置需要相同批内调用次数")
        position = row["order_in_pair"]
        if position not in (0, 1) or position in positions[row["batch_index"]]:
            raise ValueError("批次执行位置重复或无效")
        positions[row["batch_index"]][position] = variant
        for name in (*[f"{t}_mean" for t in TIMES], "batch_total_ms"):
            row[name] = float(row[name])
            if not math.isfinite(row[name]) or row[name] <= 0:
                raise ValueError("时间必须为有限正数")
        if (row["search_ms_mean"] > row["query_ms_mean"] + 1e-8
                or row["query_ms_mean"] > row["total_ms_mean"] + 1e-8
                or not math.isclose(row["batch_total_ms"], row["total_ms_mean"] * row["batch_loops"], rel_tol=1e-8)):
            raise ValueError("计时范围或批时与均值矛盾")
        grouped[variant].append(row)
    if (seen != expected or config.get("raw_rows") != len(seen)
            or config.get("total_timed_calls") != len(seen) * config["batch_loops"]):
        raise ValueError("原始批次或计时调用数不完整")
    for batch in range(config["batches"]):
        expected_order = VARIANTS if batch % 2 == 0 else VARIANTS[::-1]
        if tuple(positions[batch].get(i) for i in (0, 1)) != expected_order:
            raise ValueError("两种配置的执行次序与规则不符")
    return reference, proof, grouped


def _summarize(reference, proof, grouped):
    rows = []
    for variant in VARIANTS:
        audit, records = proof[variant], grouped[variant]
        row = {"variant": variant, "batches": len(records), "batch_loops": records[0]["batch_loops"],
               "best_target": audit["best_target"], "best_value": audit["best_value"],
               **{name: audit[name] for name in COUNTERS}}
        for field in TIMES:
            row.update({f"{field}_{name}": number for name, number in
                        distribution([record[f"{field}_mean"] for record in records]).items()})
        rows.append(row)
    original, gain = rows
    compare = {
        "node_reduction": 1 - gain["node_visits"] / original["node_visits"],
        "leaf_reduction": 1 - gain["leaf_evals"] / original["leaf_evals"],
    }
    for field in TIMES:
        o, g = original[f"{field}_median"], gain[f"{field}_median"]
        compare[f"{field}_speedup"] = o / g
        compare[f"{field}_reduction"] = 1 - g / o
        original_batches = {r["batch_index"]: r[f"{field}_mean"] for r in grouped["original"]}
        gain_batches = {r["batch_index"]: r[f"{field}_mean"] for r in grouped["gain"]}
        ratios = [original_batches[b] / gain_batches[b] for b in sorted(original_batches)]
        compare[f"{field}_paired_speedup_median"] = median(ratios)
        compare[f"{field}_paired_speedup_distribution"] = distribution(ratios)
    analysis = {"experiment": "p4_gain_v1", "reference_value": reference.value,
                "reference_best_targets": reference.best_targets, "all_root_results_validated": True,
                "original": original, "gain": gain, "comparison": compare,
                "time_note": "Both the ratios of separate medians and median of per-batch paired ratios are recorded. Gain computation and sorting included in search and query times.",
                "scope_note": "One official map; repeated batches are not independent maps or confidence intervals. Distance matrix prepared before timing."}
    return rows, analysis


def _plot(rows, folder):
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
    labels = ["原顺序", "即时收益排序"]
    colors = ["#315d7a", "#268276"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8))
    positions = [0, 1]
    bars = ax1.bar(positions, [r["node_visits"] for r in rows], color=colors, width=.55)
    ax1.bar_label(bars, padding=3)
    ax1.set_xticks(positions, labels); ax1.set_ylabel("访问节点数（含根和叶）")
    medians = [r["total_ms_median"] for r in rows]
    error = [[r["total_ms_median"] - r["total_ms_q1"] for r in rows],
             [r["total_ms_q3"] - r["total_ms_median"] for r in rows]]
    ax2.bar(positions, medians, color=colors, width=.55)
    ax2.errorbar(positions, medians, yerr=error, fmt="none", color="#3a4348", capsize=4)
    ax2.set_xticks(positions, labels); ax2.set_ylabel("总时间（毫秒，中位数及 Q1–Q3）")
    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.suptitle("题四 H=4：即时收益动态排序对照", fontsize=13, weight="bold")
    fig.text(.5, .02, "计算收益和排序计入搜索及总时间；单地图重复批次并非独立样本。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .93))
    _save(fig, directory, "01_gain_work_and_time")
    plt.close(fig)


def analyze_results(result_dir, *, make_plots=True):
    folder = Path(result_dir).resolve()
    config = json.loads((folder / "config.json").read_text(encoding="utf8"))
    _verify_sources(folder, config)
    reference, proof, grouped = _verify_data(folder, config)
    rows, analysis = _summarize(reference, proof, grouped)
    analysis.update(run_id=config["run_id"], input_sha256=config["input_sha256"],
                    raw_sha256=sha256((folder / "raw.csv").read_bytes()).hexdigest(),
                    batches=config["batches"], total_timed_calls=config["total_timed_calls"],
                    time_scope=config["time_scope"])
    _write_csv(folder / "gain_summary.csv", rows)
    _write_json(folder / "gain_analysis.json", analysis)
    if make_plots:
        _plot(rows, folder)
    config.update(analysis_status="completed", analysis_completed_at_utc=datetime.now(timezone.utc).isoformat())
    _write_json(folder / "config.json", config)
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description="分析题四即时收益动态排序的已保存结果")
    parser.add_argument("--results", type=Path, default=ROOT / "results/problem4_gain")
    args = parser.parse_args(argv)
    result = analyze_results(args.results)
    print(json.dumps({key: result[key] for key in ("all_root_results_validated", "comparison")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
