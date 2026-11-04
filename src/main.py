"""命令行入口：在工作区运行 python -m src.main --case all。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .game_tree import evaluate_tree, tree_from_dict
from .problem1_maze import search_maze
from .problem2_terrain import search_weighted
from .problem4_treasure import build_distance_matrix, evaluate_treasure


ROOT = Path(__file__).resolve().parent.parent


def read_case(path: Path) -> dict[str, Any]:
    """输入使用 UTF-8 JSON；utf-8-sig 也兼容 Windows 编辑器的 BOM。"""
    with path.open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("输入 JSON 的最外层必须是对象。")
    return value


def coordinate(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise ValueError("坐标必须是含两个整数的数组，例如 [0, 0]。")
    return value[0], value[1]


def run_problem(
    problem: int,
    case: dict[str, Any],
    *,
    algorithm: str,
    include_stats: bool = False,
    trace: bool = False,
    depth: int | None = None,
) -> Any:
    """适配 JSON 输入，核心算法和题目纯函数均不依赖命令行。"""
    details = include_stats or trace
    if problem in (1, 2):
        start, end = coordinate(case["start"]), coordinate(case["end"])
        search = search_maze if problem == 1 else search_weighted
        result = search(start, end, case["grid"], trace=trace)
        answer = result.path if problem == 1 else (
            result.cost if result.cost is not None else -1
        )
        if not details:
            return answer
        payload = {
            "answer": answer,
            "path": result.path,
            "cost": result.cost,
            "stats": asdict(result.stats),
        }
        if trace:
            payload["expanded_order"] = result.expanded_order
        return payload

    algorithms = ("minimax", "alpha-beta") if algorithm == "both" else (algorithm,)
    if problem == 3:
        root = tree_from_dict(case)
        payload = {}
        for name in algorithms:
            result = evaluate_tree(root, use_alpha_beta=name == "alpha-beta", trace=trace)
            if details:
                entry = {"value": result.value, "stats": asdict(result.stats)}
                if trace:
                    entry["trace"] = result.trace
                payload[name] = entry
            else:
                payload[name] = result.value
        return payload if algorithm == "both" else payload[algorithm]

    positions = {name: coordinate(pos) for name, pos in case["positions"].items()}
    matrix = build_distance_matrix(case["grid"], positions)
    depths = [depth] if depth is not None else case.get("depths", [2, 4])
    if not isinstance(depths, list) or not depths:
        raise ValueError("depths 必须是非空的非负整数数组。")
    if any(type(item) is not int or item < 0 for item in depths):
        raise ValueError("搜索深度必须是非负整数。")
    decisions = {}
    for horizon in depths:
        per_algorithm = {}
        for name in algorithms:
            result = evaluate_treasure(
                matrix,
                case["treasure_values"],
                horizon,
                use_alpha_beta=name == "alpha-beta",
                trace=trace,
            )
            entry = {"best_target": result.best_target, "best_value": result.best_value}
            if details:
                entry["stats"] = asdict(result.stats)
            if trace:
                entry["trace"] = result.trace
            per_algorithm[name] = entry
        decisions[str(horizon)] = (
            per_algorithm if algorithm == "both" else per_algorithm[algorithm]
        )
    return {
        "distance_labels": list(matrix),
        "distance_matrix": matrix,
        "decisions": decisions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="实验二：四题基础实现与官方用例")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", choices=["all"], help="运行全部官方用例")
    selection.add_argument("--problem", type=int, choices=[1, 2, 3, 4], help="只运行指定题目")
    parser.add_argument("--input", type=Path, help="指定题目的 JSON 输入文件")
    parser.add_argument("--algorithm", choices=["minimax", "alpha-beta", "both"])
    parser.add_argument("--depth", type=int, help="题四的搜索深度，覆盖输入中的 depths")
    parser.add_argument("--stats", action="store_true", help="附带计数和单次求解耗时")
    parser.add_argument("--trace", action="store_true", help="附带搜索轨迹（也包含统计）")
    parser.add_argument("--output", type=Path, help="同时将 JSON 结果保存到文件")
    args = parser.parse_args(argv)
    run_all = args.problem is None
    if args.input is not None and run_all:
        parser.error("--input 需要同时指定 --problem。")
    if args.depth is not None and args.problem != 4:
        parser.error("--depth 仅用于 --problem 4。")
    if args.depth is not None and args.depth < 0:
        parser.error("--depth 不能为负。")
    if args.algorithm is not None and args.problem in (1, 2):
        parser.error("--algorithm 仅适用于博弈搜索（题三、题四）。")

    algorithm = args.algorithm or ("both" if run_all or args.problem == 4 else "alpha-beta")
    try:
        if run_all:
            result = {
                f"problem{number}": run_problem(
                    number,
                    read_case(ROOT / "cases" / f"problem{number}.json"),
                    algorithm=algorithm,
                    include_stats=args.stats,
                    trace=args.trace,
                )
                for number in range(1, 5)
            }
        else:
            path = args.input or ROOT / "cases" / f"problem{args.problem}.json"
            result = run_problem(
                args.problem,
                read_case(path),
                algorithm=algorithm,
                include_stats=args.stats,
                trace=args.trace,
                depth=args.depth,
            )
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
        parser.exit(2, f"输入或运行错误：{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
