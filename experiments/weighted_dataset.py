"""题二的随机地形与独立正确性参考，不导入待测 A* 的任何函数。

反向图保留原边权：原边 u→v 的权为进入 v 的代价，反边 v→u 仍加
w(v)，不能在反向搜索时误加前驱 u 的权值。终点免费只作用于指定终点。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from heapq import heappop, heappush
import json
from math import isclose, isfinite
from pathlib import Path
import random
from typing import Any


GENERATOR_VERSION = "weighted_dataset_v1"
Position = tuple[int, int]
Grid = Sequence[Sequence[str | int]]


def _cell_weight(cell: str | int) -> int | None:
    if type(cell) is int:
        if cell < 0:
            raise ValueError("进入代价不能为负")
        return cell
    if isinstance(cell, str):
        if cell == "X":
            return None
        if cell in {"S", "E"}:
            return 1
        if cell.isascii() and cell.isdigit():
            return int(cell)
    raise ValueError("地形格必须为非负整数、ASCII 数字字符串、S、E 或 X")


def _weights(grid: Grid) -> list[list[int | None]]:
    if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes)) or not grid:
        raise ValueError("地形必须是非空矩形二维序列")
    first = grid[0]
    if not isinstance(first, Sequence) or isinstance(first, bytes) or not first:
        raise ValueError("地形必须是非空矩形二维序列")
    width = len(first)
    result = []
    for row in grid:
        if not isinstance(row, Sequence) or isinstance(row, bytes) or len(row) != width:
            raise ValueError("地形每行长度必须相同")
        result.append([_cell_weight(cell) for cell in row])
    return result


def _coordinate(value: Sequence[int], rows: int, cols: int) -> Position:
    if (
        not isinstance(value, Sequence) or isinstance(value, (str, bytes))
        or len(value) != 2 or any(type(part) is not int for part in value)
    ):
        raise ValueError("坐标必须包含两个整数")
    row, col = value
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError("坐标越界")
    return row, col


def reference_to_goal(
    grid: Grid, end: Sequence[int], *, goal_free: bool = True
) -> dict[Position, int]:
    """独立反向 Dijkstra，返回每个可达格到指定终点的最小进入代价。

    不可到达终点的格不在结果中，终点本身为 0。原边 u→v 权为 w(v)，
    因此从当前 v 向前驱 u 松弛时加当前 v 的权；E 免费时向它的前驱加 0。
    普通格允许 0，支持障碍；非法地图或墙上终点抛 ValueError。
    """
    if type(goal_free) is not bool:
        raise ValueError("goal_free 必须是布尔值")
    weights = _weights(grid)
    rows, cols = len(weights), len(weights[0])
    goal = _coordinate(end, rows, cols)
    if weights[goal[0]][goal[1]] is None:
        raise ValueError("终点不能在障碍上")
    distances = {goal: 0}
    queue = [(0, goal[0], goal[1])]
    while queue:
        distance, row, col = heappop(queue)
        current = row, col
        if distance != distances[current]:
            continue
        entering = 0 if goal_free and current == goal else weights[row][col]
        assert entering is not None
        for previous in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            r, c = previous
            if not (0 <= r < rows and 0 <= c < cols) or weights[r][c] is None:
                continue
            candidate = distance + entering
            if candidate < distances.get(previous, float("inf")):
                distances[previous] = candidate
                heappush(queue, (candidate, r, c))
    return distances


def validate_weighted_path(
    grid: Grid,
    start: Sequence[int],
    end: Sequence[int],
    path: Sequence[Sequence[int]],
    expected_cost: int | None,
    *,
    goal_free: bool = True,
) -> None:
    """独立检查首尾、邻接、障碍和累计进入代价，允许不同等价最短路径。

    expected_cost 由外部独立参考给出；None 代表不可达，此时路径必须为空。
    起点不收费，之后每次进入格收费，指定终点可免费；不把格子数当代价。
    """
    if type(goal_free) is not bool:
        raise ValueError("goal_free 必须是布尔值")
    weights = _weights(grid)
    rows, cols = len(weights), len(weights[0])
    source, goal = _coordinate(start, rows, cols), _coordinate(end, rows, cols)
    if weights[source[0]][source[1]] is None or weights[goal[0]][goal[1]] is None:
        raise ValueError("起点和终点不能在障碍上")
    if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
        raise ValueError("路径必须是坐标序列")
    if expected_cost is None:
        if len(path) != 0:
            raise ValueError("参考结果不可达，但待测结果返回非空路径")
        return
    if type(expected_cost) is not int or expected_cost < 0:
        raise ValueError("参考代价必须是非负整数或 None")
    if not path:
        raise ValueError("参考结果可达，但待测结果返回空路径")
    coordinates = [_coordinate(point, rows, cols) for point in path]
    if coordinates[0] != source or coordinates[-1] != goal:
        raise ValueError("路径起点或终点不一致")
    total = 0
    for previous, current in zip(coordinates, coordinates[1:]):
        if abs(previous[0] - current[0]) + abs(previous[1] - current[1]) != 1:
            raise ValueError("路径只能沿四方向逐格移动")
        entering = weights[current[0]][current[1]]
        if entering is None:
            raise ValueError("路径穿过障碍")
        total += 0 if goal_free and current == goal else entering
    if total != expected_cost:
        raise ValueError(f"路径累计进入代价 {total} 与参考最优代价 {expected_cost} 不一致")


def input_sha256(case: Mapping[str, Any]) -> str:
    """对实际 start/end/grid 输入作规范化哈希，不混入采样统计。"""
    payload = {"start": list(case["start"]), "end": list(case["end"]), "grid": [list(row) for row in case["grid"]]}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def create_weighted_dataset(
    output_dir: Path,
    *,
    size: int = 31,
    count: int = 30,
    weights: Sequence[int] = (1, 3, 8),
    probabilities: Sequence[float] = (0.7, 0.2, 0.1),
    master_seed: int = 20261022,
) -> dict[str, Any]:
    """生成固定规模的无障碍整数地形并保存每张实际网格及最优代价。

    生成器只接受正整数普通格权，使主实验满足“除终点外进入代价≥1”。
    oracle 自身另支持零权，供边界校验使用。所有随机格独立抽样；无障碍
    因此无需拒绝采样、路径修补或地形筛选。已有非空数据目录拒绝覆盖。
    """
    if type(size) is not int or size < 2:
        raise ValueError("地图边长必须是至少为 2 的整数")
    if type(count) is not int or count < 1:
        raise ValueError("地图数量必须是正整数")
    if type(master_seed) is not int:
        raise ValueError("主种子必须是整数")
    if not isinstance(weights, Sequence) or isinstance(weights, (str, bytes)) or not weights:
        raise ValueError("weights 必须是非空权值序列")
    values = list(weights)
    if any(type(weight) is not int or weight < 1 for weight in values) or len(set(values)) != len(values):
        raise ValueError("普通格权值必须是不重复的正整数")
    if not isinstance(probabilities, Sequence) or isinstance(probabilities, (str, bytes)):
        raise ValueError("probabilities 必须是概率序列")
    chances = list(probabilities)
    if len(chances) != len(values) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not isfinite(value) or value < 0 for value in chances
    ):
        raise ValueError("权值与概率必须等长，概率必须是非负有限数")
    if not isclose(sum(chances), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("概率之和必须为 1")

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("数据目录非空，请使用新的输出目录以免覆盖已测输入")
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "status": "incomplete",
        "config": {
            "size": size, "count": count, "weights": values, "probabilities": chances,
            "master_seed": master_seed,
            "seed_strategy": "Random(f'weighted:{master_seed}:{size}').getrandbits(64); each candidate uses Random(candidate_seed).choices(weights, probabilities)",
            "start_rule": "[0,0]", "end_rule": "[size-1,size-1]",
            "obstacles": False, "goal_free": True, "start_reentry_cost": 1,
            "sampling_rule": "independent draw per non-endpoint cell; no filtering or path carving",
            "weight_proportion_denominator": "size*size-2",
            "reference": "independent reverse Dijkstra with original entering edge weights",
        },
        "accepted_files": [],
        "n_instances": 0,
    }
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest)
    seed_source = random.Random(f"weighted:{master_seed}:{size}")
    used_seeds: set[int] = set()
    for index in range(1, count + 1):
        seed = seed_source.getrandbits(64)
        while seed in used_seeds:
            seed = seed_source.getrandbits(64)
        used_seeds.add(seed)
        rng = random.Random(seed)
        terrain = iter(rng.choices(values, weights=chances, k=size * size - 2))
        grid: list[list[str | int]] = []
        for row in range(size):
            line = []
            for col in range(size):
                if (row, col) == (0, 0):
                    line.append("S")
                elif (row, col) == (size - 1, size - 1):
                    line.append("E")
                else:
                    line.append(next(terrain))
            grid.append(line)
        start, end = [0, 0], [size - 1, size - 1]
        exact = reference_to_goal(grid, end)
        instance_id = f"weighted_{size:03d}_{index:03d}"
        totals = {str(weight): sum(cell == weight for line in grid for cell in line) for weight in values}
        case = {
            "instance_id": instance_id, "size": size, "seed": seed,
            "start": start, "end": end, "grid": grid,
            "best_cost": exact[(0, 0)],
            "weight_counts": totals,
            "observed_proportions": {key: number / (size * size - 2) for key, number in totals.items()},
        }
        case["sha256"] = input_sha256(case)
        relative = Path("maps") / f"{instance_id}_seed_{seed}.json"
        _write_json(destination / relative, case)
        manifest["accepted_files"].append(relative.as_posix())
        manifest["n_instances"] += 1
        _write_json(manifest_path, manifest)
    manifest["status"] = "complete"
    _write_json(manifest_path, manifest)
    return manifest
