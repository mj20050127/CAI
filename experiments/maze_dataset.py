"""题一对照实验的数据生成、独立 BFS 与路径检查。

随机地图先完整生成，再按 BFS 可达性进行拒绝采样；不会挖通道路。
所有接受和拒绝的候选地图均保存，以便检查条件抽样和复现数据来源。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import random
from typing import Any


GENERATOR_VERSION = "maze_dataset_v1"
Position = tuple[int, int]
Grid = Sequence[Sequence[str]]


def _dimensions(grid: Grid) -> tuple[int, int]:
    if isinstance(grid, (str, bytes)) or not isinstance(grid, Sequence) or not grid:
        raise ValueError("地图必须是非空矩形二维序列")
    first = grid[0]
    if not isinstance(first, Sequence) or isinstance(first, bytes) or not first:
        raise ValueError("地图必须是非空矩形二维序列")
    width = len(first)
    for row in grid:
        if not isinstance(row, Sequence) or isinstance(row, bytes) or len(row) != width:
            raise ValueError("地图每行长度必须相同")
        if any(not isinstance(cell, str) or cell not in {"S", "E", ".", "X"} for cell in row):
            raise ValueError("迷宫格只能是 S、E、. 或 X")
    return len(grid), width


def _coordinate(value: Sequence[int], rows: int, cols: int) -> Position:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise ValueError("坐标必须包含两个整数")
    row, col = value
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError("坐标越界")
    return row, col


def bfs_distance(grid: Grid, start: Sequence[int], end: Sequence[int]) -> int | None:
    """独立四邻接 BFS 的最短步数；墙上端点或不可达返回 None。

    该函数不导入待测 A* 的邻居生成、启发或代价函数，作为正确性参考。
    非矩形地图、非法符号、越界及非法坐标抛 ValueError。
    """
    rows, cols = _dimensions(grid)
    source = _coordinate(start, rows, cols)
    target = _coordinate(end, rows, cols)
    if grid[source[0]][source[1]] == "X" or grid[target[0]][target[1]] == "X":
        return None
    queue = deque([(source, 0)])
    seen = {source}
    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) == target:
            return distance
        for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            r, c = neighbor
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] != "X" and neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


def validate_path(
    grid: Grid,
    start: Sequence[int],
    end: Sequence[int],
    path: Sequence[Sequence[int]],
    expected_cost: int | None,
) -> None:
    """按独立参考步数核对返回路径；发现问题抛 ValueError。

    最短路径可以有多条，只检查端点、通行性、四邻接和最优步数。
    expected_cost=None 表示参考算法判定无路，此时 path 必须为空。
    """
    rows, cols = _dimensions(grid)
    source = _coordinate(start, rows, cols)
    target = _coordinate(end, rows, cols)
    if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
        raise ValueError("路径必须是坐标序列")
    if expected_cost is None:
        if len(path) != 0:
            raise ValueError("参考结果不可达，但待测算法返回了非空路径")
        return
    if type(expected_cost) is not int or expected_cost < 0:
        raise ValueError("参考代价必须是非负整数或 None")
    if not path:
        raise ValueError("参考结果可达，但待测算法返回空路径")
    coordinates = [_coordinate(position, rows, cols) for position in path]
    if coordinates[0] != source or coordinates[-1] != target:
        raise ValueError("路径起点或终点不符")
    if any(grid[row][col] == "X" for row, col in coordinates):
        raise ValueError("路径穿过障碍")
    for previous, current in zip(coordinates, coordinates[1:]):
        if abs(previous[0] - current[0]) + abs(previous[1] - current[1]) != 1:
            raise ValueError("路径相邻坐标必须是上下左右相邻格")
    if len(coordinates) - 1 != expected_cost:
        raise ValueError("路径步数与独立 BFS 最短步数不一致")


def map_sha256(start: Sequence[int], end: Sequence[int], grid: Grid) -> str:
    """仅对实际算法输入计算稳定哈希，不把统计或采样元数据混入哈希。"""
    canonical = json.dumps(
        {"start": list(start), "end": list(end), "grid": [list(row) for row in grid]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _generate_grid(size: int, density: float, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    grid = [["." for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for col in range(size):
            if (row, col) in ((0, 0), (size - 1, size - 1)):
                continue
            grid[row][col] = "X" if rng.random() < density else "."
    grid[0][0] = "S"
    grid[-1][-1] = "E"
    return grid


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def create_dataset(
    output_dir: Path,
    *,
    sizes: Sequence[int] = (21, 51, 101),
    count_per_size: int = 30,
    density: float = 0.2,
    master_seed: int = 20261021,
    max_attempts_per_size: int | None = None,
) -> dict[str, Any]:
    """创建可复现、保留拒绝候选的地图数据集并返回 manifest。

    第 k 种规模使用 Random(f'{master_seed}:{size}') 逐次抽取 64 位
    candidate seed；再以 Random(candidate_seed) 为除两个端点外的每格
    独立抽一次 Bernoulli(density)。attempt_index 从 1 开始。

    非空 maps 目录或已有 manifest 会拒绝覆盖。超过尝试上限时保存
    status=incomplete 的 manifest，再抛 RuntimeError；不会把残缺数据
    标作完成。原始地图及所有拒绝记录保持可追查。
    """
    if isinstance(sizes, (str, bytes)) or not isinstance(sizes, Sequence) or not sizes:
        raise ValueError("sizes 必须是非空的边长序列")
    dimensions = list(sizes)
    if any(type(size) is not int or size < 2 for size in dimensions):
        raise ValueError("地图边长必须是至少为 2 的整数")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("sizes 不能包含重复边长")
    if type(count_per_size) is not int or count_per_size <= 0:
        raise ValueError("每种规模的可达实例数必须是正整数")
    if isinstance(density, bool) or not isinstance(density, (int, float)) or not isfinite(density):
        raise ValueError("障碍概率必须是有限数值")
    if not 0 <= density < 1:
        raise ValueError("障碍概率必须满足 0 <= density < 1")
    if type(master_seed) is not int:
        raise ValueError("master_seed 必须是整数")
    attempt_limit = max(1000, count_per_size * 1000) if max_attempts_per_size is None else max_attempts_per_size
    if type(attempt_limit) is not int or attempt_limit < count_per_size:
        raise ValueError("最大尝试次数必须是至少等于每组目标实例数的整数")

    destination = Path(output_dir)
    manifest_path = destination / "manifest.json"
    maps_dir = destination / "maps"
    if manifest_path.exists() or (maps_dir.exists() and any(maps_dir.iterdir())):
        raise FileExistsError("数据集已存在或包含候选地图，请使用新的输出目录，避免覆盖已测输入")
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "status": "incomplete",
        "config": {
            "sizes": dimensions,
            "count_per_size": count_per_size,
            "density": float(density),
            "master_seed": master_seed,
            "max_attempts_per_size": attempt_limit,
            "seed_strategy": "Random(f'{master_seed}:{size}').getrandbits(64); each candidate uses Random(candidate_seed)",
            "obstacle_sampling": "one independent draw per non-endpoint cell; start and end forced open",
            "acceptance_rule": "independent four-neighbor BFS reaches end; no path carving",
            "observed_density_denominator": "size*size-2",
            "start_rule": "[0,0]",
            "end_rule": "[size-1,size-1]",
        },
        "accepted_files": [],
        "rejected_files": [],
        "groups": [],
    }

    for size in dimensions:
        seed_source = random.Random(f"{master_seed}:{size}")
        accepted = rejected = attempts = 0
        while accepted < count_per_size and attempts < attempt_limit:
            attempts += 1
            candidate_seed = seed_source.getrandbits(64)
            grid = _generate_grid(size, float(density), candidate_seed)
            start, end = [0, 0], [size - 1, size - 1]
            distance = bfs_distance(grid, start, end)
            reachable = distance is not None
            if reachable:
                accepted += 1
                instance_id = f"maze_{size:03d}_{accepted:03d}"
            else:
                rejected += 1
                instance_id = f"maze_{size:03d}_rejected_{attempts:04d}"
            obstacles = sum(cell == "X" for row in grid for cell in row)
            candidate = {
                "instance_id": instance_id,
                "size": size,
                "seed": candidate_seed,
                "attempt_index": attempts,
                "accepted_index": accepted if reachable else None,
                "accepted": reachable,
                "start": start,
                "end": end,
                "grid": grid,
                "bfs_cost": distance,
                "obstacle_count": obstacles,
                "observed_density": obstacles / (size * size - 2),
                "sha256": map_sha256(start, end, grid),
            }
            relative_path = Path("maps") / f"size_{size}" / f"attempt_{attempts:04d}_seed_{candidate_seed}.json"
            _write_json(destination / relative_path, candidate)
            key = "accepted_files" if reachable else "rejected_files"
            manifest[key].append(relative_path.as_posix())

        manifest["groups"].append(
            {"size": size, "accepted": accepted, "rejected": rejected, "attempts": attempts}
        )
        if accepted < count_per_size:
            manifest["failure"] = f"size={size}: only {accepted}/{count_per_size} reachable maps after {attempts} attempts"
            _write_json(manifest_path, manifest)
            raise RuntimeError(f"地图生成达到尝试上限；不完整记录已保存于 {manifest_path}")

    manifest["status"] = "complete"
    manifest["total_accepted"] = len(manifest["accepted_files"])
    manifest["total_rejected"] = len(manifest["rejected_files"])
    _write_json(manifest_path, manifest)
    return manifest
