"""题四独立参考：进入代价 Dijkstra，以及行动排列的分组回传。

不导入 src 中的网格、状态转移或博弈搜索函数。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import permutations
from math import factorial
import re


def grid_weights(grid):
    if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes)) or not grid:
        raise ValueError("地图必须为非空矩形")
    rows = []
    for row in grid:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or not row:
            raise ValueError("地图必须为非空矩形")
        weights = []
        for cell in row:
            if type(cell) is int and cell >= 0:
                weight = cell
            elif isinstance(cell, str) and cell.isascii() and cell.isdigit():
                weight = int(cell)
            elif cell == "X":
                weight = None
            elif isinstance(cell, str) and (cell in {"A", "B", "S", "E"} or re.fullmatch(r"T[1-9][0-9]*", cell)):
                weight = 1
            else:
                raise ValueError("地图含非法格子")
            weights.append(weight)
        rows.append(weights)
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("地图必须为矩形")
    return rows


def _point(value, weights):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(part) is not int for part in value)):
        raise ValueError("位置必须是两个整数")
    row, col = value
    if not (0 <= row < len(weights) and 0 <= col < len(weights[0])) or weights[row][col] is None:
        raise ValueError("位置越界或位于障碍")
    return row, col


def reference_distances(grid, positions):
    """原有向边 u→v 加 w(v)；任何目标都不免费。"""
    weights = grid_weights(grid)
    if not isinstance(positions, Mapping) or not positions:
        raise ValueError("位置映射不能为空")
    points = {name: _point(point, weights) for name, point in positions.items()}
    matrix = {}
    for name, start in points.items():
        distances = {start: 0}
        queue = [(0, start)]
        while queue:
            cost, (row, col) = heappop(queue)
            if cost != distances[row, col]:
                continue
            for other in ((row + 1, col), (row, col - 1), (row - 1, col), (row, col + 1)):
                r, c = other
                if not (0 <= r < len(weights) and 0 <= c < len(weights[0])) or weights[r][c] is None:
                    continue
                candidate = cost + weights[r][c]
                if candidate < distances.get(other, float("inf")):
                    distances[other] = candidate
                    heappush(queue, (candidate, other))
        if any(point not in distances for point in points.values()):
            raise ValueError("有位置对不可达；本实验不定义跳过回合")
        matrix[name] = {target: distances[point] for target, point in points.items()}
    return matrix


def validate_path(grid, start, end, path, expected):
    weights = grid_weights(grid)
    start, end = _point(start, weights), _point(end, weights)
    points = [_point(point, weights) for point in path]
    if not points or points[0] != start or points[-1] != end:
        raise ValueError("路径端点不符")
    total = 0
    for previous, point in zip(points, points[1:]):
        if sum(abs(a - b) for a, b in zip(previous, point)) != 1:
            raise ValueError("路径必须四邻接")
        total += weights[point[0]][point[1]]
    if total != expected:
        raise ValueError("路径累计进入代价不符，终点也必须计费")
    return total


def _validate_game(matrix, values, depth):
    if type(depth) is not int or depth < 0:
        raise ValueError("深度必须为非负整数")
    if not isinstance(matrix, Mapping) or not {"A", "B"} <= set(matrix):
        raise ValueError("矩阵必须包含 A、B")
    names = set(matrix)
    for source, row in matrix.items():
        if not isinstance(row, Mapping) or set(row) != names:
            raise ValueError("距离必须是完整方阵")
        if any(type(v) is not int or v < 0 for v in row.values()) or row[source] != 0:
            raise ValueError("距离必须非负且对角线为零")
    if not isinstance(values, Mapping) or any(
        not isinstance(name, str) or name in {"A", "B"} or name not in names or type(value) is not int
        for name, value in values.items()
    ):
        raise ValueError("宝藏标签或价值无效")


def action_ledger(matrix, values, sequence):
    """逐行动回放，不因经过其他宝藏而改变收集集合。"""
    a, b, score_a, score_b = "A", "B", 0, 0
    remaining = set(values)
    rows = []
    for index, target in enumerate(sequence):
        if target not in remaining:
            raise ValueError("行动重复取得或指定未知宝藏")
        player = "A" if index % 2 == 0 else "B"
        source = a if player == "A" else b
        cost = matrix[source][target]
        gain = values[target] - cost
        if player == "A":
            a, score_a = target, score_a + gain
        else:
            b, score_b = target, score_b + gain
        remaining.remove(target)
        rows.append({"step": index + 1, "player": player, "source": source, "target": target,
                     "distance": cost, "treasure_value": values[target], "gain": gain,
                     "score_a": score_a, "score_b": score_b, "utility": score_a - score_b})
    return rows


@dataclass
class ReferenceGame:
    value: int
    best_targets: list[str]
    action_values: dict[str, int]
    principal_variation: list[str]
    action_variations: dict[str, list[str]]
    node_count: int
    leaf_count: int


def reference_game(matrix, values, depth):
    """枚举所有终止排列，再逐层分组；保留所有首步精确值与并列动作。

    horizon=min(depth,n)。不从 Alpha-Beta 的有限窗口回传推断子树精确值。
    """
    _validate_game(matrix, values, depth)
    order = tuple(values)
    horizon = min(depth, len(order))
    tables = {}
    for sequence in permutations(order, horizon):
        ledger = action_ledger(matrix, values, sequence)
        tables[sequence] = ledger[-1]["utility"] if ledger else 0
    for level in range(horizon - 1, -1, -1):
        for prefix in permutations(order, level):
            children = [tables[prefix + (target,)] for target in order if target not in prefix]
            tables[prefix] = (max if level % 2 == 0 else min)(children)
    def continuation(prefix):
        path = prefix
        while len(path) < horizon:
            choice = next(target for target in order if target not in path
                          and tables[path + (target,)] == tables[path])
            path += (choice,)
        return list(path)
    root = tables[()]
    actions = {target: tables[target,] for target in order} if horizon else {}
    best = [target for target in order if actions.get(target) == root] if horizon else []
    count = sum(factorial(len(order)) // factorial(len(order) - level) for level in range(horizon + 1))
    return ReferenceGame(root, best, actions, continuation(()),
                         {target: continuation((target,)) for target in actions},
                         count, factorial(len(order)) // factorial(len(order) - horizon))
