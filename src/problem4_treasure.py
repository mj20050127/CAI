"""题四：先用 A* 计算距离，再用 Minimax/Alpha-Beta 选择宝藏。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import inf
from numbers import Integral
from time import perf_counter_ns
from typing import Any

from .grid import Grid, grid_search
from .stats import GameStats


@dataclass(frozen=True)
class TreasureState:
    """完整博弈状态；位置是距离矩阵中的标签，不是网格坐标。"""

    pos_a: str
    pos_b: str
    remaining: frozenset[str]
    score_a: int = 0
    score_b: int = 0
    turn: str = "A"


@dataclass
class TreasureResult:
    """根节点的精确决策；轨迹中的内部搜索值可能只是界。"""

    best_target: str | None
    best_value: int
    stats: GameStats
    trace: list[dict[str, Any]]


def _integer(value: Any, label: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label}必须是整数")
    number = int(value)
    if nonnegative and number < 0:
        raise ValueError(f"{label}不能为负")
    return number


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}必须是非空字符串")
    return value


def build_distance_matrix(
    grid: Grid,
    positions: Mapping[str, Sequence[int]],
) -> dict[str, dict[str, int]]:
    """以 A* 计算所有有序位置对的最小进入代价。

    A、B 和宝藏格的进入代价均为 1，本题没有免费终点。
    对角线为 0；其余每个方向独立搜索，不假定矩阵对称。
    positions 至少包含 A、B；额外标签可用于更多宝藏的后续实验。
    非法输入或有位置对不可达时抛 ValueError，不擅自定义跳过回合。
    """
    if not isinstance(positions, Mapping) or not {"A", "B"} <= positions.keys():
        raise ValueError("positions 必须是至少包含 A、B 的位置映射")

    points: dict[str, tuple[int, int]] = {}
    for label, position in positions.items():
        name = _name(label, "位置标签")
        if (
            not isinstance(position, Sequence)
            or isinstance(position, (str, bytes))
            or len(position) != 2
        ):
            raise ValueError(f"位置 {name} 必须是两个整数构成的坐标")
        row = _integer(position[0], f"位置 {name} 的行号")
        col = _integer(position[1], f"位置 {name} 的列号")
        points[name] = row, col

    matrix: dict[str, dict[str, int]] = {}
    for source, start in points.items():
        row_values: dict[str, int] = {}
        for target, end in points.items():
            if source == target:
                row_values[target] = 0
                continue
            # 使用默认的最小格权缩放曼哈顿启发，也支持含 0 权的扩展地图。
            result = grid_search(start, end, grid, weighted=True, goal_free=False)
            if result.cost is None:
                raise ValueError(f"位置 {source} 到 {target} 不可达，无法按本题规则完成博弈")
            row_values[target] = result.cost
        matrix[source] = row_values
    return matrix


def _validate_problem(
    distance_matrix: Mapping[str, Mapping[str, int]],
    treasure_values: Mapping[str, int],
    initial_state: TreasureState | None,
) -> tuple[dict[str, dict[str, int]], dict[str, int], TreasureState]:
    """一次性校验完整输入，包括之后可能被剪枝的分支所需距离。"""
    if not isinstance(distance_matrix, Mapping) or not {"A", "B"} <= distance_matrix.keys():
        raise ValueError("距离矩阵必须至少包含 A、B")
    labels = [_name(label, "距离矩阵标签") for label in distance_matrix]
    label_set = set(labels)
    matrix: dict[str, dict[str, int]] = {}
    for source in labels:
        distances = distance_matrix[source]
        if not isinstance(distances, Mapping) or set(distances) != label_set:
            raise ValueError("距离矩阵必须是具有相同行列标签的完整方阵")
        row: dict[str, int] = {}
        for target in labels:
            distance = _integer(distances[target], f"距离 {source}→{target}", nonnegative=True)
            if source == target and distance != 0:
                raise ValueError("距离矩阵的对角线必须为 0")
            row[target] = distance
        matrix[source] = row

    if not isinstance(treasure_values, Mapping):
        raise ValueError("宝藏价值必须是标签到整数价值的映射")
    values: dict[str, int] = {}
    for treasure, value in treasure_values.items():
        name = _name(treasure, "宝藏标签")
        if name in {"A", "B"} or name not in matrix:
            raise ValueError(f"宝藏 {name} 必须有独立的距离矩阵标签")
        values[name] = _integer(value, f"宝藏 {name} 的价值")

    if initial_state is None:
        state = TreasureState("A", "B", frozenset(values))
    else:
        if not isinstance(initial_state, TreasureState):
            raise ValueError("initial_state 必须是 TreasureState")
        if (
            not isinstance(initial_state.pos_a, str)
            or initial_state.pos_a not in matrix
            or not isinstance(initial_state.pos_b, str)
            or initial_state.pos_b not in matrix
        ):
            raise ValueError("双方当前位置必须存在于距离矩阵中")
        if not isinstance(initial_state.remaining, frozenset) or not initial_state.remaining <= values.keys():
            raise ValueError("remaining 必须是仅含已定义宝藏标签的 frozenset")
        if not isinstance(initial_state.turn, str) or initial_state.turn not in {"A", "B"}:
            raise ValueError("当前玩家 turn 必须为 A 或 B")
        state = TreasureState(
            initial_state.pos_a,
            initial_state.pos_b,
            initial_state.remaining,
            _integer(initial_state.score_a, "A 的累计收益"),
            _integer(initial_state.score_b, "B 的累计收益"),
            initial_state.turn,
        )
    return matrix, values, state


def evaluate_treasure(
    distance_matrix: Mapping[str, Mapping[str, int]],
    treasure_values: Mapping[str, int],
    depth: int,
    *,
    use_alpha_beta: bool = True,
    initial_state: TreasureState | None = None,
    trace: bool = False,
) -> TreasureResult:
    """按给定深度求根决策，并返回可复查的计数和可选搜索轨迹。

    一次玩家行动算一层。默认从 A 开始，A 最大化、B 最小化 U。
    每层保留 treasure_values 的插入顺序；并列时取先访问的动作。
    根与内部节点共用同一递归，根层 alpha 在各动作之间持续更新。
    没有缓存、额外排序、跳过回合或未来收益评价等优化。
    elapsed_ms 只包含博弈搜索，不含输入校验和 A* 距离预计算。
    """
    horizon = _integer(depth, "搜索深度", nonnegative=True)
    if not isinstance(use_alpha_beta, bool) or not isinstance(trace, bool):
        raise ValueError("use_alpha_beta 和 trace 必须为布尔值")
    matrix, values, state = _validate_problem(distance_matrix, treasure_values, initial_state)
    order = tuple(values)
    stats = GameStats()
    events: list[dict[str, Any]] = []

    def bound(number: int | float) -> int | None:
        # JSON 不支持无穷大；None 在轨迹中表示该方向没有有限边界。
        return None if number in (-inf, inf) else int(number)

    def state_data(current: TreasureState) -> dict[str, Any]:
        return {
            "pos_a": current.pos_a,
            "pos_b": current.pos_b,
            "remaining": [target for target in order if target in current.remaining],
            "score_a": current.score_a,
            "score_b": current.score_b,
            "turn": current.turn,
        }

    def search(
        current: TreasureState,
        remaining_depth: int,
        alpha: int | float,
        beta: int | float,
        path: tuple[str, ...],
    ) -> tuple[int, str | None]:
        stats.node_visits += 1
        if trace:
            events.append({
                "event": "visit", "path": list(path), "state": state_data(current),
                "remaining_depth": remaining_depth, "alpha": bound(alpha), "beta": bound(beta),
            })
        if remaining_depth == 0 or not current.remaining:
            stats.leaf_evals += 1
            value = current.score_a - current.score_b
            if trace:
                events.append({
                    "event": "evaluate", "path": list(path), "value": value,
                    "value_kind": "exact",
                    "reason": "depth" if remaining_depth == 0 else "no_treasures",
                })
            return value, None

        maximizing = current.turn == "A"
        actions = [target for target in order if target in current.remaining]
        best_value: int | None = None
        best_target: str | None = None
        incoming_alpha, incoming_beta = alpha, beta

        for index, target in enumerate(actions):
            position = current.pos_a if maximizing else current.pos_b
            distance = matrix[position][target]
            gain = values[target] - distance
            remaining = current.remaining - {target}
            if maximizing:
                child = TreasureState(
                    target, current.pos_b, remaining,
                    current.score_a + gain, current.score_b, "B",
                )
            else:
                child = TreasureState(
                    current.pos_a, target, remaining,
                    current.score_a, current.score_b + gain, "A",
                )
            if trace:
                events.append({
                    "event": "action", "path": list(path), "player": current.turn,
                    "source": position, "target": target, "distance": distance,
                    "treasure_value": values[target], "gain": gain,
                    "next_state": state_data(child),
                })
            child_value, _ = search(child, remaining_depth - 1, alpha, beta, path + (target,))
            if best_value is None or (
                child_value > best_value if maximizing else child_value < best_value
            ):
                best_value, best_target = child_value, target

            if use_alpha_beta:
                if maximizing:
                    alpha = max(alpha, best_value)
                else:
                    beta = min(beta, best_value)
                if alpha >= beta:
                    skipped = actions[index + 1:]
                    if skipped:
                        stats.cutoff_events += 1
                        stats.pruned_branches += len(skipped)
                        if trace:
                            events.append({
                                "event": "cutoff", "path": list(path),
                                "alpha": bound(alpha), "beta": bound(beta),
                                "skipped_targets": skipped,
                            })
                    break

        # current.remaining 非空且输入已校验，故 actions 一定非空。
        assert best_value is not None
        if trace:
            value_kind = "exact"
            if use_alpha_beta and best_value <= incoming_alpha:
                value_kind = "upper_bound"
            elif use_alpha_beta and best_value >= incoming_beta:
                value_kind = "lower_bound"
            events.append({
                "event": "return", "path": list(path), "value": best_value,
                "value_kind": value_kind, "best_target": best_target,
            })
        return best_value, best_target

    started = perf_counter_ns()
    value, target = search(state, horizon, -inf, inf, ())
    stats.elapsed_ms = (perf_counter_ns() - started) / 1_000_000
    return TreasureResult(target, value, stats, events)


def solve_treasure(
    distance_matrix: Mapping[str, Mapping[str, int]],
    treasure_values: Mapping[str, int],
    depth: int,
    *,
    use_alpha_beta: bool = True,
) -> dict[str, str | int | None]:
    """返回题目要求的两个字段；需要轨迹和统计时用 evaluate_treasure。"""
    result = evaluate_treasure(
        distance_matrix, treasure_values, depth, use_alpha_beta=use_alpha_beta,
    )
    return {"best_target": result.best_target, "best_value": result.best_value}
