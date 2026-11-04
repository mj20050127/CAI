"""题四动态即时收益排序；同一搜索器支持基线与排序对照。"""

from __future__ import annotations

from math import inf
from time import perf_counter_ns

from src.problem4_treasure import (
    TreasureResult,
    TreasureState,
    _integer,
    _validate_problem,
)
from src.stats import GameStats


def evaluate_gain_treasure(
    distance_matrix,
    treasure_values,
    depth,
    *,
    ordering="original",
    use_alpha_beta=True,
    initial_state=None,
    trace=False,
):
    """求解单次博弈；搜索时间包括排序及计算排序键的全部开销。

    原序在每层按输入宝藏的插入顺序过滤剩余宝藏。gain 顺序在每个
    展开节点，按当前行动者的 value-distance 降序稳定排序；同收益
    维持输入顺序。A 和 B 都优先取得自身收益较高的宝藏。顺序影响
    同值最优动作的选取，但不改变根的最优值。两策略使用同一递归。

    可选轨迹仅供计时外审计。order 事件记录已排序的全部候选；action
    事件只记录实际访问的前缀。剪枝计数与生产实现使用相同口径。
    """
    horizon = _integer(depth, "搜索深度", nonnegative=True)
    if not isinstance(ordering, str) or ordering not in ("original", "gain"):
        raise ValueError("ordering 必须为 original 或 gain")
    if not isinstance(use_alpha_beta, bool) or not isinstance(trace, bool):
        raise ValueError("use_alpha_beta 和 trace 必须为布尔值")
    matrix, values, state = _validate_problem(distance_matrix, treasure_values, initial_state)
    order = tuple(values)
    stats = GameStats()
    events = []

    def bound(number):
        return None if number in (-inf, inf) else int(number)

    def state_data(current):
        return {
            "pos_a": current.pos_a,
            "pos_b": current.pos_b,
            "remaining": [target for target in order if target in current.remaining],
            "score_a": current.score_a,
            "score_b": current.score_b,
            "turn": current.turn,
        }

    def search(current, remaining_depth, alpha, beta, path):
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
        position = current.pos_a if maximizing else current.pos_b
        actions = [target for target in order if target in current.remaining]
        if ordering == "gain" and len(actions) > 1:
            # Python 的稳定排序保留同收益时的原优先级；key 包含所有
            # 候选，包括稍后可能被剪枝的候选，其计算成本计入搜索。
            actions.sort(key=lambda target: values[target] - matrix[position][target], reverse=True)
        if trace:
            events.append({
                "event": "order", "path": list(path), "player": current.turn, "source": position,
                "candidates": [{"target": target, "gain": values[target] - matrix[position][target]}
                               for target in actions],
            })
        best_value = None
        best_target = None
        incoming_alpha, incoming_beta = alpha, beta

        for index, target in enumerate(actions):
            distance = matrix[position][target]
            gain = values[target] - distance
            remaining = current.remaining - {target}
            if maximizing:
                child = TreasureState(target, current.pos_b, remaining,
                                      current.score_a + gain, current.score_b, "B")
            else:
                child = TreasureState(current.pos_a, target, remaining,
                                      current.score_a, current.score_b + gain, "A")
            if trace:
                events.append({
                    "event": "action", "path": list(path), "player": current.turn,
                    "source": position, "target": target, "distance": distance,
                    "treasure_value": values[target], "gain": gain,
                    "next_state": state_data(child),
                })
            child_value, _ = search(child, remaining_depth - 1, alpha, beta, path + (target,))
            if best_value is None or (child_value > best_value if maximizing else child_value < best_value):
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
