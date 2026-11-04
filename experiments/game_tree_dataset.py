"""题三的固定随机树与独立自底向上参考值。

参考器只读原始字典，不使用待测 Minimax 或 Alpha-Beta 的递归代码。
实际输入按文件字节计算 SHA-256；种子之外也保存整棵树。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
GENERATOR_VERSION = "game_tree_dataset_v1"
ENTRY_FIELDS = (
    "instance_id", "source", "tree_file", "tree_sha256", "seed", "depth",
    "branch_factor", "node_count", "leaf_count", "reference_value",
)


def _root_dict(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("树输入必须是字典")
    if "tree_structure" in data:
        if any(key in data for key in ("value", "children", "is_maximizing_player")):
            raise ValueError("tree_structure 包装不能混入根节点字段")
        data = data["tree_structure"]
    if not isinstance(data, dict):
        raise ValueError("树根必须是字典")
    return data


def reference_tree(data: dict[str, Any]) -> dict[str, int]:
    """用显式后序栈求完整树的精确值，根为 MAX、逐层交替。

    拒绝共享对象和环，以保证本实验的节点数同时表示唯一节点与访问数。
    不等分支的 ``branch_factor`` 为 0，单叶树也为 0；深度为最大边数。
    """
    root = _root_dict(data)
    stack = [(root, 0, False)]
    seen: set[int] = set()
    values: dict[int, int] = {}
    branches: set[int] = set()
    node_count = leaf_count = max_depth = 0
    while stack:
        node, depth, returning = stack.pop()
        if returning:
            children = node["children"]
            child_values = [values[id(child)] for child in children]
            values[id(node)] = max(child_values) if depth % 2 == 0 else min(child_values)
            continue
        if not isinstance(node, dict):
            raise ValueError("每个节点必须是字典")
        if id(node) in seen:
            raise ValueError("数据集必须是无共享、无环的独立节点树")
        seen.add(id(node))
        node_count += 1
        max_depth = max(max_depth, depth)
        role = node.get("is_maximizing_player")
        if role is not None and (type(role) is not bool or role != (depth % 2 == 0)):
            raise ValueError("显式 MAX/MIN 类型与根为 MAX、逐层交替的规则矛盾")
        children = node.get("children", [])
        if not isinstance(children, list):
            raise ValueError("children 必须是列表")
        if children:
            if node.get("value") is not None:
                raise ValueError("内部节点不能同时具有叶子分数")
            branches.add(len(children))
            stack.append((node, depth, True))
            stack.extend((child, depth + 1, False) for child in reversed(children))
        else:
            value = node.get("value")
            if type(value) is not int:
                raise ValueError("叶子分数必须是整数，不能是 bool")
            leaf_count += 1
            values[id(node)] = value
    return {
        "value": values[id(root)], "node_count": node_count, "leaf_count": leaf_count,
        "depth": max_depth, "branch_factor": next(iter(branches)) if len(branches) == 1 else 0,
    }


def derive_seed(master_seed: int, depth: int, index: int) -> int:
    """由版本、主种子、深度、编号稳定派生 64 位实例种子。"""
    payload = f"{GENERATOR_VERSION}:{master_seed}:{depth}:{index}".encode("ascii")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _validate_configuration(
    depths: Sequence[int], count_per_depth: int, branch_factor: int,
    value_min: int, value_max: int, master_seed: int,
) -> list[int]:
    if isinstance(depths, (str, bytes)) or not isinstance(depths, Sequence) or not depths:
        raise ValueError("depths 必须是非空深度序列")
    dimensions = list(depths)
    if any(type(depth) is not int or depth < 0 for depth in dimensions):
        raise ValueError("深度必须是非负整数")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("深度不能重复")
    if type(count_per_depth) is not int or count_per_depth < 1:
        raise ValueError("每种深度的实例数必须是正整数")
    if type(branch_factor) is not int or branch_factor < 2:
        raise ValueError("分支因子必须是至少为 2 的整数")
    if any(type(value) is not int for value in (value_min, value_max, master_seed)):
        raise ValueError("叶值范围和主种子必须是整数")
    if value_min > value_max:
        raise ValueError("叶值下界不能超过上界")
    return dimensions


def _generate_tree(depth: int, branch_factor: int, low: int, high: int, seed: int):
    rng = random.Random(seed)

    def build(level: int) -> dict[str, Any]:
        if level == depth:
            return {"value": rng.randint(low, high)}
        return {
            "is_maximizing_player": level % 2 == 0,
            "children": [build(level + 1) for _ in range(branch_factor)],
        }

    return {"tree_structure": build(0)}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def create_dataset(
    output_dir: Path, *, depths: Sequence[int] = (4, 6, 8), count_per_depth: int = 30,
    branch_factor: int = 3, value_min: int = -50, value_max: int = 50,
    master_seed: int = 20261022, official_path: Path | None = None,
) -> dict[str, Any]:
    """创建实际树、元数据和独立参考值；拒绝覆盖非空输出目录。"""
    dimensions = _validate_configuration(
        depths, count_per_depth, branch_factor, value_min, value_max, master_seed,
    )
    official_path = Path(official_path or ROOT / "cases/problem3.json")
    official_bytes = official_path.read_bytes()
    official_data = json.loads(official_bytes.decode("utf-8-sig"))
    official_reference = reference_tree(official_data)
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("数据集目录非空，请使用新的目录以免覆盖输入")
    destination.mkdir(parents=True, exist_ok=True)
    trees_dir = destination / "trees"
    trees_dir.mkdir()
    manifest: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION, "status": "incomplete",
        "config": {
            "depths": dimensions, "count_per_depth": count_per_depth,
            "branch_factor": branch_factor, "value_min": value_min, "value_max": value_max,
            "master_seed": master_seed,
            "depth_definition": "number of edges from root (depth 0) to leaves",
            "seed_strategy": "int.from_bytes(SHA256(f'game_tree_dataset_v1:{master_seed}:{depth}:{index}').digest()[:8], 'big')",
            "leaf_sampling": "Random(instance_seed).randint(value_min,value_max), independent draws in left-to-right depth-first leaf order",
            "tree_structure": "full fixed-branch tree; MAX root, alternating roles; no shared nodes; original generated child order",
            "official_input": "verbatim bytes from the supplied official JSON; no generated-tree constraints applied",
        },
        "instances": [],
    }
    for depth in dimensions:
        for index in range(count_per_depth):
            instance_id = f"tree_d{depth:03d}_{index:03d}"
            seed = derive_seed(master_seed, depth, index)
            data = _generate_tree(depth, branch_factor, value_min, value_max, seed)
            reference = reference_tree(data)
            relative_path = f"trees/{instance_id}.json"
            path = destination / relative_path
            _write_json(path, data)
            manifest["instances"].append({
                "instance_id": instance_id, "source": "random", "tree_file": relative_path,
                "tree_sha256": sha256(path.read_bytes()).hexdigest(), "seed": seed,
                **{key: reference[key] for key in ("depth", "branch_factor", "node_count", "leaf_count")},
                "reference_value": reference["value"],
            })
    official_file = trees_dir / "official.json"
    official_file.write_bytes(official_bytes)
    manifest["instances"].append({
        "instance_id": "official", "source": "official", "tree_file": "trees/official.json",
        "tree_sha256": sha256(official_bytes).hexdigest(), "seed": None,
        **{key: official_reference[key] for key in ("depth", "branch_factor", "node_count", "leaf_count")},
        "reference_value": official_reference["value"],
    })
    manifest["status"] = "complete"
    _write_json(destination / "manifest.json", manifest)
    return manifest


def load_dataset(dataset_dir: Path):
    """核对字节哈希、生成身份及全部数值元数据，返回 manifest 和求值输入。"""
    # 只在完成独立字典审计后转换为待测接口的节点类型，不调用求值器。
    from src.game_tree import tree_from_dict

    directory = Path(dataset_dir).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or manifest.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("未知博弈树数据集版本")
    if manifest.get("status") != "complete":
        raise ValueError("数据集未完整生成")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("数据集缺少生成配置")
    try:
        depths = _validate_configuration(*(config[key] for key in (
            "depths", "count_per_depth", "branch_factor", "value_min", "value_max", "master_seed",
        )))
    except KeyError as error:
        raise ValueError("生成配置字段不完整") from error
    entries = manifest.get("instances")
    expected = {
        f"tree_d{depth:03d}_{index:03d}": (depth, index)
        for depth in depths for index in range(config["count_per_depth"])
    }
    if not isinstance(entries, list) or len(entries) != len(expected) + 1:
        raise ValueError("数据集实例数与配置不一致")
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    cases = []
    for entry in entries:
        if not isinstance(entry, dict) or not set(ENTRY_FIELDS) <= set(entry):
            raise ValueError("实例元数据字段不完整")
        instance_id = entry["instance_id"]
        if not isinstance(instance_id, str) or instance_id in seen_ids:
            raise ValueError("实例编号无效或重复")
        seen_ids.add(instance_id)
        file_name = entry["tree_file"]
        if not isinstance(file_name, str):
            raise ValueError("实例文件名必须是字符串")
        path = (directory / file_name).resolve()
        if not path.is_relative_to(directory) or path in seen_paths:
            raise ValueError("实例文件越过数据目录或被重复引用")
        seen_paths.add(path)
        payload = path.read_bytes()
        if sha256(payload).hexdigest() != entry["tree_sha256"]:
            raise ValueError(f"树文件哈希不一致：{instance_id}")
        data = json.loads(payload.decode("utf-8-sig"))
        reference = reference_tree(data)
        for field in ("depth", "branch_factor", "node_count", "leaf_count", "reference_value"):
            actual = reference["value"] if field == "reference_value" else reference[field]
            if type(entry[field]) is not int or entry[field] != actual:
                raise ValueError(f"独立参考值或计数不一致：{instance_id}/{field}")
        if instance_id == "official":
            if entry["source"] != "official" or entry["seed"] is not None:
                raise ValueError("官方实例的来源或种子字段无效")
        else:
            if instance_id not in expected or entry["source"] != "random":
                raise ValueError("随机实例编号或来源与配置不一致")
            depth, index = expected[instance_id]
            if type(entry["seed"]) is not int or entry["seed"] != derive_seed(config["master_seed"], depth, index):
                raise ValueError("随机实例种子与固定派生规则不一致")
            branches = config["branch_factor"]
            expected_branches = branches if depth else 0
            if (entry["depth"], entry["branch_factor"], entry["node_count"], entry["leaf_count"]) != (
                depth, expected_branches, (branches ** (depth + 1) - 1) // (branches - 1), branches ** depth,
            ):
                raise ValueError("随机实例不符合指定深度的满树结构")
            pending = [_root_dict(data)]
            while pending:
                node = pending.pop()
                children = node.get("children", [])
                if children:
                    pending.extend(children)
                elif not config["value_min"] <= node["value"] <= config["value_max"]:
                    raise ValueError("随机叶值超出生成范围")
        cases.append({**entry, "tree_data": data, "root": tree_from_dict(data)})
    if seen_ids != set(expected) | {"official"}:
        raise ValueError("数据集缺少规定实例")
    return manifest, cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成题三固定随机博弈树及独立参考值")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--count-per-depth", type=int, default=30)
    parser.add_argument("--branch-factor", type=int, default=3)
    parser.add_argument("--value-min", type=int, default=-50)
    parser.add_argument("--value-max", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args(argv)
    try:
        manifest = create_dataset(
            args.out, depths=args.depths, count_per_depth=args.count_per_depth,
            branch_factor=args.branch_factor, value_min=args.value_min, value_max=args.value_max,
            master_seed=args.seed,
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"数据集配置或文件错误：{error}\n")
    print(f"Created {len(manifest['instances'])} trees: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
