"""地图生成、拒绝采样、输入哈希和独立路径检查的必要测试。"""

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

from experiments.maze_dataset import bfs_distance, create_dataset, map_sha256, validate_path


WORKSPACE = Path(__file__).resolve().parent.parent


@contextmanager
def workspace_temporary_directory():
    """测试沙箱采用继承权限，自动清理前核对路径确实位于 tests 下。"""
    temporary_root = (WORKSPACE / "tests").resolve()
    temporary = temporary_root / f"tmp_maze_dataset_{uuid4().hex}"
    temporary.mkdir()
    try:
        yield temporary
    finally:
        if not temporary.resolve().is_relative_to(temporary_root):
            raise RuntimeError("拒绝清理 tests 目录外的路径")
        shutil.rmtree(temporary)


class MazeDatasetTests(unittest.TestCase):
    def test_bfs_four_neighbors_and_single_point(self):
        self.assertEqual(bfs_distance([["S", ".", "E"]], (0, 0), (0, 2)), 2)
        self.assertIsNone(bfs_distance([["S", "X"], ["X", "E"]], (0, 0), (1, 1)))
        self.assertEqual(bfs_distance([["S"]], (0, 0), (0, 0)), 0)
        with self.assertRaises(ValueError):
            bfs_distance([["S"], [".", "E"]], (0, 0), (1, 1))

    def test_path_checks_allow_any_optimal_route(self):
        grid = [["S", "."], [".", "E"]]
        validate_path(grid, (0, 0), (1, 1), [(0, 0), (0, 1), (1, 1)], 2)
        validate_path(grid, (0, 0), (1, 1), [(0, 0), (1, 0), (1, 1)], 2)
        for path, cost in (([(0, 0), (1, 1)], 1), ([], 2), ([(0, 0), (1, 0)], 1)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_path(grid, (0, 0), (1, 1), path, cost)
        validate_path([["S", "X", "E"]], (0, 0), (0, 2), [], None)
        with self.assertRaises(ValueError):
            validate_path([["S", "X", "E"]], (0, 0), (0, 2), [(0, 0), (0, 1), (0, 2)], 2)

    def test_dataset_reproducibility_and_rejection_audit(self):
        with workspace_temporary_directory() as temporary:
            first, second = Path(temporary) / "first", Path(temporary) / "second"
            settings = {"sizes": (7, 9), "count_per_size": 4, "density": 0.4, "master_seed": 20261021}
            one = create_dataset(first, **settings)
            two = create_dataset(second, **settings)
            self.assertEqual(one, two)
            self.assertEqual(one["total_accepted"], 8)
            self.assertGreater(one["total_rejected"], 0)
            for relative in one["accepted_files"] + one["rejected_files"]:
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
                record = json.loads((first / relative).read_text(encoding="utf-8"))
                self.assertEqual(record["sha256"], map_sha256(record["start"], record["end"], record["grid"]))
                self.assertEqual(record["bfs_cost"], bfs_distance(record["grid"], record["start"], record["end"]))
                self.assertEqual(record["accepted"], record["bfs_cost"] is not None)
                size = record["size"]
                self.assertAlmostEqual(record["observed_density"], record["obstacle_count"] / (size * size - 2))
            for group in one["groups"]:
                self.assertEqual(group["accepted"], 4)
                self.assertEqual(group["attempts"], group["accepted"] + group["rejected"])
            with self.assertRaises(FileExistsError):
                create_dataset(first, **settings)

    def test_generator_parameters_and_attempt_limit(self):
        with workspace_temporary_directory() as temporary:
            root = Path(temporary)
            for index, settings in enumerate((
                {"sizes": ()}, {"sizes": (1,)}, {"sizes": (3, 3)},
                {"count_per_size": 0}, {"density": 1}, {"density": float("nan")},
                {"master_seed": True},
            )):
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    create_dataset(root / f"invalid_{index}", **settings)
            failed = root / "failed"
            with self.assertRaises(RuntimeError):
                create_dataset(failed, sizes=(7,), count_per_size=1, density=0.999, max_attempts_per_size=2)
            manifest = json.loads((failed / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "incomplete")
            self.assertEqual(len(manifest["rejected_files"]), 2)
            self.assertEqual(manifest["groups"][0]["accepted"], 0)


if __name__ == "__main__":
    unittest.main()
