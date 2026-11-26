"""旧 E0 兼容、E1 模型分派、配置校验和初始权重指纹的 CPU 检查。"""

import copy
import importlib.util
import subprocess
import sys
import unittest

from src.common import ROOT, normalize_config, read_json


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None


def old_e0_config():
    """使用仍保留旧格式的 E0 文件，不人为替它补 model 字段。"""
    return read_json(ROOT / "configs/e0.json")


class ConfigurationTests(unittest.TestCase):
    def test_legacy_config_defaults_paths_idempotence_and_input_preservation(self):
        raw = old_e0_config()
        before = copy.deepcopy(raw)
        self.assertNotIn("model", raw)
        result = normalize_config(raw)
        self.assertEqual(raw, before)
        self.assertEqual(result["model"], "simple_cnn")
        self.assertIs(result["smoke"], False)
        self.assertEqual(result["device"], "cuda")
        for field in ["data_dir", "output_dir"]:
            self.assertEqual(result[field], str((ROOT / before[field]).resolve()))
        self.assertEqual(normalize_config(result), result)

    def test_explicit_e1_and_runtime_fields(self):
        raw = read_json(ROOT / "configs/e1.json")
        self.assertEqual(raw["experiment"], "E1")
        self.assertEqual(raw["model"], "cifar_resnet18")
        raw.update(device="cpu", smoke=True, seed=0, split_seed=0, monitor_seed=0)
        result = normalize_config(raw)
        self.assertEqual(result["model"], "cifar_resnet18")
        self.assertEqual(result["device"], "cpu")
        self.assertIs(result["smoke"], True)
        self.assertEqual(result["seed"], 0)

    def test_e1_requires_model_and_rejects_unknown_or_mismatched_models(self):
        e1 = old_e0_config()
        e1["experiment"] = "E1"
        with self.assertRaises(ValueError):
            normalize_config(e1)
        cases = [("E1", "simple_cnn"), ("E0", "cifar_resnet18"),
                 ("E1", "unknown_resnet"), ("E1", 1), ("E1", None)]
        for experiment, model in cases:
            with self.subTest(experiment=experiment, model=model), self.assertRaises(ValueError):
                normalize_config(dict(old_e0_config(), experiment=experiment, model=model))

    def test_unknown_fields_and_missing_required_fields_are_rejected(self):
        raw = old_e0_config()
        raw["epoch_typo"] = 10
        with self.assertRaises(ValueError):
            normalize_config(raw)
        raw = old_e0_config()
        del raw["seed"]
        with self.assertRaises(ValueError):
            normalize_config(raw)

    def test_seeds_reject_negative_boolean_noninteger_and_overflow(self):
        for field in ["seed", "split_seed", "monitor_seed"]:
            for value in [-1, True, False, 1.5, "42", 2 ** 32]:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    normalize_config(dict(old_e0_config(), **{field: value}))

    def test_counts_reject_boolean_negative_or_noninteger_values(self):
        for field in ["epochs", "batch_size", "log_interval"]:
            for value in [-1, 0, True, False, 1.5]:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    normalize_config(dict(old_e0_config(), **{field: value}))
        for value in [-1, True, False, 0.5]:
            with self.subTest(num_workers=value), self.assertRaises(ValueError):
                normalize_config(dict(old_e0_config(), num_workers=value))

    def test_flags_device_and_numeric_hyperparameters_are_validated(self):
        for field in ["amp", "smoke"]:
            for value in [0, 1, "false", None]:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    normalize_config(dict(old_e0_config(), **{field: value}))
        for changes in [
            {"device": "cuda:wrong"}, {"learning_rate": 0}, {"learning_rate": True},
            {"learning_rate": float("nan")}, {"weight_decay": -0.1},
            {"momentum": 1}, {"dropout": 1}, {"min_learning_rate": 1},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                normalize_config(dict(old_e0_config(), **changes))


@unittest.skipUnless(HAS_TORCH, "需要已安装 PyTorch 的 Python 环境")
class ModelFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_legacy_e0_can_build_with_torchvision_import_forbidden(self):
        # 新进程可检验首次导入的惰性行为，不受其它测试已导入模块的影响。
        script = """
import builtins
import torch
torch.set_num_threads(1)
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'torchvision' or name.startswith('torchvision.') or 'cifar_resnet' in name:
        raise AssertionError('旧 E0 构建不应导入 torchvision/ResNet')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from src.models import build_model
from src.common import ROOT, read_json
model = build_model(read_json(ROOT / 'configs/e0.json'))
assert type(model).__name__ == 'SimpleCNN'
assert sum(p.numel() for p in model.parameters()) == 1148874
print('legacy-e0-factory-ok')
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True,
                                text=True, encoding="utf-8", timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("legacy-e0-factory-ok", result.stdout)

    @unittest.skipUnless(HAS_TORCHVISION, "E1 构建需要 torchvision")
    def test_factory_builds_distinct_models_and_expected_parameter_counts(self):
        from src.models import build_model

        e0 = build_model(old_e0_config())
        e1 = build_model(read_json(ROOT / "configs/e1.json"))
        self.assertEqual(type(e0).__name__, "SimpleCNN")
        self.assertEqual(type(e1).__name__, "CIFARResNet18")
        self.assertEqual(sum(p.numel() for p in e0.parameters()), 1_148_874)
        self.assertEqual(sum(p.numel() for p in e1.parameters()), 11_173_962)
        self.assertEqual(e0.classifier[0].p, 0.3)
        self.assertEqual(e1.fc[0].p, 0.3)
        self.assertEqual(e1.conv1.kernel_size, (3, 3))

    def test_checkpoint_legacy_and_consistent_explicit_identifiers(self):
        from src.models import model_name_from_checkpoint

        old = {"config": old_e0_config(), "model": {}}
        self.assertEqual(model_name_from_checkpoint(old), "simple_cnn")
        e1 = {"config": read_json(ROOT / "configs/e1.json"), "model_name": "cifar_resnet18", "model": {}}
        self.assertEqual(model_name_from_checkpoint(e1), "cifar_resnet18")

    def test_checkpoint_rejects_conflicting_or_unknown_identifiers(self):
        from src.models import model_name_from_checkpoint

        cases = [
            {"config": old_e0_config(), "model_name": "cifar_resnet18"},
            {"config": old_e0_config(), "model_name": "unknown"},
            {"config": read_json(ROOT / "configs/e1.json"), "model_name": "simple_cnn"},
            {"config": dict(old_e0_config(), experiment="E1")},
        ]
        for state in cases:
            with self.subTest(state=state), self.assertRaises(ValueError):
                model_name_from_checkpoint(state)

    def test_state_hash_tracks_parameters_and_batchnorm_buffers(self):
        import torch
        from src.models import build_model, state_dict_sha256

        state = build_model(old_e0_config()).state_dict()
        first = state_dict_sha256(state)
        self.assertRegex(first, r"^[0-9a-f]{64}$")
        cloned = {key: value.clone() for key, value in state.items()}
        self.assertEqual(first, state_dict_sha256(cloned))
        # 哈希应由内容决定，而不是字典插入顺序或 torch.save 容器格式决定。
        self.assertEqual(first, state_dict_sha256(dict(reversed(list(cloned.items())))))
        weight_key = next(key for key in state if key.endswith("weight"))
        cloned[weight_key].flatten()[0] += 1
        self.assertNotEqual(first, state_dict_sha256(cloned))
        cloned = {key: value.clone() for key, value in state.items()}
        buffer_key = next(key for key in state if key.endswith("running_mean"))
        cloned[buffer_key][0] += 1
        self.assertNotEqual(first, state_dict_sha256(cloned))
        cloned = {key: value.clone() for key, value in state.items()}
        counter_key = next(key for key in state if key.endswith("num_batches_tracked"))
        self.assertEqual(cloned[counter_key].ndim, 0)
        cloned[counter_key] += 1
        self.assertNotEqual(first, state_dict_sha256(cloned))

    def test_state_hash_includes_tensor_names_shapes_and_dtypes(self):
        import torch
        from src.models import state_dict_sha256

        values = torch.arange(4, dtype=torch.float32)
        first = state_dict_sha256({"weight": values})
        self.assertNotEqual(first, state_dict_sha256({"different_name": values}))
        self.assertNotEqual(first, state_dict_sha256({"weight": values.reshape(2, 2)}))
        self.assertNotEqual(first, state_dict_sha256({"weight": values.view(torch.int32)}))


if __name__ == "__main__":
    unittest.main()
