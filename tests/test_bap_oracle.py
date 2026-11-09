import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from molbo.oracles.boltz import BoltzOracle
from molbo.oracles.flashbind import FlashBindOracle
from molbo.oracles.nesso import NessoOracle

MODELS = [
    (BoltzOracle, "boltz", -2.5),
    (NessoOracle, "nesso", -2.5),
    (FlashBindOracle, "flashbind", -3.5),
]


@pytest.fixture
def bap(monkeypatch):
    names = [
        "bap",
        "bap.benchmark",
        "bap.benchmark.registers",
        "bap.benchmark.registers.pipelines",
        "bap.benchmark.registers.targets",
        "bap.pipeline",
        "bap.pipeline.results_caches",
        "bap.pipeline.results_caches.disk_results_cache",
    ]
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)

    pipeline = Mock()
    factories = {name: Mock(return_value=pipeline) for name in ("boltz", "nesso", "flashbind")}
    target = Mock()
    cache = Mock()
    modules["bap.benchmark.registers.pipelines"].PIPELINE_REGISTRY = factories
    modules["bap.benchmark.registers.targets"].TARGET_REGISTRY = {"TEST": target}
    modules["bap.pipeline.results_caches.disk_results_cache"].GlobalDatasetDiskResultsCache = cache
    return SimpleNamespace(pipeline=pipeline, factories=factories, target=target, cache=cache)


@pytest.mark.parametrize("cls,name,expected", MODELS)
def test_pipeline_options_and_objectives(bap, tmp_path, cls, name, expected):
    oracle = cls(
        "TEST",
        results_cache_dir=tmp_path,
        results_cache_override="none",
        gpus_list=[0],
        batch_size=4,
    )
    bap.pipeline.run.assert_not_called()
    bap.factories[name].assert_called_once_with(
        targets=[bap.target.return_value], gpus_list=[0], batch_size=4
    )
    bap.cache.assert_called_once_with(tmp_path)
    bap.pipeline.run.return_value = [
        SimpleNamespace(score_dict={"affinity_pred_value": -2.5, "flashbind_pred_value": 3.5}),
        SimpleNamespace(score_dict={}),
        SimpleNamespace(
            score_dict={"affinity_pred_value": float("inf"), "flashbind_pred_value": float("inf")}
        ),
    ]
    values = oracle(["CC", "FAIL", "NONFINITE"])
    assert values.shape == (3, 1)
    assert values.dtype == torch.float64
    assert values[0].item() == expected
    assert torch.isnan(values[1:]).all()
    bap.pipeline.run.assert_called_once_with(
        smiles_list=["CC", "FAIL", "NONFINITE"],
        results_cache=bap.cache.return_value,
        results_cache_override="none",
        progress_bar=False,
    )


@pytest.mark.parametrize("cls,name,expected", MODELS)
def test_optional_ligand_efficiency(bap, tmp_path, cls, name, expected):
    scores = {"affinity_pred_value": -2.5, "flashbind_pred_value": 3.5}
    bap.pipeline.run.return_value = [SimpleNamespace(score_dict=scores) for _ in range(4)] + [
        SimpleNamespace(score_dict={})
    ]
    smiles = ["CC", "CCCC", "INVALID", "[H][H]", "CO"]
    normalized = cls("TEST", results_cache_dir=tmp_path, ligand_efficiency=True)
    torch.testing.assert_close(
        normalized(smiles),
        torch.tensor(
            [[expected / 2], [expected / 4], [float("nan")], [float("nan")], [float("nan")]],
            dtype=torch.float64,
        ),
        equal_nan=True,
    )
    assert "ligand_efficiency" not in bap.factories[name].call_args.kwargs
    assert scores == {"affinity_pred_value": -2.5, "flashbind_pred_value": 3.5}


def test_empty_inputs_and_result_alignment(bap, tmp_path):
    oracle = BoltzOracle("TEST", results_cache_dir=tmp_path)
    assert oracle([]).shape == (0, 1)
    bap.pipeline.run.assert_not_called()
    bap.pipeline.run.return_value = []
    with pytest.raises(RuntimeError, match="different number"):
        oracle(["CC"])
