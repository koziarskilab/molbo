from pathlib import Path

import pytest


@pytest.fixture
def candidate_csv(tmp_path):
    path = tmp_path / "candidates.csv"
    path.write_text("molecule_id,smiles\n0,CCO\n1,CCN\n2,CCC\n3,CCCl\n4,c1ccccc1\n5,CC(=O)O\n")
    return path


@pytest.fixture
def bo_config(candidate_csv):
    hydra = pytest.importorskip("hydra")

    def build(*overrides):
        with hydra.initialize_config_dir(
            version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")
        ):
            cfg = hydra.compose(
                config_name="pool",
                overrides=[f"data.path={candidate_csv}", "bo.evaluation_budget=5", *overrides],
            )
        if cfg.get("featurizer") and cfg.featurizer._target_.endswith(
            "MorganFingerprintFeaturizer"
        ):
            cfg.featurizer.fp_size = 64
        if cfg.acquisition.get("sampler"):
            cfg.acquisition.sampler.num_samples = 8
        return cfg

    return build
