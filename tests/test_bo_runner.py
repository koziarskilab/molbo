# Optional Hydra imports follow importorskip so base-only installations can collect tests.
# ruff: noqa: E402
import csv
import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("hydra")
from hydra import compose, initialize_config_dir

from molbo.acqf_optimizer.continuous import ContinuousMaximizer
from molbo.bo import History
from scripts.bo_runner import load_inputs, run_bo, setup_pool


def test_replay_run_exports_failed_attempts_and_fit_context(tmp_path, bo_config, candidate_csv):
    smiles = [row["smiles"] for row in csv.DictReader(candidate_csv.open())]
    scores = [0.0, 1.0, None, 3.0, 4.0, 5.0]  # Seed 0 attempts failed candidate 2 first.
    path = tmp_path / "scores.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["canonical_smiles", "score"])
        writer.writerows(zip(smiles[::-1], scores[::-1]))
    cfg = bo_config(
        "oracle=replay",
        "metrics=reference",
        f"evaluation.reference.path={path}",
        "evaluation.reference.column=score",
        "evaluation.reference.smiles_column=canonical_smiles",
        "evaluation.reference.multiplier=-1",
    )
    output = tmp_path / "run"
    history = run_bo(cfg, output)
    rows = list(csv.DictReader((output / "evaluations.csv").open()))
    assert [int(row["pool_index"]) for row in rows] == history.observed_indices
    assert rows[0]["success"] == "False" and rows[0]["objective"] == ""
    assert [float(row["objective"]) for row in rows[1:]] == [
        -scores[i] for i in history.observed_indices[1:]
    ]
    summary = json.loads((output / "summary.json").read_text())
    assert (summary["attempted_evaluations"], summary["failed_evaluations"]) == (5, 1)
    metrics = list(csv.DictReader((output / "metrics.csv").open()))
    assert len(metrics) == len(history.metrics_history)
    assert metrics[0]["surrogate/loss"] == ""
    loss = metrics[-1]
    assert tuple(int(loss[key]) for key in ("fit_n_attempted", "n_train", "n_attempted")) == (
        3,
        2,
        5,
    )
    assert float(loss["surrogate/loss"]) == history.surrogate_loss[0]


def test_continuous_resume_uses_saved_observations(tmp_path, monkeypatch):
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")
    ):
        cfg = compose(
            config_name="continuous",
            overrides=[
                "bo.n_init=3",
                "bo.n_iters=1",
                "bo.checkpoint_freq=10",
                "optimizer.num_restarts=2",
                "optimizer.raw_samples=8",
                "acquisition.sampler.num_samples=8",
            ],
        )
    run_bo(cfg, tmp_path / "first")
    history = History.load(tmp_path / "first/history.pt")
    assert history.n_successful == 4  # Completion saves beyond the periodic checkpoint.

    def unexpected_init(*args, **kwargs):
        pytest.fail("Continuation must skip initialization")

    monkeypatch.setattr(ContinuousMaximizer, "sample_init", unexpected_init)
    cfg.bo.n_iters = 2
    resumed = run_bo(cfg, tmp_path / "resumed", history=history)
    assert resumed is history and resumed.n_successful == 5 and len(resumed.iteration) == 2


def test_gneprop_setup_registers_smiles_as_candidate_ids(tmp_path, bo_config):
    weights = tmp_path / "synthetic.ckpt"
    weights.write_text("SYNTHETIC TEST WEIGHTS")
    cfg = bo_config("surrogate=gneprop", f"surrogate.weights_path={weights}")
    candidates, values = load_inputs(cfg)
    surrogate, _, _, pool, _ = setup_pool(cfg, candidates, values)
    assert surrogate.candidate_smiles == [row["smiles"] for row in candidates]
    torch.testing.assert_close(
        pool.candidates, torch.arange(len(candidates), dtype=torch.float64).reshape(-1, 1)
    )
