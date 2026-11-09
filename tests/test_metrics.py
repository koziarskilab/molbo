import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from molbo.acqf_optimizer.base import OptimizationResult
from molbo.acqf_optimizer.pool import PoolMaximizer
from molbo.bo import BOLoop, History
from molbo.bo.logger import Logger
from molbo.bo.metrics import (
    BestObserved,
    MetricsList,
    PosteriorMeanRecall,
    PredictionError,
    TopKMean,
    TopKMeanAUC,
)
from molbo.oracles.lookup import LookupOracle


class StubSurrogate:
    def initialize(self, X, y):
        self.queries = 0

    def fit(self):
        torch.rand(1)
        self.model = torch.nn.Module()

    def update(self, X, y):
        assert self.model.training  # Reporting must restore the fitted model's mode.
        self.model = None

    def __call__(self, X):
        assert self.model is not None
        self.queries += 1
        torch.rand(1)
        np.random.rand()
        random.random()
        self.model.eval()
        return X[..., :1], 2 * torch.ones_like(X[..., :1])

    def loss(self):
        assert self.model is not None  # Loss must be read before update invalidates the fit.
        return torch.tensor(0.5)


class RecordingLogger(Logger):
    def __init__(self):
        self.rows = []

    def log(self, data):
        self.rows.append(data)


class StubAcquisition:
    X_pending = None

    def __call__(self, X):
        return torch.rand(len(X))

    def set_X_pending(self, X):
        self.X_pending = X


def _run(metrics=()):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    features = torch.arange(6, dtype=torch.float64).reshape(-1, 1)
    history = History(
        features[:2],
        features[:2],
        [0, 1],
        successful_indices=[0, 1],
        metrics=MetricsList(metrics),
        logger=RecordingLogger(),
    )
    surrogate = StubSurrogate()
    acquisition = SimpleNamespace(update=lambda model: None, acq_func=StubAcquisition())
    loop = BOLoop(
        history,
        surrogate,
        acquisition,
        LookupOracle(features, features),
        PoolMaximizer(q=2),
        candidates=features,
        evaluation_budget=6,
        prediction_batch_size=2,
    )
    return loop.run(), surrogate


def test_metrics_list_namespaces_values():
    metrics = MetricsList([BestObserved(), TopKMean(k=2)])
    history = History(torch.ones(2, 1), torch.tensor([[1.0], [3.0]]))
    assert metrics.compute_metrics(history) == {"best/value": 3.0, "top2/value": 2.0}
    with pytest.raises(ValueError, match="Metric names must be unique"):
        MetricsList([BestObserved(), BestObserved()])


def test_reporting_preserves_selection_and_rng():
    plain, plain_model = _run()
    torch_state, numpy_state, python_state = (
        torch.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
    )
    reported, model = _run(
        [
            PredictionError(list(range(6))),
            PosteriorMeanRecall([0.0] * 6, threshold=1, k=2),
        ]
    )
    assert reported.observed_indices == plain.observed_indices
    assert torch.equal(torch_state, torch.get_rng_state())
    np.testing.assert_equal(numpy_state, np.random.get_state())
    assert python_state == random.getstate()
    assert plain_model.queries == 0 and model.queries == 6  # Two shared three-chunk passes.


def test_prediction_metrics_use_pre_batch_model():
    history, _ = _run([PredictionError(list(range(6)))])
    assert [
        (row["fit_n_attempted"], row["n_attempted"], row["prediction/count"])
        for row in history.metrics_history
    ] == [(2, 4, 4), (4, 6, 2)]
    assert all(row["prediction/rmse"] == 0 for row in history.metrics_history)
    np.testing.assert_array_equal(history.predictions["mean"], np.arange(6))
    np.testing.assert_array_equal(history.predictions["variance"], np.full(6, 4.0))
    np.testing.assert_array_equal(history.predictions["order"], np.arange(5, -1, -1))
    assert history.predictions["n_attempted"] == history.predictions["n_train"] == 4


def test_history_logs_metrics_and_one_loss_per_iteration():
    history, _ = _run([BestObserved()])
    assert history.logger.rows == history.metrics_history
    assert (
        [row["surrogate/loss"] for row in history.metrics_history]
        == history.surrogate_loss
        == [0.5, 0.5]
    )
    assert all("best/value" in row for row in history.logger.rows)


def test_checkpoint_keeps_predictions_and_scalar_metrics(tmp_path):
    history, _ = _run([PredictionError(list(range(6)))])
    path = tmp_path / "history.pt"
    history.save(path)
    loaded = History.load(path)
    np.testing.assert_equal(loaded.predictions, history.predictions)
    assert loaded.metrics_history == history.metrics_history
    assert loaded.termination_reason == history.termination_reason


@pytest.mark.parametrize("successful_index, expected_auc", [(2, 3.0), (3, 2.0)])
def test_auc_counts_failed_attempts_in_order(successful_index, expected_auc):
    history = History(torch.tensor([[1.0]]), torch.tensor([[2.0]]), [0, 1], successful_indices=[1])
    result = OptimizationResult(
        torch.tensor([[2.0]]), torch.tensor([0.4]), [successful_index], 0.01
    )
    history.update(result, torch.tensor([[4.0]]), 0.5, 0, observed_indices=[2, 3])
    assert TopKMeanAUC(k=1).compute(history)["value"] == expected_auc


def test_posterior_ranking_breaks_ties_by_candidate_id():
    features = torch.full((2, 1), 2.0)
    history = History(
        features[:1],
        torch.ones(1, 1),
        [0],
        successful_indices=[0],
        metrics=MetricsList([PosteriorMeanRecall([1.0, 0.0], threshold=0.5, k=1)]),
    )
    loop = BOLoop(
        history,
        StubSurrogate(),
        SimpleNamespace(update=lambda model: None, acq_func=lambda X: X[:, 0, 0]),
        LookupOracle(features, torch.ones(2, 1)),
        PoolMaximizer(q=1),
        candidates=features,
        candidate_ids=["b", "a"],
        evaluation_budget=2,
    )
    loop.run()
    np.testing.assert_array_equal(history.predictions["order"], [1, 0])
    assert history.metrics_history[-1]["recall@1/recall"] == 1
