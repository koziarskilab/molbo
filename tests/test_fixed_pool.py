import pytest
import torch

from molbo.acqf_optimizer.pool import PoolMaximizer
from molbo.acquisition.monte_carlo import qLogEIAcquisition
from molbo.bo import BOLoop, History
from molbo.oracles.lookup import LookupOracle
from molbo.oracles.on_demand_pool import OnDemandPoolOracle
from molbo.surrogates.gp import TanimotoGPSurrogate

N, D, N_INIT = 12, 8, 2


# ── Fixtures ──────────────────────────────────────────────────────────────────


class StubOracle:
    def __init__(self, objects):
        self.scores = {obj: float(i) for i, obj in enumerate(objects)}
        self.failed = set()
        self.calls = []

    def __call__(self, objects):
        self.calls.append(list(objects))
        return torch.tensor(
            [float("nan") if obj in self.failed else self.scores[obj] for obj in objects],
            dtype=torch.float64,
        ).unsqueeze(-1)


@pytest.fixture
def candidates():
    torch.manual_seed(0)
    return torch.rand(N, D, dtype=torch.float64)


@pytest.fixture
def scorer():
    return StubOracle([f"molecule-{i}" for i in range(N)])


@pytest.fixture
def oracle(candidates, scorer):
    return OnDemandPoolOracle(candidates, list(scorer.scores), scorer)


@pytest.fixture
def qei(candidates, oracle):
    surrogate = TanimotoGPSurrogate()
    train_X, train_y = oracle[list(range(N_INIT))]
    surrogate.initialize(train_X, train_y)
    surrogate.fit()
    acq = qLogEIAcquisition()
    acq.update(surrogate)
    return acq


def _build_loop(candidates, oracle, optimizer, budget=None, checkpoint_path=None):
    init = optimizer.sample_init_on_demand(oracle, n_init=N_INIT, evaluation_budget=budget, seed=0)
    history = History(
        X_init=init.train_X,
        y_init=init.train_y,
        observed_indices=init.observed_indices,
        successful_indices=init.successful_indices,
        checkpoint_path=checkpoint_path,
        checkpoint_freq=1,
    )
    return (
        BOLoop(
            history=history,
            surrogate=TanimotoGPSurrogate(),
            acq_func=qLogEIAcquisition(),
            oracle=oracle,
            acqf_optimizer=optimizer,
            candidates=candidates,
            evaluation_budget=budget,
        ),
        history,
    )


# ── PoolMaximizer ─────────────────────────────────────────────────────────


class TestPoolMaximizer:

    def test_initialization_stops_at_budget(self, oracle, scorer):
        scorer.failed.update(scorer.scores)
        with pytest.raises(RuntimeError, match="got 0 successes in 3 attempts"):
            PoolMaximizer().sample_init_on_demand(oracle=oracle, n_init=1, evaluation_budget=3)
        assert sum(map(len, scorer.calls)) == 3

    def test_caps_batch_at_available_candidates(self, candidates, qei):
        optimizer = PoolMaximizer(q=4, max_batch_size=3)
        result = optimizer.optimize(qei, candidates, observed_indices=list(range(N - 2)))
        assert set(result.observed_indices) == {N - 2, N - 1}
        assert optimizer.q == 4
        assert qei.acq_func.X_pending is None
        assert len(optimizer.optimize(qei, candidates).observed_indices) == 4
        with pytest.raises(ValueError, match="No unobserved candidates remain"):
            optimizer.optimize(qei, candidates, observed_indices=list(range(N)))

    def test_nonpositive_batch_size_is_rejected(self):
        with pytest.raises(ValueError, match="q must be at least 1"):
            PoolMaximizer(q=0)

    def test_lookup_initialization_keeps_original_behavior(self, candidates):
        y = torch.arange(N, dtype=torch.float64).unsqueeze(-1)
        oracle = LookupOracle(candidates, y, noise_std=0.1)
        torch.manual_seed(42)
        expected = torch.randperm(N)[:N_INIT]
        expected_rng = torch.get_rng_state()
        # The original lookup path returns supplied values without retrying NaNs.
        y[expected[0]] = float("nan")
        torch.manual_seed(42)
        init = PoolMaximizer().sample_init(oracle=oracle, n_init=N_INIT)
        assert init.observed_indices == expected.tolist()
        assert torch.equal(init.train_X, candidates[expected])
        torch.testing.assert_close(init.train_y, y[expected], equal_nan=True)
        assert torch.equal(torch.get_rng_state(), expected_rng)

    def test_wrong_oracle_output_shape(self, oracle):
        oracle.oracle = lambda objects: torch.zeros(len(objects))
        with pytest.raises(ValueError, match="Oracle must return shape"):
            PoolMaximizer().sample_init_on_demand(oracle, n_init=N_INIT)


# ── End-to-end on-demand BOLoop ────────────────────────────────────────────────


class TestOnDemandBOLoop:

    def test_scores_only_initial_and_selected_objects(self, candidates, oracle, scorer):
        assert scorer.calls == []
        optimizer = PoolMaximizer(q=2)
        loop, history = _build_loop(candidates, oracle, optimizer, budget=5)
        assert [len(batch) for batch in scorer.calls] == [N_INIT]
        assert loop.run(n_iters=None) is history
        assert [obj for batch in scorer.calls for obj in batch] == [
            oracle.objects[i] for i in history.observed_indices
        ]
        assert [len(batch) for batch in scorer.calls] == [2, 2, 1]
        assert len(set(history.observed_indices)) == 5 < N
        assert torch.equal(history.X_all, candidates[history.observed_indices])
        expected_y = torch.tensor(
            [[scorer.scores[oracle.objects[i]]] for i in history.observed_indices],
            dtype=torch.float64,
        )
        assert torch.equal(history.y_all, expected_y)
        assert optimizer.q == 2

    def test_failures_consume_budget_without_training(self, candidates, oracle, scorer):
        first = torch.randperm(N, generator=torch.Generator().manual_seed(0))[0].item()
        scorer.failed.add(oracle.objects[first])
        loop, history = _build_loop(candidates, oracle, PoolMaximizer(q=2), budget=7)
        assert len(history.observed_indices) == 3
        assert len(history.y_all) == N_INIT
        initial_successes = history.successful_indices.copy()
        assert initial_successes == history.observed_indices[1:]

        # Fail the first BO batch, then finish with finite scores.
        scorer.failed.update(
            oracle.objects[i] for i in range(N) if i not in history.observed_indices
        )
        loop.run(n_iters=1)
        assert len(history.observed_indices) == 5
        assert len(history.y_all) == N_INIT
        assert history.successful_indices == initial_successes
        scorer.failed.clear()
        loop.run(n_iters=None)
        assert len(history.observed_indices) == len(set(history.observed_indices)) == 7
        assert sum(map(len, scorer.calls)) == 7
        assert len(history.y_all) == len(loop.surrogate.train_y) == 4
        assert history.successful_indices == initial_successes + history.observed_indices[-2:]
        assert torch.equal(history.X_all, candidates[history.successful_indices])
        assert torch.isfinite(loop.surrogate.train_y).all()

    @pytest.mark.parametrize(
        "n_iters,budget,size,attempts,reason",
        [
            (None, 10, 5, 5, "candidate_pool_exhausted"),
            (10, None, 5, 5, "candidate_pool_exhausted"),
            (1, 10, 8, 4, "n_iters_reached"),
            (None, N_INIT, 8, N_INIT, "evaluation_budget_reached"),
        ],
    )
    def test_stopping_limits(self, candidates, scorer, n_iters, budget, size, attempts, reason):
        candidates = candidates[:size]
        oracle = OnDemandPoolOracle(candidates, list(scorer.scores)[:size], scorer)
        loop, history = _build_loop(candidates, oracle, PoolMaximizer(q=2), budget=budget)
        loop.run(n_iters=n_iters)
        assert loop.termination_reason == history.termination_reason == reason
        assert len(history.observed_indices) == sum(map(len, scorer.calls)) == attempts

    def test_checkpoint_resume(self, candidates, oracle, scorer, tmp_path):
        checkpoint = tmp_path / "ckpt.pt"
        loop, history = _build_loop(
            candidates, oracle, PoolMaximizer(q=2), budget=7, checkpoint_path=checkpoint
        )
        scorer.failed.update(
            oracle.objects[i] for i in range(N) if i not in history.observed_indices
        )
        loop.run(n_iters=1)

        # Resume with finite scores; failed rows must remain excluded and counted.
        scorer.failed.clear()
        history2 = History.load(checkpoint)
        assert history2.start_iteration == 1
        assert history2.observed_indices == history.observed_indices
        assert history2.successful_indices == history.successful_indices
        assert history2.n_initial_attempts == N_INIT
        assert len(history2.observed_indices) == 4
        loop2 = BOLoop(
            history=history2,
            surrogate=TanimotoGPSurrogate(),
            acq_func=qLogEIAcquisition(),
            oracle=oracle,
            acqf_optimizer=PoolMaximizer(q=2),
            candidates=candidates,
            evaluation_budget=7,
        )
        loop2.run(n_iters=None)
        assert len(history2.observed_indices) == len(set(history2.observed_indices)) == 7
        assert sum(map(len, scorer.calls)) == 7
        assert len(history2.y_all) == 5


@pytest.mark.parametrize(
    "observed, successful, initial", [([0, 1], [0, 1], 1), ([0, 1, 2], None, None)]
)
def test_legacy_checkpoint_infers_only_unambiguous_indices(tmp_path, observed, successful, initial):
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "X_init": torch.zeros(1, 1),
            "y_init": torch.tensor([[2.0]]),
            "X_observed": torch.ones(1, 1),
            "y_observed": torch.tensor([[4.0]]),
            "observed_indices": observed,
            "acq_vals": [0.4],
            "iteration": [0],
            "model_loss": [0.5],
            "optimization_time": [0.01],
            "rng_state": torch.get_rng_state(),
        },
        path,
    )
    history = History.load(path)
    torch.testing.assert_close(history.y_all, torch.tensor([[2.0], [4.0]]))
    assert history.surrogate_loss == [0.5]
    assert history.successful_indices == successful and history.n_initial_attempts == initial
