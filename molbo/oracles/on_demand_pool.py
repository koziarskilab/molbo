from typing import Any, Callable, Sequence

import torch

from molbo.oracles.base import Oracle


class OnDemandPoolOracle(Oracle):
    """Map fixed-pool indices to objects for a black-box oracle.

    Features are used for acquisition scoring only. Constructing this adapter
    never calls the wrapped oracle; only requested indices are evaluated.
    """

    def __init__(self, features: torch.Tensor, objects: Sequence[Any], oracle: Callable):
        super().__init__()

        if len(features) != len(objects):
            raise ValueError("features and objects must have equal length")

        self.candidates = features
        self.objects = list(objects)
        self.oracle = oracle

    def _evaluate(self, indices: list[int]):
        return self.oracle([self.objects[i] for i in indices])

    def __getitem__(self, indices):
        return self.candidates[indices], self(indices)


def validate_pool_scores(scores, n_expected):
    """Validate pool scores and identify finite values.

    Args:
        scores: Scores returned by the oracle.
        n_expected: The number of evaluated candidates.

    Returns:
        Scores on CPU and a boolean mask identifying finite scores.
    """
    y = scores.detach().cpu()

    if y.shape != (n_expected, 1):
        raise ValueError(f"Oracle must return shape ({n_expected}, 1), got {tuple(y.shape)}")

    return y, torch.isfinite(y.squeeze(-1))
