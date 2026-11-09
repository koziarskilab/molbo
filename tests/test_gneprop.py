import sys
from types import ModuleType

import numpy as np
import pytest
import torch
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

from molbo.surrogates.gneprop import GNEPropPosteriorModel, GNEPropSurrogate


def test_gneprop_refits_and_scales_without_consuming_prediction_rng(tmp_path, monkeypatch):
    fits, estimators, draws = [], [], []

    class Regressor:
        @classmethod
        def from_checkpoint(cls, path, **options):
            model = cls()
            model.model_ = torch.nn.Sequential(torch.nn.Dropout(0.0))
            estimators.append(model)
            return model

        def fit(self, smiles, targets):
            fits.append(np.asarray(targets).copy())
            self.history_ = [{"loss": 0.25}]

        def predict_uncertainty(self, smiles, mc_samples):
            assert self.model_[0].p == 0.13
            draws.append(torch.rand(3))
            return np.arange(len(smiles), dtype=float), np.full(len(smiles), 0.2)

    module = ModuleType("gneprop")
    module.GNEPropRegressor = Regressor
    monkeypatch.setitem(sys.modules, "gneprop", module)
    weights = tmp_path / "synthetic.ckpt"
    weights.write_bytes(b"SYNTHETIC TEST WEIGHTS")
    surrogate = GNEPropSurrogate(weights, dropout=0.13, random_state=17)
    candidates = surrogate.prepare_candidates(["CC", "CO", "CN", "CC"])
    assert candidates.flatten().tolist() == [0, 1, 2, 0]
    surrogate.initialize(candidates[:2], torch.tensor([[1.0], [3.0]], dtype=torch.float64))
    rng = torch.get_rng_state()
    surrogate.fit()
    assert torch.equal(torch.get_rng_state(), rng)
    np.testing.assert_array_equal(fits[0], [-1.0, 1.0])
    assert surrogate.model.means.tolist() == [2.0, 3.0, 4.0]
    assert surrogate.model.variances.tolist() == [0.2, 0.2, 0.2]
    assert surrogate.loss().item() == 0.25
    surrogate.update(candidates[2:3], torch.tensor([[5.0]], dtype=torch.float64))
    surrogate.fit()
    assert len(estimators) == 2 and estimators[0] is not estimators[1]
    torch.testing.assert_close(draws[0], draws[1], rtol=0, atol=0)


@pytest.mark.parametrize("variances", [[0.0, 0.0, 0.0], [0.0, 0.2, 1e-30]])
def test_dropout_posterior_supports_analytic_and_joint_acquisitions(variances):
    model = GNEPropPosteriorModel([-1.0, 0.0, 1.0], variances)
    X = torch.arange(3, dtype=torch.float64).reshape(-1, 1, 1)
    posterior = model.posterior(X)
    torch.testing.assert_close(posterior.mean.flatten(), model.means)
    duplicate = model.posterior(X[[1, 1]].reshape(2, 1))
    assert duplicate.distribution.covariance_matrix.tolist() == [[variances[1]] * 2] * 2
    assert torch.isfinite(duplicate.rsample(torch.Size([4]))).all()
    for acquisition in (
        UpperConfidenceBound(model, beta=1.0),
        LogExpectedImprovement(model, best_f=1.0),
        qUpperConfidenceBound(model, beta=1.0),
        qLogExpectedImprovement(model, best_f=1.0),
    ):
        assert torch.isfinite(acquisition(X)).all()
