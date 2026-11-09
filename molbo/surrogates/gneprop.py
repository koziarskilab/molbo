from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DenseLinearOperator

from molbo.surrogates.base import Surrogate
from molbo.surrogates.featurizer import IdentityFeaturizer


class GNEPropPosteriorModel(Model):
    """Gaussian posterior built from GNEprop MC-dropout means and variances."""

    def __init__(self, means, variances):
        super().__init__()
        self.register_buffer("means", torch.as_tensor(means, dtype=torch.float64))
        self.register_buffer("variances", torch.as_tensor(variances, dtype=torch.float64))

    @property
    def num_outputs(self):
        return 1

    @property
    def batch_shape(self):
        return torch.Size()

    def posterior(self, X, posterior_transform=None):
        """Build a Gaussian posterior for the requested candidate indices.

        Args:
            X: Tensor of shape (..., q, 1) containing candidate indices.
            posterior_transform: Optional transform to apply to the posterior.

        Returns:
            A GPyTorchPosterior object representing the Gaussian posterior.
        """
        indices = X.squeeze(-1).long()
        mean = self.means.to(X)[indices]
        variance = self.variances.to(X)[indices]
        same = indices.unsqueeze(-1) == indices.unsqueeze(-2)
        covariance = same.to(mean) * variance.unsqueeze(-1)
        posterior = GPyTorchPosterior(MultivariateNormal(mean, DenseLinearOperator(covariance)))

        if posterior_transform is not None:
            posterior = posterior_transform(posterior, X=X)

        return posterior


class GNEPropSurrogate(Surrogate):
    """Fit GNEprop using `gneprop` library."""

    def __init__(
        self,
        weights_path: str,
        mc_samples=25,
        standardize_targets=True,
        **model_options,
    ):
        self.weights_path = Path(weights_path).expanduser().resolve(strict=True)

        if mc_samples < 2 or not 0 < model_options.get("dropout") < 1:
            raise ValueError("MC dropout requires mc_samples >= 2 and dropout in (0, 1)")

        self.model_options = {"num_workers": 0, "drop_last": False, **model_options}
        self.mc_samples = mc_samples
        self.standardize_targets = standardize_targets
        self.featurizer = IdentityFeaturizer()
        self.candidate_smiles = None
        self.model = None

    def prepare_candidates(self, smiles):
        """Register the pool and return internal lookup IDs for acquisition scoring.

        Repeated SMILES share an ID. These keys carry no chemical information;
        GNEprop constructs molecular graphs from the registered SMILES itself.
        """
        self.candidate_smiles = list(dict.fromkeys(smiles))
        indices = {smi: i for i, smi in enumerate(self.candidate_smiles)}
        self.model = None
        return torch.tensor([indices[smi] for smi in smiles], dtype=torch.float64).reshape(-1, 1)

    def initialize(self, train_X, train_y):
        """Store internal pool IDs and the finite observations supplied by the BO loop."""
        self.train_X, self.train_y = train_X.detach().clone(), train_y.detach().clone()
        self.model = None

    def update(self, new_X, new_y):
        self.initialize(
            torch.cat([self.train_X, new_X.to(self.train_X)]),
            torch.cat([self.train_y, new_y.to(self.train_y)]),
        )

    def fit(self):
        import gneprop

        self.model = None
        indices = self.train_X.squeeze(-1).long().cpu().tolist()
        smiles = [self.candidate_smiles[i] for i in indices]
        targets = self.train_y.squeeze(-1).cpu().numpy()
        center = float(targets.mean()) if self.standardize_targets else 0.0
        scale = (float(targets.std()) or 1.0) if self.standardize_targets else 1.0
        seed = self.model_options.get("random_state")
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

        estimator = gneprop.GNEPropRegressor.from_checkpoint(
            str(self.weights_path), **self.model_options
        )
        for module in estimator.model_.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = self.model_options.get("dropout")  # set dropout probability
        estimator.fit(smiles, (targets - center) / scale)

        # GNEprop seeds loading/training; isolate and seed only MC-dropout prediction here.
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                torch.manual_seed(seed)
            means, variances = estimator.predict_uncertainty(
                self.candidate_smiles, mc_samples=self.mc_samples
            )

        means = np.asarray(means, dtype=np.float64) * scale + center
        variances = np.asarray(variances, dtype=np.float64) * scale**2
        loss = float(estimator.history_[-1]["loss"])

        self.model = GNEPropPosteriorModel(means, variances).to(self.train_X)
        self.training_loss = loss

    def __call__(self, X):
        self._check_fitted()
        posterior = self.model.posterior(X)
        return posterior.mean, posterior.variance.sqrt().unsqueeze(-1)

    def loss(self):
        self._check_fitted()
        return self.train_y.new_tensor(self.training_loss)

    @property
    def is_fitted(self):
        return self.model is not None
