"""Continuous BO: maximize a noisy 1D function with a GP surrogate and LogEI."""

import torch

from molbo.acqf_optimizer.continuous import ContinuousMaximizer
from molbo.acquisition import LogEIAcquisition
from molbo.bo import BOLoop, History
from molbo.oracles import AnalyticOracle
from molbo.surrogates.gp import GPSurrogate

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def f(X):
    return torch.sin(3 * X) + 0.5 * X


bounds = torch.tensor([[-3.0], [3.0]])
oracle = AnalyticOracle(f=f, dim=1, bounds=bounds, noise_std=0.1)

optimizer = ContinuousMaximizer(q=1)
init = optimizer.sample_init(oracle, n_init=5)

history = History(X_init=init.train_X, y_init=init.train_y)
BOLoop(
    history=history,
    surrogate=GPSurrogate(),
    acq_func=LogEIAcquisition(),
    oracle=oracle,
    acqf_optimizer=optimizer,
).run(n_iters=20)

best = history.y_all.argmax()
print(f"Best x = {history.X_all[best].item():.3f}, y = {history.y_all[best].item():.3f}")
