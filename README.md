# molbo

A lightweight Bayesian optimization library for molecular design.

## Installation

Install the latest version:

```bash
pip install git+https://github.com/wvirany/molbo
```

## Overview

molbo is easy to use! It just requires the following key ingredients:

- **Oracle**: the objective function being optimized
- **Surrogate model**: a probabilistic model of the objective
- **Acquisition function**: scores candidates using model predictions
- **Acquisition optimizer**: selects the next candidate(s) to evaluate
- **History**: handles bookkeeping: metrics, logging, and checkpointing

Three settings are supported, each defined by an oracle and acquisition optimizer pair:

- **Continuous:** `AnalyticOracle` + `ContinuousMaximizer` (gradient-based optimization over a bounded domain)
- **Fixed pool:** `LookupOracle` + `PoolMaximizer` or `PoolSampler` (selection from a fixed candidate set)
- **General:** a custom oracle (e.g., a `MolecularOracle` on SMILES) + a custom `AcqfOptimizer` (e.g., a generative model)


## Example

```python
import torch
from molbo.acqf_optimizer.pool import PoolMaximizer
from molbo.acquisition import LogEIAcquisition
from molbo.bo import BOLoop, History
from molbo.oracles.lookup import LookupOracle
from molbo.surrogates.featurizer import MorganFingerprintFeaturizer
from molbo.surrogates.gp import TanimotoGPSurrogate

torch.set_default_dtype(torch.float64)

featurizer = MorganFingerprintFeaturizer()

smiles = [...]                                  # candidate pool
X = featurizer(smiles)
y = ...                                         # (N, 1) objective values
oracle = LookupOracle(X_data=X, y_data=y)

optimizer = PoolMaximizer(q=1)
init = optimizer.sample_init(oracle, n_init=10)

history = History(
    X_init=init.train_X,
    y_init=init.train_y,
    observed_indices=init.observed_indices,
)

bo_loop = BOLoop(
    history=history,
    surrogate=TanimotoGPSurrogate(),
    acq_func=LogEIAcquisition(),
    oracle=oracle,
    acqf_optimizer=optimizer,
    candidates=X,
)
bo_loop.run(n_iters=20)
```

See [`docs/guide.md`](docs/guide.md) for documentation and [`examples/`](examples/) for runnable scripts.
