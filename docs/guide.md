# molbo guide

## Oracles

An oracle wraps the objective function and handles evaluation. Three are provided in `molbo.oracles`:

- `AnalyticOracle`: continuous domains; takes a callable `f` and `bounds`
- `LookupOracle`: fixed candidate pools; evaluates by index into precomputed values
- `MolecularOracle`: general molecular optimization on SMILES; subclass and implement `_evaluate(smiles) -> (N, 1) tensor`

```python
from molbo.oracles import AnalyticOracle
from molbo.oracles.lookup import LookupOracle

oracle = AnalyticOracle(f=f, dim=1, bounds=bounds)
oracle = LookupOracle(X_data=X, y_data=y)
```

## Featurization

BoTorch operators on tensors, but molbo surrogates are representation-agnostic: each surrogate uses a `Featurizer` that maps candidates to the tensors then used to make predictions. Two featurizers are provided:

```py
from molbo.surrogates.featurizer import IdentityFeaturizer, MorganFingerprintFeaturizer

IdentityFeaturizer()                                  # pass-through (default)
featurizer = MorganFingerprintFeaturizer(radius=2, fp_size=2048)
X = featurizer(smiles)                                # list[str] -> (N, 2048) count fingerprints
```

Custom featurizers subclass `Featurizer` and implement `__call__(X) -> (N, d) tensor`.

In the **fixed-pool setting**, featurize the pool once up front and pass the tensor to the oracle and loop. The surrogate then uses `IdentityFeaturizer`, avoiding re-featurizing the pool every iteration.

In the **general setting**, give the surrogate a featurizer so it can consume raw candidates:

```py
surrogate = TanimotoGPSurrogate(featurizer=MorganFingerprintFeaturizer())
```

## Surrogates

A surrogate wraps a BoTorch model (available as `surrogate.model`) with the following common functions: `initialize`, `fit`, `update`, `loss`, and `__call__(X) -> (mean, stddev)`. Two are currently provided in `molbo.surrogates.gp`:

- `GPSurrogate`: GP with RBF kernel, for continuous domains
- `TanimotoGPSurrogate`: GP with Tanimoto kernel, for molecular fingerprints

```python
from molbo.surrogates.gp import GPSurrogate, TanimotoGPSurrogate

surrogate = GPSurrogate()
surrogate = TanimotoGPSurrogate()
```

Custom surrogates subclass `molbo.surrogates.Surrogate`. To work with molbo's acquisition functions, `surrogate.model` must be a BoTorch `Model` (see [BoTorch: implementing custom models](https://botorch.org/docs/models#implementing-custom-models)).

## Acquisition functions

Analytic acquisition functions are in `molbo.acquisition.base`:

```python
from molbo.acquisition.base import LogEIAcquisition, UCBAcquisition, TSAcquisition

acq = LogEIAcquisition()
acq = UCBAcquisition(beta=1.0)
acq = TSAcquisition()
```

Batch acquisition functions (for $q > 1$) are in `molbo.acquisition.monte_carlo`:

```python
from molbo.acquisition.monte_carlo import qLogEIAcquisition, qUCBAcquisition

acq = qLogEIAcquisition()
acq = qUCBAcquisition(beta=1.0)
```

`LogEIAcquisition` / `qLogEIAcquisition` are recommended over plain EI ([ref](https://botorch.readthedocs.io/en/stable/acquisition.html#botorch.acquisition.analytic.ExpectedImprovement)).

## Acquisition optimizers

The acquisition optimizer selects candidates given the acquisition function.

**Continuous:**
```python
from molbo.acqf_optimizer.continuous import ContinuousMaximizer, ContinuousSampler

optimizer = ContinuousMaximizer(q=1)
optimizer = ContinuousMaximizer(q=4)  # batch; requires an MC acquisition function
```

**Fixed pool:**
```python
from molbo.acqf_optimizer.pool import PoolMaximizer, PoolSampler

optimizer = PoolMaximizer(q=1)
optimizer = PoolMaximizer(q=4)  # batch; requires an MC acquisition function
optimizer = PoolSampler(q=1)    # samples proportional to acquisition values
```

**General:** subclass `AcqfOptimizer` and implement `_optimize` and `sample_init`.

## History

`History` tracks the state of a run: observations, metrics, logging, and checkpointing.

```python
from molbo.bo import History

history = History(
    X_init=init.train_X,
    y_init=init.train_y,
    metrics=[my_metric],          # optional: list of Callable[[History], dict]
    logger=WandBLogger(...),      # optional: logs metrics each iteration
    checkpoint_path="run.pt",     # optional: save every checkpoint_freq iterations
    checkpoint_freq=10,
)
```

Metrics are callables `f(history) -> dict`:

```python
def my_metric(history: History) -> dict:
    return {"metric_name": value}
```

To resume from a checkpoint:

```python
history = History.load("run.pt", metrics=[my_metric], logger=WandBLogger(...))
```

## Devices

The device is set by the oracle:

```python
oracle = AnalyticOracle(...).to(device)
```

This works for `LookupOracle` too. Training data inherits the oracle's device via `sample_init`, and the surrogate follows its training data. `History` stores data on the CPU, so when running on another device, pass the initial data to `History` on the CPU:

```python
history = History(
    X_init=init.train_X.detach().cpu(),
    y_init=init.train_y.detach().cpu(),
    ...
)
```

## Memory

In the fixed-pool setting, the usual bottleneck is evaluating the GP on $N_\mathrm{test}$ candidates, which builds an $N_\mathrm{train} \times N_\mathrm{test}$ cross-covariance matrix. Pool optimizers evaluate candidates in batches, shrinking this to $N_\mathrm{train} \times$ `max_batch_size`. If you run out of memory, reduce `max_batch_size` on the pool optimizer.

## Data types

molbo does not set a global default dtype, but GP fitting needs `torch.float64` for numerical stability. Set it at the top of your script:

```python
torch.set_default_dtype(torch.float64)
```
