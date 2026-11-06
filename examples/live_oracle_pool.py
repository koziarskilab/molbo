"""Score selected molecules on demand with the public QED oracle."""

from molbo.acqf_optimizer.pool import PoolMaximizer
from molbo.acquisition.base import LogEIAcquisition
from molbo.bo import FixedPoolBOLoop, History
from molbo.oracles.molecular import QEDOracle
from molbo.oracles.on_demand_pool import OnDemandPoolOracle
from molbo.surrogates.featurizer import MorganFingerprintFeaturizer
from molbo.surrogates.gp import TanimotoGPSurrogate

smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCOC", "CC(C)O"]
features = MorganFingerprintFeaturizer()(smiles)
oracle = OnDemandPoolOracle(features, smiles, QEDOracle())
optimizer = PoolMaximizer(q=1)
budget = 4
init = optimizer.sample_init_on_demand(oracle, n_init=2, evaluation_budget=budget, seed=0)

history = History(
    X_init=init.train_X,
    y_init=init.train_y,
    observed_indices=init.observed_indices,
)

loop = FixedPoolBOLoop(
    history=history,
    surrogate=TanimotoGPSurrogate(),
    acq_func=LogEIAcquisition(),
    oracle=oracle,
    acqf_optimizer=optimizer,
    candidates=features,
    evaluation_budget=budget,
)

loop.run()
print("Evaluated pool rows:", history.observed_indices)
print("Best observed QED:", history.best_observed)
