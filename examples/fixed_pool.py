"""Fixed-pool BO: maximizing QED in a small pool with a Tanimoto GP."""

import torch

from molbo.acqf_optimizer.pool import PoolMaximizer
from molbo.acquisition import LogEIAcquisition
from molbo.bo import BOLoop, History
from molbo.oracles.lookup import LookupOracle
from molbo.oracles.molecular import QEDOracle
from molbo.surrogates.featurizer import MorganFingerprintFeaturizer
from molbo.surrogates.gp import TanimotoGPSurrogate

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(=O)Nc1ccc(O)cc1",
    "CN1CCC[C@H]1c1cccnc1",
    "OC(=O)c1ccccc1O",
    "c1ccc2c(c1)cccc2",
    "CCO",
    "CCN(CC)CC",
    "c1ccccc1",
    "CC(C)NCC(O)c1ccc(O)c(O)c1",
    "COc1ccc2[nH]cc(CCN)c2c1",
    "NC(=O)c1cccnc1",
    "CC1=C(C(=O)OC)C(c2ccccc2[N+](=O)[O-])C(C(=O)OC)=C(C)N1",
    "O=C(O)CCc1ccccc1",
    "CN(C)CCCN1c2ccccc2CCc2ccccc21",
    "Clc1ccc(cc1)C(c1ccccc1)N1CCNCC1",
    "OCC1OC(O)C(O)C(O)C1O",
    "CCCCCCCCCCCCCCCC(=O)O",
    "c1ccc(cc1)-c1ccccc1",
]

# Objective: QED drug-likeness, precomputed for the whole pool
y = QEDOracle()(smiles)

# Featurize the pool once up front
X = MorganFingerprintFeaturizer()(smiles)
oracle = LookupOracle(X_data=X, y_data=y)

optimizer = PoolMaximizer(q=1)
init = optimizer.sample_init(oracle, n_init=5)

history = History(
    X_init=init.train_X,
    y_init=init.train_y,
    observed_indices=init.observed_indices,
)
BOLoop(
    history=history,
    surrogate=TanimotoGPSurrogate(),
    acq_func=LogEIAcquisition(),
    oracle=oracle,
    acqf_optimizer=optimizer,
    candidates=X,
).run(n_iters=10)

best = history.observed_indices[history.y_all.argmax()]
print(f"Best found: {smiles[best]} (QED = {y[best].item():.3f})")
print(f"Pool optimum: {smiles[y.argmax()]} (QED = {y.max().item():.3f})")
