from itertools import count
from typing import Optional

import torch
from tqdm import tqdm

from molbo.acqf_optimizer import AcqfOptimizer
from molbo.acqf_optimizer.base import OptimizationResult
from molbo.acquisition import Acquisition
from molbo.bo.history import History
from molbo.oracles import Oracle
from molbo.oracles.on_demand_pool import validate_pool_scores
from molbo.surrogates import Surrogate


class BOLoop:
    """
    BO loop.

    Args:
        history: History object tracking observations, metrics, logging, and checkpointing.
                 Pass a freshly constructed History for a new run, or use History.load()
                 to resume from a checkpoint.
        surrogate: Surrogate model
        acq_func: Acquisition function
        oracle: Oracle
        acqf_optimizer: Acquisition function optimizer
        candidates: (N, d) candidate pool for fixed-pool optimization (optional)
        device: Device for oracle evaluations and surrogate
        evaluation_budget: Optional total pool-attempt budget, including initialization/failures
    """

    def __init__(
        self,
        history: History,
        surrogate: Surrogate,
        acq_func: Acquisition,
        oracle: Oracle,
        acqf_optimizer: AcqfOptimizer,
        candidates: Optional[torch.Tensor] = None,
        device: str = "cpu",
        evaluation_budget: Optional[int] = None,
    ):
        if evaluation_budget is not None and evaluation_budget < 1:
            raise ValueError("evaluation_budget must be at least 1")
        if evaluation_budget is not None and candidates is None:
            raise ValueError("evaluation_budget requires a candidate pool")

        self.history = history
        self.surrogate = surrogate
        self.acq_func = acq_func
        self.oracle = oracle
        self.acqf_optimizer = acqf_optimizer
        self.candidates = candidates
        self.device = device
        self.evaluation_budget = evaluation_budget

        # Initialize surrogate model from all data seen so far (handles fresh start and resume)
        X_all = history.X_all
        self.surrogate.initialize(
            X_all.to(device) if torch.is_tensor(X_all) else X_all, history.y_all.to(device)
        )

    def run(self, n_iters: Optional[int] = None) -> History:
        """Run continuous BO to an iteration limit, or pool BO to either limit.

        Pool oracles receive selected indices only. Failed scores consume the
        budget and exclude their indices from selection, but never train the GP.
        If both limits are set, stop at whichever is reached first.
        """
        if self.candidates is None and n_iters is None:
            raise ValueError("Continuous BO requires n_iters")
        iterations = (
            range(self.history.start_iteration, n_iters)
            if n_iters is not None
            else count(self.history.start_iteration)
        )

        for i in tqdm(
            iterations,
            desc="BO",
            unit="iter",
            position=0,
            leave=True,
        ):
            # Check remaining budget for pool optimization
            if self.candidates is not None:
                remaining = len(self.candidates) - len(set(self.history.observed_indices))
                if self.evaluation_budget is not None:
                    remaining = min(
                        remaining, self.evaluation_budget - len(self.history.observed_indices)
                    )
                if remaining <= 0:
                    break
                batch_size = min(self.acqf_optimizer.q, remaining)

            # Update surrogate model and acquisition function.
            self.surrogate.fit()
            self.acq_func.update(self.surrogate)
            surrogate_loss = self.surrogate.loss().item()

            # Pool-based
            if self.candidates is not None:
                result = self.acqf_optimizer.optimize(
                    self.acq_func,
                    self.candidates.to(self.device),
                    observed_indices=self.history.observed_indices,
                )
                # Truncate to max allowed budget if next batch exceeds evaluation budget
                indices = list(result.observed_indices)[:batch_size]
                selected_X = result.new_X[:batch_size]
                acquisition_values = result.acq_val.reshape(-1)[:batch_size]
                new_y, finite = validate_pool_scores(self.oracle(indices), len(indices))
                result = OptimizationResult(
                    new_X=selected_X.detach().cpu()[finite],
                    acq_val=acquisition_values.detach().cpu()[finite],
                    observed_indices=[i for i, valid in zip(indices, finite.tolist()) if valid],
                    optimization_time=result.optimization_time,
                )
                new_y = new_y[finite]

            # Continuous
            else:
                result = self.acqf_optimizer.optimize(self.acq_func)
                indices = result.observed_indices
                new_y = self.oracle(result.new_X)

            # Add successful evaluations to surrogate.
            if len(new_y):
                train_X = (
                    result.new_X.to(self.device) if torch.is_tensor(result.new_X) else result.new_X
                )
                self.surrogate.update(train_X, new_y.to(self.device))

            # Update history (handles metrics, logging, checkpointing)
            self.history.update(
                result=result,
                new_y=new_y,
                surrogate_loss=surrogate_loss,
                iteration=i,
                observed_indices=indices,
            )

        return self.history
