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
        device: Device for oracle evaluations
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
    ):
        self.history = history
        self.surrogate = surrogate
        self.acq_func = acq_func
        self.oracle = oracle
        self.acqf_optimizer = acqf_optimizer
        self.candidates = candidates
        self.device = device

        # Initialize surrogate model from all data seen so far (handles fresh start and resume)
        X_all = history.X_all
        self.surrogate.initialize(
            X_all.to(device) if torch.is_tensor(X_all) else X_all, history.y_all.to(device)
        )

    def run(self, n_iters: int) -> History:
        for i in tqdm(
            range(self.history.start_iteration, n_iters),
            desc="BO",
            unit="iter",
            position=0,
            leave=True,
        ):
            # Update surrogate model and acquisition function
            self.surrogate.fit()
            self.acq_func.update(self.surrogate)

            # Pool-based
            if self.candidates is not None:
                result = self.acqf_optimizer.optimize(
                    self.acq_func, self.candidates, observed_indices=self.history.observed_indices
                )
                new_y = self.oracle(result.observed_indices).to(self.device)

            # Continuous / generative
            else:
                result = self.acqf_optimizer.optimize(self.acq_func)
                new_y = self.oracle(result.new_X).to(self.device)

            # Update surrogate model
            self.surrogate.update(result.new_X, new_y)

            # Update history (handles metrics, logging, checkpointing)
            self.history.update(
                result=result,
                new_y=new_y,
                surrogate_loss=self.surrogate.loss().item(),
                iteration=i,
                observed_indices=result.observed_indices,
            )


class FixedPoolBOLoop(BOLoop):
    """Unlabelled fixed-pool BO loop.

    Args:
        history: History object tracking observations, metrics, logging, and checkpointing.
                 Pass a freshly constructed History for a new run, or use History.load()
                 to resume from a checkpoint.
        surrogate: Surrogate model
        acq_func: Acquisition function
        oracle: Oracle
        acqf_optimizer: Acquisition function optimizer
        candidates: (N, d) candidate pool for fixed-pool optimization
        device: Device for oracle evaluations
        evaluation_budget: Optional limit on total pool evaluations (including failures)
    """

    def __init__(
        self,
        history: History,
        surrogate: Surrogate,
        acq_func: Acquisition,
        oracle: Oracle,
        acqf_optimizer: AcqfOptimizer,
        candidates: torch.Tensor,
        device: str = "cpu",
        evaluation_budget: Optional[int] = None,
    ):
        if evaluation_budget is not None and evaluation_budget < 1:
            raise ValueError("evaluation_budget must be at least 1")

        super().__init__(history, surrogate, acq_func, oracle, acqf_optimizer, candidates, device)

        self.evaluation_budget = evaluation_budget
        self.termination_reason = None

    def _batches(self, n_iters):
        """Yield iterations and batch sizes within the iteration, budget, and pool limits."""
        iterations = (
            range(self.history.start_iteration, n_iters)
            if n_iters is not None
            else count(self.history.start_iteration)
        )
        for i in iterations:
            remaining = len(self.candidates) - len(set(self.history.observed_indices))
            if self.evaluation_budget is not None:
                remaining = min(
                    remaining, self.evaluation_budget - len(self.history.observed_indices)
                )
            if remaining <= 0:
                return
            yield i, min(self.acqf_optimizer.q, remaining)

    def run(self, n_iters: Optional[int] = None) -> History:
        """Run to an iteration limit or total pool-attempt budget.

        Pool oracles receive selected indices only. Failed scores consume the
        budget and exclude their indices from selection, but never train the GP.
        If both limits are set, stop at whichever is reached first.
        """
        if n_iters is None and self.evaluation_budget is None:
            raise ValueError("Set n_iters or a fixed-pool evaluation_budget")

        evaluation_target = (
            min(self.evaluation_budget, len(self.candidates))
            if self.evaluation_budget is not None
            else None
        )
        show_budget = n_iters is not None and evaluation_target is not None
        self.termination_reason = None
        with tqdm(
            total=n_iters if n_iters is not None else evaluation_target,
            initial=(
                self.history.start_iteration
                if n_iters is not None
                else len(self.history.observed_indices)
            ),
            desc="BO",
            unit="iter" if n_iters is not None else "eval",
            position=0,
            leave=True,
            postfix=(
                {"evaluations": f"{len(self.history.observed_indices)}/{evaluation_target}"}
                if show_budget
                else None
            ),
        ) as progress:
            for i, batch_size in self._batches(n_iters):
                self.surrogate.fit()
                self.acq_func.update(self.surrogate)
                surrogate_loss = self.surrogate.loss().item()

                # Select batch of candidates for oracle scoring
                result = self.acqf_optimizer.optimize(
                    self.acq_func,
                    self.candidates.to(self.device),
                    observed_indices=self.history.observed_indices,
                )

                # Keep the final oracle batch within budget without changing optimizer.q.
                indices = list(result.observed_indices)[:batch_size]
                selected_X = result.new_X[:batch_size]
                acquisition_values = result.acq_val.reshape(-1)[:batch_size]

                # Evaluate selected candidates and filter failures.
                new_y = self.oracle(indices)
                new_y, finite = validate_pool_scores(new_y, len(indices))

                result = OptimizationResult(
                    new_X=selected_X.detach().cpu()[finite],
                    acq_val=acquisition_values.detach().cpu()[finite],
                    optimization_time=result.optimization_time,
                )
                new_y = new_y[finite].to(self.device)

                # Add successful evaluations to surrogate
                if len(new_y):
                    self.surrogate.update(result.new_X.to(self.device), new_y)

                # Update history (handles metrics, logging, checkpointing)
                self.history.update(
                    result=result,
                    new_y=new_y,
                    surrogate_loss=surrogate_loss,
                    iteration=i,
                    observed_indices=indices,
                )

                if show_budget:
                    progress.set_postfix(
                        evaluations=f"{len(self.history.observed_indices)}/{evaluation_target}",
                        refresh=False,
                    )
                progress.update(1 if n_iters is not None else len(indices))

        if (
            self.evaluation_budget is not None
            and len(self.history.observed_indices) >= self.evaluation_budget
        ):
            self.termination_reason = "evaluation_budget_reached"
        elif len(set(self.history.observed_indices)) >= len(self.candidates):
            self.termination_reason = "candidate_pool_exhausted"
        else:
            self.termination_reason = "n_iters_reached"
        return self.history
