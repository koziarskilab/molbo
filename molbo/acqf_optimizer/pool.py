import torch

from molbo.acqf_optimizer.base import AcqfOptimizer, Initialization, OptimizationResult
from molbo.oracles.on_demand_pool import OnDemandPoolOracle, validate_pool_scores


def _optimize_acqf_discrete(
    acq_function, q: int, choices: torch.Tensor, observed_indices=None, max_batch_size: int = 1024
):
    """Custom implementation of BoTorch optimize_acqf_discrete which returns chosen indices.

    This is particularly useful when checking equality between candidates is expensive due to
    high dimensionality, e.g., optimizing molecular fingerprints.

    Args:
        acq_function: BoTorch acquisition function
        q: Batch size
        choices: featurized candidate set
        observed_indices: Optional = None, previously observed indices to filter
        max_batch_size: int = 1024 Maximum number of candidates to evaluate in a batch,
            GP inference requires building N_train x N_test covariance
            matrix; reduce this to limit memory usage
    """
    # Build candidate indices, filter observed indices
    observed = set(observed_indices or [])
    choices_idx = [i for i in range(len(choices)) if i not in observed]

    if not choices_idx:
        raise ValueError("No unobserved candidates remain")

    batch_size = min(q, len(choices_idx))  # cap batch size to remaining candidates if < q

    # q = 1 or batch size = 1
    if batch_size == 1:
        with torch.no_grad():
            acq_values = torch.cat(
                [
                    acq_function(X_)
                    for X_ in choices[choices_idx].unsqueeze(-2).split(max_batch_size)
                ]
            )
        local_idx = torch.argmax(acq_values).item()  # best idx w.r.t. filtered candidates
        return [choices_idx[local_idx]], acq_values[local_idx]

    # q > 1 or batch size > 1
    chosen_indices, acq_value_list = [], []
    base_X_pending = acq_function.X_pending

    for _ in range(batch_size):
        with torch.no_grad():
            acq_values = torch.cat(
                [
                    acq_function(X_)
                    for X_ in choices[choices_idx].unsqueeze(-2).split(max_batch_size)
                ]
            )
        local_idx = torch.argmax(acq_values).item()  # best idx w.r.t. filtered candidates
        chosen_indices.append(choices_idx.pop(local_idx))
        acq_value_list.append(acq_values[local_idx])

        acq_function.set_X_pending(
            torch.cat([base_X_pending, choices[chosen_indices]], dim=-2)
            if base_X_pending is not None
            else choices[chosen_indices]
        )

    acq_function.set_X_pending(base_X_pending)
    return chosen_indices, torch.stack(acq_value_list)


class PoolBase(AcqfOptimizer):
    def __init__(self, q: int = 1, max_batch_size: int = 1024):
        # check valid batch size
        if q < 1:
            raise ValueError("q must be at least 1")

        self.q = q
        self.max_batch_size = max_batch_size

    def sample_init(self, oracle, n_init: int) -> Initialization:
        indices = torch.randperm(len(oracle.candidates))[:n_init].tolist()
        train_X, train_y = oracle[indices]
        return Initialization(train_X=train_X, train_y=train_y, observed_indices=indices)

    def sample_init_on_demand(
        self,
        oracle: OnDemandPoolOracle,
        n_init: int,
        evaluation_budget=None,
        batch_size=None,
        seed=None,
    ) -> Initialization:
        """Sample finite initial scores, counting all attempts against the budget.

        Args:
            oracle: The on-demand pool oracle.
            n_init: The number of initial evaluations to perform.
            evaluation_budget: The maximum number of evaluations to perform.
            batch_size: The number of evaluations to perform in each batch.
            seed: The random seed to use. Default is Torch RNG

        Returns:
            An Initialization object containing all attempted and successful indices.
        """
        budget = len(oracle.candidates)
        if evaluation_budget is not None:
            budget = min(evaluation_budget, budget)
        batch_size = n_init if batch_size is None else batch_size
        if n_init < 1 or n_init > budget:
            raise ValueError("n_init must be positive and fit within the pool evaluation budget")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        order = torch.randperm(len(oracle.candidates), generator=generator).tolist()
        attempted, successful_indices, features, values = [], [], [], []
        successful = 0

        while successful < n_init and len(attempted) < budget:
            count = min(batch_size, n_init - successful, budget - len(attempted))
            indices = order[len(attempted) : len(attempted) + count]

            # Read each batch once, preserving feature/score alignment and device placement.
            batch_X, batch_y = oracle[indices]
            _, finite = validate_pool_scores(batch_y, len(indices))

            attempted.extend(indices)
            successful_indices.extend(i for i, valid in zip(indices, finite.tolist()) if valid)
            successful += int(finite.sum())
            features.append(batch_X[finite.to(batch_X.device)])
            values.append(batch_y[finite.to(batch_y.device)])

        if successful < n_init:
            raise RuntimeError(
                f"Unable to obtain {n_init} successful initialization evaluations: "
                f"got {successful} successes in {len(attempted)} attempts "
                f"within an effective evaluation budget of {budget}"
            )

        return Initialization(
            train_X=torch.cat(features),
            train_y=torch.cat(values),
            observed_indices=attempted,
            successful_indices=successful_indices,
        )


class PoolMaximizer(PoolBase):
    def _optimize(self, acq_func, candidates: torch.Tensor, observed_indices=None):
        chosen_indices, acq_val = _optimize_acqf_discrete(
            acq_function=acq_func.acq_func,
            q=self.q,
            choices=candidates,
            observed_indices=observed_indices,
            max_batch_size=self.max_batch_size,
        )
        return OptimizationResult(
            new_X=candidates[chosen_indices], acq_val=acq_val, observed_indices=chosen_indices
        )


class PoolSampler(PoolBase):
    def _optimize(self, acq_func, candidates: torch.Tensor, observed_indices=None):
        observed = set(observed_indices or [])
        available = [i for i in range(len(candidates)) if i not in observed]
        available_candidates = candidates[available]

        with torch.no_grad():
            acq_values = torch.cat(
                [
                    acq_func(chunk.reshape(-1, 1, chunk.shape[-1]))
                    for chunk in available_candidates.split(self.max_batch_size)
                ]
            )

        acq_clamped = acq_values.clamp(min=0)
        if acq_clamped.sum() == 0:
            print(
                "Warning: all acquisition values are zero after clamping; falling back to uniform sampling"
            )
            probs = torch.ones(len(available), device=candidates.device) / len(available)
        else:
            probs = acq_clamped / acq_clamped.sum()

        n_nonzero = (probs > 0).sum().item()
        if n_nonzero < self.q:
            print(
                f"Warning: only {n_nonzero} non-zero probability candidates, falling back to replacement sampling"
            )
            local_idx = torch.multinomial(probs, num_samples=self.q, replacement=True)
        else:
            local_idx = torch.multinomial(probs, num_samples=self.q, replacement=False)

        chosen_indices = [available[i] for i in local_idx.tolist()]
        return OptimizationResult(
            new_X=candidates[chosen_indices],
            acq_val=acq_values[local_idx],
            observed_indices=chosen_indices,
        )
