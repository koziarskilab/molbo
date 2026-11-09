from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Metric(ABC):
    """Base class for Metrics."""

    requires_predictions = False

    def __init__(self, *, name):
        self.name = name

    @abstractmethod
    def compute(self, history) -> dict:
        pass


class MetricsList(list):
    """Container for Metric objects."""

    def __init__(self, metrics: Sequence[Metric] = ()):
        super().__init__(metrics)
        names = [metric.name for metric in self]
        if len(names) != len(set(names)):
            raise ValueError("Metric names must be unique")

    def compute_metrics(self, history):
        return {
            f"{metric.name}/{key}": value
            for metric in self
            for key, value in metric.compute(history).items()
        }


def _values(values, candidate_ids):
    """Copy numeric values supplied in candidate-pool order."""
    array = np.array(values, dtype=float)
    if array.ndim != 1 or (candidate_ids is not None and len(array) != len(candidate_ids)):
        raise ValueError("Metric inputs must align with the candidate pool")
    return array


def _retrieval(indices, labels, threshold, direction):
    labeled = np.isfinite(labels)
    active = labeled & ((labels <= threshold) if direction == "le" else (labels >= threshold))
    n_active, n_labeled = int(active.sum()), int(labeled.sum())
    hits, selected_labeled = int(active[indices].sum()), int(labeled[indices].sum())
    precision = hits / selected_labeled if selected_labeled else None

    return dict(
        hits=hits,
        total_actives=n_active,
        labeled_selected=selected_labeled,
        total_labeled=n_labeled,
        recall=hits / n_active if n_active else None,
        precision=precision,
        enrichment=(
            precision / (n_active / n_labeled) if precision is not None and n_active else None
        ),
    )


class BestObserved(Metric):
    def __init__(self, name="best"):
        super().__init__(name=name)

    def compute(self, history):
        return {"value": history.best_observed if history.n_successful else None}


class TopKMean(Metric):
    def __init__(self, k=10, name=None):
        super().__init__(name=name or f"top{k}")
        self.k = k

    def compute(self, history):
        y = history.y_all.flatten()
        return {"value": y.topk(self.k).values.mean().item() if len(y) >= self.k else None}


class SimpleRegret(Metric):
    def __init__(self, reference=None, optimum=None, candidate_ids=None, name="regret"):
        super().__init__(name=name)
        values = _values(reference, candidate_ids) if reference is not None else np.array([])
        finite = values[np.isfinite(values)]
        self.optimum = (
            optimum if optimum is not None else float(finite.max()) if len(finite) else None
        )

    def compute(self, history):
        return {
            "value": (
                self.optimum - history.best_observed
                if self.optimum is not None and history.n_successful
                else None
            )
        }


class TopKMeanAUC(TopKMean):
    """Left-step top-k area per attempt after initialization, normalized by duration."""

    def __init__(self, k=10, name=None):
        super().__init__(k=k, name=name or f"top{k}_auc")

    def compute(self, history):
        start = history.n_initial_attempts
        if (
            start is None
            or history.successful_indices is None
            or len(history.successful_indices) != history.n_successful
        ):
            return {"value": None}
        end = len(history.observed_indices)
        best = sorted(history.y_init.flatten().tolist(), reverse=True)[: self.k]
        if len(best) < self.k:
            return {"value": None, "area": None, "start": start, "end": end}

        scores = dict(zip(history.successful_indices, history.y_all.flatten().tolist()))
        mean = float(np.mean(best))
        area = 0.0
        for index in history.observed_indices[start:]:
            area += mean
            if index in scores:
                best = sorted([*best, scores[index]], reverse=True)[: self.k]
                mean = float(np.mean(best))
        return {
            "value": area / (end - start) if end > start else None,
            "area": area,
            "start": start,
            "end": end,
        }


class QueriedRecall(Metric):
    def __init__(self, labels, threshold, direction="le", candidate_ids=None, name="queried"):
        super().__init__(name=name)
        if direction not in {"le", "ge"}:
            raise ValueError("direction must be le or ge")
        self.labels = _values(labels, candidate_ids)
        self.threshold, self.direction = threshold, direction

    def compute(self, history):
        return _retrieval(
            sorted(set(history.observed_indices)), self.labels, self.threshold, self.direction
        )


class PosteriorMeanRecall(QueriedRecall):
    requires_predictions = True

    def __init__(self, labels, threshold, k=50, name=None, *, direction="le", candidate_ids=None):
        super().__init__(
            labels,
            threshold,
            direction=direction,
            candidate_ids=candidate_ids,
            name=name or f"recall@{k}",
        )
        self.k = k

    def compute(self, history):
        predictions = history.predictions
        indices = predictions["order"][: self.k]

        return {
            "k": len(indices),
            **_retrieval(indices, self.labels, self.threshold, self.direction),
        }


class ReferenceTopKRecall(Metric):
    requires_predictions = True

    def __init__(self, reference, k=50, candidate_ids=None, name=None):
        super().__init__(name=name or f"reference@{k}")
        self.reference = _values(reference, candidate_ids)
        self.k = k
        ids = list(range(len(self.reference))) if candidate_ids is None else candidate_ids
        finite = np.flatnonzero(np.isfinite(self.reference))
        self.reference_order = sorted(finite, key=lambda i: (-self.reference[i], ids[i]))

    def compute(self, history):
        predictions = history.predictions
        indices, top = predictions["order"][: self.k], self.reference_order[: self.k]
        hits = len(set(indices) & set(top))
        scores = self.reference[indices]
        scores = scores[np.isfinite(scores)]

        return dict(
            k=len(indices),
            hits=hits,
            reference_count=len(top),
            recall=hits / len(top) if top else None,
            finite_count=len(scores),
            mean=float(scores.mean()) if len(scores) else None,
            best=float(scores.max()) if len(scores) else None,
            top10_mean=(
                float(np.mean(sorted(scores, reverse=True)[:10])) if len(scores) >= 10 else None
            ),
            regret=(
                float(self.reference[self.reference_order[0]] - scores.max())
                if len(scores)
                else None
            ),
        )


class PredictionError(Metric):
    requires_predictions = True

    def __init__(self, reference, candidate_ids=None, name="prediction"):
        super().__init__(name=name)
        self.reference = _values(reference, candidate_ids)

    def compute(self, history):
        from scipy.stats import spearmanr

        predictions = history.predictions
        unseen = np.isfinite(self.reference)
        unseen[history.observed_indices[: predictions["n_attempted"]]] = False
        predicted, target = predictions["mean"][unseen], self.reference[unseen]

        return dict(
            count=len(target),
            rmse=float(np.sqrt(np.mean((predicted - target) ** 2))) if len(target) else None,
            mae=float(np.mean(np.abs(predicted - target))) if len(target) else None,
            spearman=(
                float(spearmanr(predicted, target).statistic)
                if len(target) > 1 and np.ptp(predicted) and np.ptp(target)
                else None
            ),
        )
