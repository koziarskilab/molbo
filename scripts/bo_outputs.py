import csv
import json
from pathlib import Path

import numpy as np
import torch

RUN_OUTPUTS = (
    "config.yaml",
    "candidates.csv",
    "metric_inputs.json",
    "history.pt",
    "evaluations.csv",
    "metrics.csv",
    "summary.json",
)


def convert_json(value):
    """Convert Torch/NumPy data and non-finite values for JSON and CSV exports."""
    if torch.is_tensor(value):
        return convert_json(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return convert_json(value.tolist())
    if isinstance(value, np.generic):
        return convert_json(value.item())
    if isinstance(value, dict):
        return {key: convert_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [convert_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(convert_json(rows))


def export_run(history, output_dir, candidates=None, summary=None):
    """Export history checkpoint to files"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(summary or {})
    attempts, successes = history.n_attempted, history.n_successful
    indices = history.successful_indices
    mapping_known = indices is not None and len(indices) == successes
    scores, acquisitions = {}, {}
    if mapping_known:
        scores = dict(zip(indices, history.y_all.flatten().tolist()))
        acquisitions = dict(zip(indices[len(history.y_init) :], history.acq_vals))

    evaluations = []
    for attempt, index in enumerate(history.observed_indices, 1):
        phase = None
        if history.n_initial_attempts is not None:
            phase = "initialization" if attempt <= history.n_initial_attempts else "optimization"
        entry = dict(
            attempt=attempt,
            pool_index=index,
            phase=phase,
            objective=scores.get(index),
            success=index in scores if mapping_known else None,
            acquisition_value=acquisitions.get(index),
        )
        if candidates is not None:
            candidate = candidates[index]
            entry.update(candidate_id=candidate["candidate_id"], smiles=candidate["smiles"])
        evaluations.append(entry)
    write_csv(output / "evaluations.csv", evaluations)

    write_csv(output / "metrics.csv", history.metrics_history)

    summary.update(
        attempted_evaluations=attempts,
        successful_evaluations=successes,
        failed_evaluations=attempts - successes,
        best_objective=history.best_observed if successes else None,
        completed_iterations=len(history.iteration),
    )
    (output / "summary.json").write_text(
        json.dumps(convert_json(summary), indent=2, allow_nan=False) + "\n"
    )
