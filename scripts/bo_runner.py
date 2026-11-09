"""Hydra setup and reporting for pool and continuous BOLoop runs."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf

from molbo.bo import BOLoop, History
from molbo.bo.metrics import MetricsList
from molbo.oracles.lookup import LookupOracle
from molbo.oracles.on_demand_pool import OnDemandPoolOracle
from molbo.surrogates.gneprop import GNEPropSurrogate
from scripts.bo_outputs import RUN_OUTPUTS, convert_json, export_run, write_csv

OmegaConf.register_new_resolver("pool_label", lambda path: Path(path).stem, replace=True)


def _read_table(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _instantiate(config, path_keys=(), **kwargs):
    paths = {key: to_absolute_path(config[key]) for key in path_keys if config.get(key)}
    return instantiate(config, **paths, _convert_="all", **kwargs)


def load_inputs(cfg):
    """Read each table once and align reference/label vectors to candidate rows."""

    path = to_absolute_path(cfg.data.path)
    tables = {path: _read_table(path)}
    rows = tables[path]

    smiles = [row[cfg.data.smiles_column] for row in rows]
    ids = [row[cfg.data.id_column] if cfg.data.id_column else i for i, row in enumerate(rows)]

    candidates = [
        {**row, "candidate_id": identity, "smiles": smi}
        for row, identity, smi in zip(rows, ids, smiles)
    ]
    values = {}
    for name, spec in cfg.get("evaluation", {}).items():
        source = to_absolute_path(spec.path) if spec.get("path") else path
        if source not in tables:
            tables[source] = _read_table(source)
        if source == path:
            vector = [float(row[spec.column] or "nan") for row in rows]
        else:
            key = spec.get("smiles_column", cfg.data.smiles_column)
            mapping = {row[key]: float(row[spec.column] or "nan") for row in tables[source]}
            vector = [mapping[smi] for smi in smiles]
        values[name] = np.array(vector) * spec.get("multiplier", 1.0)

    return candidates, values


def setup_pool(cfg, candidates, values):
    smiles = [row["smiles"] for row in candidates]
    ids = [row["candidate_id"] for row in candidates]
    surrogate = _instantiate(cfg.surrogate, ("weights_path",))

    if isinstance(surrogate, GNEPropSurrogate):
        features = surrogate.prepare_candidates(smiles)
    else:
        features = _instantiate(cfg.featurizer, ("path", "checkpoint"))(smiles)

    features = features.double()

    if cfg.oracle is None:
        pool = LookupOracle(
            features,
            torch.tensor(values["reference"], dtype=features.dtype).reshape(-1, 1),
            top_k=[],
        )
    else:
        scorer = _instantiate(cfg.oracle, ("results_cache_dir",))
        pool = OnDemandPoolOracle(features, smiles, scorer)

    metrics = []
    for spec in cfg.get("metrics", []):
        inputs = {key: values[spec[key]] for key in ("labels", "reference") if key in spec}
        metrics.append(
            instantiate(
                spec, **inputs, **({"candidate_ids": ids} if inputs else {}), _convert_="all"
            )
        )

    return (
        surrogate,
        instantiate(cfg.acquisition),
        instantiate(cfg.optimizer),
        pool,
        MetricsList(metrics),
    )


def setup_continuous(cfg):
    if cfg.bo.n_iters is None:
        raise ValueError("Continuous BO requires n_iters")

    surrogate = instantiate(cfg.surrogate)
    acquisition = instantiate(cfg.acquisition)
    optimizer = instantiate(cfg.optimizer)
    oracle = instantiate(cfg.oracle).to(cfg.bo.device)
    optimizer.bounds = oracle.bounds
    metrics = MetricsList(instantiate(spec) for spec in cfg.get("metrics", []))

    return surrogate, acquisition, optimizer, oracle, metrics


def report_initial(history):
    """Record observation metrics once, before a fitted model is available."""
    values = MetricsList(
        metric for metric in history.metrics if not metric.requires_predictions
    ).compute_metrics(history)
    if values:
        row = dict(n_attempted=history.n_attempted, n_successful=history.n_successful, **values)
        history.metrics_history.append(row)
        if history.logger is not None:
            history.logger.log(row)


def run_bo(cfg, output_dir, history=None):
    """Configure components, run BO, and export the recorded results."""
    output = Path(output_dir)
    if any((output / name).exists() for name in RUN_OUTPUTS):
        raise FileExistsError("Run outputs already exist; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)

    fresh = history is None
    if fresh:
        torch.manual_seed(cfg.bo.seed)
        np.random.seed(cfg.bo.seed % 2**32)
        random.seed(cfg.bo.seed)

    candidates = None
    if cfg.setting == "pool":
        candidates, values = load_inputs(cfg)
        write_csv(output / "candidates.csv", candidates)
        (output / "metric_inputs.json").write_text(
            json.dumps(convert_json(values), allow_nan=False) + "\n"
        )

    logger = instantiate(cfg.logger) if cfg.get("logger") else None
    try:
        surrogate, acquisition, optimizer, oracle, metrics = (
            setup_pool(cfg, candidates, values) if cfg.setting == "pool" else setup_continuous(cfg)
        )
        checkpoint = output / "history.pt"
        if fresh:
            initial = (
                optimizer.sample_init_on_demand(
                    oracle,
                    cfg.bo.n_init,
                    evaluation_budget=cfg.bo.evaluation_budget,
                    batch_size=optimizer.q,
                    seed=cfg.bo.seed,
                )
                if cfg.setting == "pool"
                else optimizer.sample_init(oracle, cfg.bo.n_init)
            )
            history = History(
                initial.train_X.detach().cpu(),
                initial.train_y.detach().cpu(),
                observed_indices=initial.observed_indices,
                successful_indices=initial.successful_indices,
                metrics=metrics,
                logger=logger,
                checkpoint_path=checkpoint,
                checkpoint_freq=cfg.bo.checkpoint_freq,
            )
            report_initial(history)
            history.save(checkpoint)
        else:
            history.metrics, history.logger = metrics, logger
            history.checkpoint_path, history.checkpoint_freq = checkpoint, cfg.bo.checkpoint_freq
        points = oracle.candidates if cfg.setting == "pool" else None
        loop = BOLoop(
            history=history,
            surrogate=surrogate,
            acq_func=acquisition,
            oracle=oracle,
            acqf_optimizer=optimizer,
            candidates=points,
            device=cfg.bo.device,
            evaluation_budget=cfg.bo.evaluation_budget,
            prediction_batch_size=cfg.reporting.batch_size,
            candidate_ids=(
                [row["candidate_id"] for row in candidates] if points is not None else None
            ),
        )
        loop.run(n_iters=cfg.bo.n_iters)
        history.save(checkpoint)
        summary = dict(
            status="completed",
            termination_reason=loop.termination_reason,
            setting=cfg.setting,
            requested_evaluation_budget=cfg.bo.evaluation_budget,
            requested_iterations=cfg.bo.n_iters,
        )
        if candidates is not None:
            summary.update(
                candidate_count=len(candidates),
                effective_evaluation_budget=(
                    min(cfg.bo.evaluation_budget, len(candidates))
                    if cfg.bo.evaluation_budget is not None
                    else len(candidates)
                ),
            )
        export_run(history, output, candidates=candidates, summary=summary)
        return history
    finally:
        finish = getattr(logger, "finish", None)
        if finish is not None:
            finish()
