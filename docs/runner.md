# BO runner

See the [pool BO example](../examples/pool_runner.py) for a minimal Python run.
Install the optional runner dependencies with `pip install -e '.[hydra]'`.

## Hydra

The default is [pool.yaml](../configs/pool.yaml); `--config-name continuous`
selects [continuous.yaml](../configs/continuous.yaml). Components are configured
under [configs/](../configs/):

| Directory | Purpose |
| --- | --- |
| [acquisition/](../configs/acquisition/) | Acquisition function and its parameters. |
| [featurizer/](../configs/featurizer/) | Morgan, CheMeleon, or precomputed molecular features. |
| [kernel/](../configs/kernel/) | Tanimoto, MinMax, linear, or RBF GP kernels. |
| [logger/](../configs/logger/) | Optional W&B logging. |
| [metrics/](../configs/metrics/) | Metrics for observations, reference scores, and activity labels. |
| [optimizer/](../configs/optimizer/) | Acquisition optimization over the candidate pool. |
| [oracle/](../configs/oracle/) | Objective scoring or saved-score replay; see [BAP options](bap.md). |
| [surrogate/](../configs/surrogate/) | GP or GNEprop model settings. |

- **Sweeps:** `-m kernel=tanimoto,minmax bo.seed=0,1` runs all four combinations.
- **Output directory:** defaults to `runs/<pool>/<overrides>/seed<N>`; existing results are protected.

## Metrics

- `metrics=observed` (default): best observed score.
- `metrics=reference`: top-k scores, AUC, regret, ranking recall, and prediction error.
- `metrics=screening`: reference metrics plus recall, precision, and enrichment from activity labels.

Define score and label columns under `evaluation.reference` and `evaluation.labels`;
separate tables align by SMILES. Screening uses `evaluation.labels.threshold` and
`evaluation.labels.direction` (`le` or `ge`). Use `multiplier: -1` for reference
scores where lower is better.

Custom metrics extend `Metric` and are listed under `metrics`; `metrics=[]`
disables extra metrics. Each BO iteration records one `surrogate/loss` value.
Add `logger=wandb` to log the same scalars to W&B.

## Outputs

- `config.yaml`, `candidates.csv`, `metric_inputs.json`: configuration and pool inputs.
- `history.pt`: observations, metrics, and latest predictions.
- `evaluations.csv`: attempted pool evaluations, including failures.
- `metrics.csv`: scalar metrics, one row per reporting event.
- `summary.json`: run status, evaluation counts, and best score.

`bo.checkpoint_freq` controls iteration saves.
