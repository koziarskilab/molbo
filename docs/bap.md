# BAP oracle

Supports setting different binding-affinity and docking-based models as BO oracle.

## Adapter options

These options are handled by MolBO. Other keywords are forwarded to the selected
BAP pipeline through `**pipeline_options`.

| Option | Default | Meaning |
| --- | --- | --- |
| `target_name` | Required | Key in BAP's `TARGET_REGISTRY`; one target per adapter. |
| `results_cache_dir` | `.cache/<model>_results_cache` | Prediction-results cache, where `<model>` is `boltz`, `flashbind`, or `nesso`. |
| `results_cache_override` | `"failed"` | Retry cached failures. `"none"` retains cached results; `"all"` recomputes them. Cache misses are evaluated in every mode. |
| `ligand_efficiency` | `False` | Divide the larger-is-better objective by the input molecule's heavy-atom count, without a score shift. |

The base `BAPOracle` also requires `pipeline_name` and an explicit
`results_cache_dir`. The named model adapters select their own pipeline.

### Objectives and failures

| Adapter | Pipeline | Larger-is-better objective |
| --- | --- | --- |
| `BoltzOracle` | `boltz` | BAP's aggregated `affinity_pred_value`, already negated by its pipeline. |
| `NessoOracle` | `nesso` | BAP's aggregated `affinity_pred_value`, already negated by its pipeline. |
| `FlashBindOracle` | `flashbind` | Negative aggregated `flashbind_pred_value`; the adapter applies this negation. |

With LE enabled, the returned objective is `objective / heavy_atom_count`.
Invalid SMILES or zero heavy atoms yield NaN. The transform happens after BAP
returns its results, so the same raw prediction cache can serve raw and LE runs.

Missing or non-finite objectives return NaN. In `FixedPoolBOLoop`, failed scores
consume evaluation budget and their indices remain observed, but they do not
train the GP. Retrying cached failures on a later oracle call is separate from
BO selection: an observed pool index is not selected again within the run.

## Shared BAP options

The following defaults come from the local BAP registry checked on 2026-11-05.
Experiment configs may override them. The wrappers inherit BAP defaults for
omitted pipeline options; they do not reproduce every default in older lab
wrappers. For reproducibility, set the relevant options explicitly.

| Option | BAP default | Meaning |
| --- | --- | --- |
| `n_parallel_jobs` | `0` | Number of BAP workers; `0` runs serially. |
| `gpus_list` | `"auto"` | GPU IDs visible to the process, e.g. `[0, 1]`. `"auto"` uses available GPUs with a CPU fallback; `"all"` requires CUDA; `None` selects CPU. Model-specific device settings still apply. |
| `batch_size` | Boltz/FlashBind: `4`; NESSO: `0` | Ligands per model invocation within each worker. Values <= 0 disable subdivision. |
| `aggregation_methods` | Boltz/FlashBind: `"median"`; NESSO: `"mean"` | Reduction of per-sample/model scores: `"mean"`, `"median"`, `"min"`, or `"max"`. |

Use a single aggregation method. Multiple methods make BAP add suffixes to score
keys, while these adapters read the unqualified objective key.

`batch_size` is separate from BO's `q`, the number of molecules selected per
round. Acquisition chunk size controls GP scoring memory and does not change
either model-invocation batch size or BO batch size.

Use BAP's keyword names: `cache_dir`, `gpus_list`, and `aggregation_methods`.
Legacy aliases such as `model_cache_dir`, `gpus`, and `aggregation_method` are
not translated by the current adapters.

## Boltz additional parameters

| Option | BAP default | Meaning |
| --- | --- | --- |
| `diffusion_samples` | `5` | Structure/affinity diffusion samples per ligand. |
| `condition_on_residues` | `False` | Apply pocket constraints from the registered target. |
| `msa_source` | `"cache"` | Use cached MSAs, `"server"` for server retrieval, or `None` to disable MSAs. |
| `boltz_env` | BAP's `external/envs/boltz/.venv` | Independent Boltz environment directory. Omit the argument to use the default. |
| `cache_dir` | `"external/boltz_cache"` | Model assets; separate from `results_cache_dir`. |

Boltz resolves a supplied relative environment path from the working directory;
its relative model-cache path also uses that directory. Prefer absolute paths
for custom environments and caches. In particular, the older lab wrapper used
pocket conditioning and mean aggregation by default; set
`condition_on_residues=True, aggregation_methods="mean"` to request those settings.

## FlashBind additional parameters

| Option | BAP default | Meaning |
| --- | --- | --- |
| `flashbind_root` | `"external/flashbind"` | FlashBind checkout and model resources. |
| `flashbind_env` | BAP's `external/envs/flashbind/.venv` | FlashBind environment directory. |
| `fabind_env` | BAP's `external/envs/fabind/.venv` | FABind+ environment directory. |
| `checkpoints` | `["checkpoints/value_1.ckpt", "checkpoints/value_2.ckpt"]` | Value-model ensemble checkpoints. |
| `cache_dir` | `"external/flashbind_cache"` | Persistent target-dependent representations. |
| `pose_cache_dir` | `None` | Optional persistent FABind+ pose cache; separate from prediction results. |
| `work_dir` | `None` | Optional parent directory for temporary inputs. |
| `fabind_batch_size` | `4` | FABind+ inference batch size. |
| `fabind_post_optim_workers` | `1` | Workers for FABind+ coordinate refinement; values above one need supporting runner code. |
| `fabind_num_threads` | `0` | FABind+ molecule-preprocessing threads. |
| `torchdrug_n_jobs` | `-1` | TorchDrug representation workers. |
| `esm3_model` | `"esm3_sm_open_v1"` | Protein representation model. |
| `esm3_device` | `"auto"` | Device for ESM3 representations. |
| `devices` | `1` | Lightning device count per invocation. |
| `nproc_per_node` | `None` | torchrun process count; defaults to `devices`. |
| `accelerator` | `"gpu"` | Lightning accelerator. |
| `num_workers` | `8` | FlashBind dataloader workers. |
| `distance_threshold` | `20.0` | Protein pocket-crop distance threshold. |
| `include_auxiliary_scores` | `False` | Include auxiliary BAP score columns; the adapter still returns one objective. |

The FlashBind registry resolves relative repository, environment, cache, pose,
and work paths against the BAP repository root. Its multi-target settings
(`targets_aggregation_methods`, `run_targets_in_parallel`) do not affect these
single-target adapters.

## Nesso additional parameters

| Option | BAP default | Meaning |
| --- | --- | --- |
| `nesso_env` |`"external/nesso/.venv"`  | Nesso environment directory. |
| `cache_dir` | `"external/nesso_cache"` | Model and CCD assets; `None` honors `NESSO_CACHE`. |
| `work_dir` | `None` | Reusable target embeddings and working files; defaults to the resolved model cache. |
| `checkpoint` | `None` | Optional local model directory. |
| `ccd` | `None` | Optional local chemical-component dictionary. |
| `precision` | `"bf16-mixed"` | Lightning inference precision. |
| `recycling_steps` | `5` | NESSO recycling steps. |
| `num_workers` | `2` | Preprocessing/dataloader workers per invocation. |
| `no_kernels` | `True` | Disable optional CUDA kernels. |
| `seed` | `0` | Prediction seed; separate from BO's seed unless explicitly linked in configuration. |
| `refine_protein_inference` | `True` | Enable the refined protein-pocket pass. |
| `refine_protein_cutoff` | `22.0` | Refinement distance cutoff in angstroms. |
| `refine_protein_tokens_budget` | `256` | Protein token budget for refinement. |
| `affinity_protein_cutoff` | `15.0` | Affinity-head distance cutoff in angstroms. |
| `reject_zero_entropy` | `True` | Treat zero protein–ligand interface entropy as a failure. |
| `save_metadata` | `False` | Retain upstream tensor artifacts. |
| `timeout` | `None` | Subprocess timeout in seconds; `None` imposes no timeout. |

The NESSO registry resolves supplied relative environment, model-cache, work,
checkpoint, and CCD paths against the BAP repository root.
