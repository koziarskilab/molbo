from pathlib import Path

import torch
from rdkit import Chem

from molbo.oracles.molecular import MolecularOracle


class BAPOracle(MolecularOracle):
    """Evaluate SMILES with a registered BAP pipeline.

    BAP is an optional lab dependency, imported only when constructing an oracle.
    Pipelines supply the larger-is-better objective; non-finite scores become NaNs.
    """

    def __init__(
        self,
        target_name: str,
        pipeline_name: str,
        results_cache_dir: str | Path,
        results_cache_override: str = "failed",
        ligand_efficiency: bool = False,
        **pipeline_options,
    ):
        """Construct a registered target and pipeline with a disk results cache.

        Args:
            target_name: Target key in BAP's TARGET_REGISTRY.
            pipeline_name: Pipeline key in BAP's PIPELINE_REGISTRY.
            results_cache_dir: Directory for cached prediction results.
            results_cache_override: "failed" retries cached failures, "none" retains cached results, and "all" recomputes results. Cache misses are always re-evaluated.
            ligand_efficiency: Divide the oriented objective by heavy-atom count, without a shift. Invalid SMILES or zero heavy atoms yield NaN.
            pipeline_options: Options forwarded to the registered pipeline, such as model environments, GPUs, batch sizes, and aggregation.
        """
        super().__init__()

        if results_cache_override not in {"all", "none", "failed"}:
            raise ValueError("results_cache_override must be one of 'all', 'none', or 'failed'")

        from bap.benchmark.registers.pipelines import PIPELINE_REGISTRY
        from bap.benchmark.registers.targets import TARGET_REGISTRY
        from bap.pipeline.results_caches.disk_results_cache import GlobalDatasetDiskResultsCache

        if pipeline_name not in PIPELINE_REGISTRY:
            raise ValueError(f"Unknown BAP pipeline {pipeline_name!r}")
        if target_name not in TARGET_REGISTRY:
            raise ValueError(f"Unknown BAP target {target_name!r}")

        self.pipeline = PIPELINE_REGISTRY[pipeline_name](
            targets=[TARGET_REGISTRY[target_name]()], **pipeline_options
        )
        self.results_cache = GlobalDatasetDiskResultsCache(results_cache_dir)
        self.results_cache_override = results_cache_override
        self.ligand_efficiency = ligand_efficiency

    def _objective(self, score_dict: dict) -> float:
        return score_dict.get("affinity_pred_value", float("nan"))

    def _evaluate(self, smiles: list[str]) -> torch.Tensor:
        if not smiles:
            return torch.empty((0, 1), dtype=torch.float64)

        results = self.pipeline.run(
            smiles_list=smiles,
            results_cache=self.results_cache,
            results_cache_override=self.results_cache_override,
            progress_bar=False,
        )
        if len(results) != len(smiles):
            raise RuntimeError("BAP returned a different number of results than input SMILES")

        values = torch.tensor(
            [self._objective(result.score_dict or {}) for result in results], dtype=torch.float64
        ).unsqueeze(-1)

        if self.ligand_efficiency:
            counts = []
            for smi in smiles:
                molecule = Chem.MolFromSmiles(smi)
                counts.append(molecule.GetNumHeavyAtoms() if molecule is not None else 0)
            heavy_atoms = torch.tensor(counts, dtype=values.dtype).unsqueeze(-1)
            values = (values / heavy_atoms).masked_fill(heavy_atoms == 0, float("nan"))

        return values.masked_fill(~torch.isfinite(values), float("nan"))
