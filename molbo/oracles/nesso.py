from pathlib import Path

from molbo.oracles.bap import BAPOracle


class NessoOracle(BAPOracle):
    """Use BAP's NESSO pipeline (higher-is-better).

    Additional options are forwarded to BAP's registered pipeline unchanged.
    """

    def __init__(
        self,
        target_name: str,
        results_cache_dir: str | Path = ".cache/nesso_results_cache",
        results_cache_override: str = "failed",
        **pipeline_options,
    ):
        super().__init__(
            target_name=target_name,
            pipeline_name="nesso",
            results_cache_dir=results_cache_dir,
            results_cache_override=results_cache_override,
            **pipeline_options,
        )
