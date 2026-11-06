from pathlib import Path

from molbo.oracles.bap import BAPOracle


class FlashBindOracle(BAPOracle):
    """Use BAP's FlashBind/FABind+ pipeline with negated FlashBind scores for higher-is-better optimization.

    Additional options are forwarded to BAP's registered pipeline unchanged.
    """

    def __init__(
        self,
        target_name: str,
        results_cache_dir: str | Path = ".cache/flashbind_results_cache",
        results_cache_override: str = "failed",
        **pipeline_options,
    ):
        super().__init__(
            target_name=target_name,
            pipeline_name="flashbind",
            results_cache_dir=results_cache_dir,
            results_cache_override=results_cache_override,
            **pipeline_options,
        )

    def _objective(self, score_dict: dict) -> float:
        return -score_dict.get("flashbind_pred_value", float("nan"))
