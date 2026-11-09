from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from scripts.bo_runner import run_bo


@hydra.main(version_base="1.3", config_path="../configs", config_name="pool")
def main(cfg: DictConfig) -> None:
    run_bo(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
