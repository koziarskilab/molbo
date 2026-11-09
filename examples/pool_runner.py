"""Pool-based BO using Hydra config and runner."""

from pathlib import Path

from hydra import compose, initialize_config_dir

from scripts.bo_runner import run_bo

smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCOC", "CC(C)O", "CC(=O)C", "c1ccncc1"]
output = Path("runs/pool_runner")
output.mkdir(parents=True)
pool = output / "pool.csv"
pool.write_text("smiles\n" + "\n".join(smiles) + "\n")

with initialize_config_dir(
    version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")
):
    cfg = compose(
        config_name="pool",
        overrides=[
            f"data.path={pool}",
            "data.id_column=null",
            "bo.n_init=2",
            "bo.q=2",
            "bo.evaluation_budget=6",
        ],
    )

history = run_bo(cfg, output)
print("Best observed QED:", history.best_observed)
