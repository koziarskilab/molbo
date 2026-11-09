from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from molbo.utils.helpers import morgan_fp


class Featurizer(ABC):
    """Maps a candidate representation to the tensor a model's kernel consumes.

    A model owns a Featurizer instance and applies it internally. This keeps the
    rest of the BO stack (History, BOLoop, acquisition optimizers) agnostic to
    the candidate representation - they pass identity-space X around, and only
    the model turns it into features.
    """

    @abstractmethod
    def __call__(self, X) -> torch.Tensor:
        """Featurize a batch of candidates into an (n, d) tensor."""
        ...


class IdentityFeaturizer(Featurizer):
    """Pass-through featurizer for domains where X is already a feature tensor.

    Used by continuous-domain models - the candidate representation and the
    kernel input are the same thing.
    """

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X


class MorganFingerprintFeaturizer(Featurizer):
    """Featurizes molecules into Morgan fingerprints.

    Accepts SMILES strings or RDKit Mol objects. Default is count-based fingerprints;
    see morgan_fp().

    Args:
        radius: Morgan radius (default 2, i.e. ECFP4).
        fp_size: Fingerprint length in bits.
    """

    def __init__(self, radius: int = 2, fp_size: int = 2048):
        self.radius = radius
        self.fp_size = fp_size

    def __call__(self, X: list[str | Chem.Mol]) -> torch.Tensor:
        return torch.stack([morgan_fp(x, radius=self.radius, fp_size=self.fp_size) for x in X])


class CheMeleonFeaturizer(Featurizer):
    """Frozen CheMeleon fingerprints for SMILES strings or RDKit molecules.

    Adapted from https://github.com/JacksonBurns/chemeleon/blob/main/chemeleon_fingerprint.py.

    Args:
        checkpoint: Path to local CheMeleon message-passing weights.
        device: Device for fingerprint generation; None uses the model default.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device | None = None,
    ):
        from chemprop import featurizers, nn
        from chemprop.models import MPNN

        weights = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=True)

        # Loading unused initial weights must not change BO's RNG sequence.
        with torch.random.fork_rng(devices=[]):
            mp = nn.BondMessagePassing(**weights["hyper_parameters"])
            mp.load_state_dict(weights["state_dict"])
            self.model = MPNN(mp, nn.MeanAggregation(), nn.RegressionFFN(input_dim=mp.output_dim))

        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.model.eval().requires_grad_(False)
        if device is not None:
            self.model.to(device=device)

    def __call__(self, X: list[str | Chem.Mol]) -> torch.Tensor:
        from chemprop.data import BatchMolGraph

        batch = BatchMolGraph(
            [self.featurizer(Chem.MolFromSmiles(m) if isinstance(m, str) else m) for m in X]
        )
        batch.to(device=self.model.device)
        with torch.no_grad():
            return self.model.fingerprint(batch).cpu()


class PrecomputedFeaturizer(Featurizer):
    """Load precomputed features for a set of SMILES strings."""

    def __init__(self, path: str, smiles_col: str = "smiles", features_col: str = "features"):
        path = Path(path).expanduser().resolve(strict=True)

        with np.load(path, allow_pickle=False) as archive:
            smiles, features = archive[smiles_col], archive[features_col]

        self.indices = {str(s): i for i, s in enumerate(smiles)}
        self.features = torch.tensor(features, dtype=torch.float64)

    def __call__(self, X):
        return self.features[[self.indices[s] for s in X]].clone()
