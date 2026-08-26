"""
Central configuration for the steel-defect detection project.

Import from here in both the notebook driver and the headless training job so
paths, class map, and hyperparameters live in one place.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict
import torch

_ROOT_MARKERS = ("data/raw", "requirements.txt", ".git")

def find_project_root(start: Path, markers=_ROOT_MARKERS) -> Path:
    """Walk upward from *start* until a directory containing one of *markers*.
    Lets callers (notebooks, headless jobs) resolve the repo root from any
    working directory instead of hardcoding an absolute path.
    """
    start = Path(start).resolve()
    for base in (start, *start.parents):
        if any((base / m).exists() for m in markers):
            return base
    return start

# Primary: this file lives at <root>/src/utils/config.py, so parents[2] is the
# repo root. Fall back to a marker search if the tree has been relocated so the
# module still resolves correctly for anyone who clones/moves the repo.
_ROOT_BY_LAYOUT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = (
    _ROOT_BY_LAYOUT
    if (_ROOT_BY_LAYOUT / "data" / "raw").exists()
    else find_project_root(Path(__file__).parent)
)

CLASS_MAP: Dict[str, int] = {
    "crazing": 1,
    "inclusion": 2,
    "patches": 3,
    "pitted_surface": 4,
    "rolled-in_scale": 5,
    "scratches": 6,
}
CLASS_NAMES = list(CLASS_MAP.keys())
NUM_CLASSES = len(CLASS_MAP) + 1  # +1 for background

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@dataclass
class Config:
    # Paths
    project_root: Path = PROJECT_ROOT
    data_root: Path = PROJECT_ROOT / "data" / "raw"
    train_img: Path = PROJECT_ROOT / "data" / "raw" / "train_images"
    train_ann: Path = PROJECT_ROOT / "data" / "raw" / "train_annotations"
    val_img: Path = PROJECT_ROOT / "data" / "raw" / "valid_images"
    val_ann: Path = PROJECT_ROOT / "data" / "raw" / "valid_annotations"

    models_dir: Path = PROJECT_ROOT / "models"
    visualizations_dir: Path = PROJECT_ROOT / "visualizations"     

    # Model
    num_classes: int = NUM_CLASSES
    backbone_pretrained: bool = True

    # Training
    batch_size: int = 24
    num_epochs: int = 50
    learning_rate: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    grad_clip_norm: float = 10.0

    # LR schedule
    step_size: int = 8
    gamma: float = 0.5

    # Early stopping
    early_stopping_patience: int = 8

    # Hardware
    device: torch.device = field(default_factory=pick_device)
    num_workers: int = 8
    pin_memory: bool = False

    # Reproducibility / output
    seed: int = 42
    save_plots: bool = True
    plot_interval: int = 4

    def to_serializable_dict(self) -> dict:
        """dataclass -> plain dict (Paths → str, device → str) for JSON dumps."""
        raw = asdict(self)
        for k, v in raw.items():
            if isinstance(v, Path):
                raw[k] = str(v)
            elif isinstance(v, torch.device):
                raw[k] = str(v)
        return raw

def set_seed(seed: int) -> None:
    """Seed python-random, numpy, torch (CPU + CUDA) for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn — ensures augmentation streams are deterministic."""
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)