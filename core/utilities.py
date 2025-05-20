import argparse
from pathlib import Path
import random

import numpy as np

import torch


def set_seed(seed: int = 2024) -> None:
    """
    Set random seeds for reproducibility across NumPy, PyTorch, and CUDA.

    Args:
        seed (int): A seed value to ensure deterministic behaviour.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for experiment configuration.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run experiments for emotion and trigger classification."
    )

    parser.add_argument(
        "dataset", type=str, help="The dataset to use ('MELD', 'MaSaC')."
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the experiment configuration YAML file."
    )
    parser.add_argument(
        "--weight_triggers",
        action="store_true",
        default=False,
        help="Whether to apply weight to trigger classification. Pass flag to apply weighting.",
    )
    parser.add_argument(
        "--train_bert",
        action="store_true",
        default=False,
        help="Whether to set BERT params as trainable. Pass flag to set params to trainable.",
    )

    return parser.parse_args()


def get_project_root() -> Path:
    """
    Returns the project root directory for both scripts and notebooks.

    - If run as a script (via `__file__`), uses the script file's location.
    - If run interactively (e.g., notebook), uses the current working directory.

    Returns:
        Path: Path to the project root.
    """
    file_path = globals().get("__file__")
    if file_path:
        # Assumes the script is directly in EDiReF/scripts/ or
        # the notebook is run from EDiReF/notebooks/
        return Path(file_path).resolve().parent.parent
    else:
        return Path.cwd()
