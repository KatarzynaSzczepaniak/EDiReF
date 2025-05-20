from typing import Any, Dict, List

import numpy as np

from .utilities import get_project_root


def write_confusion_matrix(cm: np.ndarray, labels: List[str], title: str) -> List[str]:
    """
    Format a confusion matrix as a list of strings for saving to a .txt file.

    Args:
        title (str): Title to be displayed above the matrix.
        cm (np.ndarray): Confusion matrix of shape (n_classes, n_classes).
        labels (List[str]): Class labels corresponding to matrix rows/columns.

    Returns:
        List[str]: A list of strings representing the formatted confusion matrix,
            ready to be written to a plain text file.
    """
    cm2txt = []
    col_width = max(len(str(label)) for label in labels) + 1

    cm2txt.append(f"\n{title}\n")

    # Header row
    header = f"{'':<{col_width}}" + "".join(f"{label:<{col_width}}" for label in labels)
    cm2txt.append(header + "\n")

    # Matrix rows
    for i, label in enumerate(labels):
        row = f"{label:<{col_width}}" + "".join(
            f"{value:<{col_width}}" for value in cm[i]
        )
        cm2txt.append(row + "\n")

    return cm2txt


def _save_experiment(
    file_name: str,
    dataset: str,
    setup_lines: List[str],
    logs: List[str],
    metrics: Dict[str, Any],
    emotion_cm: np.ndarray,
    trigger_cm: np.ndarray,
    labels_to_ids: Dict[str, int],
) -> None:
    """
    Save an experiment under specified file name.
    """
    root = get_project_root()
    file_path = root / "results" / dataset / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists():
        print(f"⚠️ Overwriting existing file: {file_path.name}")

    with file_path.open("w") as f:
        f.writelines(setup_lines)
        f.writelines(logs)
        f.write("\n")

        experiment_results = []
        for task, results in metrics.items():
            experiment_results.append(f"\nTask: {task.upper()}\n")
            for metric, score in results.items():
                if metric == "f1" and isinstance(score, dict):
                    experiment_results.append("      f1:\n")
                    max_label_len = max(len(str(k)) for k in score.keys())
                    for x, y in score.items():
                        experiment_results.append(
                            f"          {x:<{max_label_len}}: {y:.4f}\n"
                        )
                else:
                    experiment_results.append(f"      {metric}: {score:.4f}\n")
        f.writelines(experiment_results)

        f.writelines(
            write_confusion_matrix(
                emotion_cm, list(labels_to_ids), "Emotion confusion matrix"
            )
        )
        f.write("\n")
        f.writelines(
            write_confusion_matrix(
                trigger_cm, ["No trigger", "Trigger"], "Trigger confusion matrix"
            )
        )


def save_moe_experiment(
    gate_count: int,
    weight_triggers: bool,
    dataset: str,
    train_bert: bool,
    gate_type: str,
    num_experts: int,
    expert_type: str,
    top_k: int,
    learning_rate: float,
    num_epochs: int,
    max_length: int,
    batch_size: int,
    logs: List[str],
    metrics: Dict[str, Any],
    emotion_cm: np.ndarray,
    trigger_cm: np.ndarray,
    labels_to_ids: Dict[str, int],
) -> None:
    """
    Save full experiment metadata, training logs, performance metrics,
    and confusion matrices to a .txt file.

    This function creates a uniquely named log file under the `results/` directory
    based on the experiment configuration (e.g., model architecture, number of
    experts, learning rate). It logs:

    - Experimental setup metadata (e.g., batch size, learning rate, top-k, etc.)
    - Training and validation logs (loss and accuracy per epoch)
    - Final aggregated metrics (accuracy, precision, recall, F1)
    - Confusion matrices for both emotion and trigger classification

    Args:
        gate_count (int): 1 for single gating, 2 for dual gating.
        weight_triggers (bool): Whether to apply weight to trigger classification.
        dataset (str): The name of the dataset to use ("MELD", or "MaSaC").
        train_bert (bool): Whether to fine-tune the pretrained BERT model.
        gate_type (str): Type of gating mechanism.
        num_experts (int): Number of expert modules.
        expert_type (str): Type of expert layer.
        top_k (int): Number of top experts to use per forward pass.
        learning_rate (float): Learning rate for optimizer.
        num_epochs (int): Number of training epochs.
        max_length (int): Maximum length to pad/truncate each sequence to.
        batch_size (int): Number of samples per batch.
        logs (List[str]): A list of formatted strings logging training and validation
            progress for each epoch and key steps within.
        metrics (Dict[str, Any]): Dictionary with aggregated performance metrics.
        emotion_cm (np.ndarray): Confusion matrix for emotion classification.
        trigger_cm (np.ndarray): Confusion matrix for trigger classification.
        labels_to_ids (Dict[str, int]): Dictionary mapping strings into integers.
    """
    # Construct file path
    model_name_prefix = (
        f"weighted_moe_model" if weight_triggers and dataset == "MaSaC" else "moe_model"
    )
    num_gates = "single" if gate_count == 1 else "dual"
    file_name = (
        f"{model_name_prefix}_{train_bert}_train_bert_{gate_type}_{num_gates}_gate_"
        f"{num_experts}_{expert_type}_experts_{top_k}_active_"
        f"{learning_rate}_lr_{num_epochs}_epochs.txt"
    )

    setup = [
        f"max_length = {max_length}\n",
        f"batch_size = {batch_size}\n\n",
        f"gate_type = {gate_type}\n",
        f"expert_type = {expert_type}\n",
        f"num_experts = {num_experts}\n",
        f"top_k = {top_k}\n",
        f"learning_rate = {learning_rate}\n",
        f"num_epochs = {num_epochs}\n",
        f"train_bert = {train_bert}\n",
        f"weight_triggers = {weight_triggers}\n\n",
    ]

    _save_experiment(
        file_name, dataset, setup, logs, metrics, emotion_cm, trigger_cm, labels_to_ids
    )


def save_reference_experiment(
    weight_triggers: bool,
    dataset: str,
    train_bert: bool,
    learning_rate: float,
    num_epochs: int,
    max_length: int,
    batch_size: int,
    logs: List[str],
    metrics: Dict[str, Any],
    emotion_cm: np.ndarray,
    trigger_cm: np.ndarray,
    labels_to_ids: Dict[str, int],
) -> None:
    """
    Save full experiment metadata, training logs, performance metrics,
    and confusion matrices to a .txt file.

    This function creates a uniquely named log file under the `results/` directory
    based on the experiment configuration (e.g., model architecture, number of
    experts, learning rate). It logs:

    - Experimental setup metadata (e.g., batch size, learning rate, top-k, etc.)
    - Training and validation logs (loss and accuracy per epoch)
    - Final aggregated metrics (accuracy, precision, recall, F1)
    - Confusion matrices for both emotion and trigger classification

    Args:
        weight_triggers (bool): Whether to apply weight to trigger classification.
        dataset (str): The name of the dataset to use ("MELD", or "MaSaC").
        train_bert (bool): Whether to fine-tune the pretrained BERT model.
        learning_rate (float): Learning rate for optimizer.
        num_epochs (int): Number of training epochs.
        max_length (int): Maximum length to pad/truncate each sequence to.
        batch_size (int): Number of samples per batch.
        logs (List[str]): A list of formatted strings logging training and validation
            progress for each epoch and key steps within.
        metrics (Dict[str, Any]): Dictionary with aggregated performance metrics.
        emotion_cm (np.ndarray): Confusion matrix for emotion classification.
        trigger_cm (np.ndarray): Confusion matrix for trigger classification.
        labels_to_ids (Dict[str, int]): Dictionary mapping strings into integers.
    """
    model_prefix = (
        "weighted_reference_model"
        if weight_triggers and dataset == "MaSaC"
        else "reference_model"
    )
    file_name = (
        f"{model_prefix}_{train_bert}_train_bert_"
        f"{learning_rate}_lr_{num_epochs}_epochs.txt"
    )

    setup = [
        f"max_length = {max_length}\n",
        f"batch_size = {batch_size}\n\n",
        f"learning_rate = {learning_rate}\n",
        f"num_epochs = {num_epochs}\n",
        f"train_bert = {train_bert}\n",
        f"weight_triggers = {weight_triggers}\n\n",
    ]

    _save_experiment(
        file_name, dataset, setup, logs, metrics, emotion_cm, trigger_cm, labels_to_ids
    )
