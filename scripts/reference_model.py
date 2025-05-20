#!/usr/bin/env python3
"""
Script description:
Script for training reference models for emotion classification and emotion recognition
in conversation. Experiment configurations are loaded from YAML files and results
are logged to the "results/" directory.

Usage:
$ python reference_model.py <dataset> <config_path> [--weight_triggers] [--train_bert]
"""

# --- Imports ---

# Standard libraries
from typing import Any, Dict, Tuple
import yaml

# PyTorch
import torch
import torch.nn as nn
from torch import cuda
from torch.optim import AdamW

# Transformers
from transformers import BertTokenizer, BertModel

from core import (
    get_data,
    train_val_test_split,
    create_dataloader,
    save_reference_experiment,
    train_and_validate,
    get_metrics,
    set_seed,
    parse_arguments,
)

# --- Model Definition ---


class EDiReFReferenceModel(nn.Module):
    """
    A reference model for emotion and trigger classification.

    This architecture uses a single fully-connected layer and two linear layers
    for two separate tasks: emotion classification and trigger detection.

    Attributes:
        model (transformers.BertModel): Pretrained BERT model for feature extraction.
        fc (nn.Linear): A fully-connected layer.
        emotion_classifier (nn.Linear): Output layer for emotion classification.
        trigger_classifier (nn.Linear): Output layer for trigger classification.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        train_bert: bool = True,
        hidden_size: int = 256,
    ) -> None:
        """
        Initialize the reference model.

        Args:
            num_classes (int): Number of emotion classes.
            model_name (str): Name of the pretrained BERT model. Set to "bert-base-cased"
                for MELD and "bert-base-multilingual-cased" for MaSaC.
            train_bert (bool): Whether to fine-tune the BERT model. Defaults to True.
            hidden_size (int): Number of neurons in a hidden layer. Defaults to 256.
        """
        super(EDiReFReferenceModel, self).__init__()

        self.model = BertModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = train_bert  # True to fine-tune model

        self.fc = nn.Linear(self.model.config.hidden_size, hidden_size)

        self.emotion_classifier = nn.Linear(hidden_size, num_classes)
        self.trigger_classifier = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - emotion_logits: Tensor of shape (batch_size, seq_len, num_classes).
                - trigger_logits: Tensor of shape (batch_size, seq_len).
        """
        model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = model_outputs.last_hidden_state
        pooled_embeddings = self.dropout(embeddings)

        fc_output = torch.relu(self.fc(pooled_embeddings))

        emotion_logits = self.emotion_classifier(fc_output)
        trigger_logits = self.trigger_classifier(fc_output).squeeze(-1)

        return emotion_logits, trigger_logits


# --- Experiment Execution ---


def _run_single_experiment(
    experiment_id: int,
    params: Dict[str, Any],
    dataset: str,
    weight_triggers: bool,
    train_bert: bool,
    device: torch.device,
) -> None:
    """
    Run a single experiment with specified configuration parameters.
    """
    # Unpack params
    learning_rate = params["LEARNING_RATE"]
    num_epochs = params["NUM_EPOCHS"]
    max_length = params["MAX_LENGTH"]
    batch_size = params["BATCH_SIZE"]

    print(f"Starting experiment number {experiment_id}...")

    # --- Data Loading & Preprocessing ---

    if dataset == "MELD":
        model_name = "bert-base-cased"
    elif dataset == "MaSaC":
        model_name = "bert-base-multilingual-cased"
    else:
        raise ValueError(f"{dataset} dataset is not supported.")

    train_conversations, train_emotions, train_triggers = get_data(dataset, "train")
    val_conversations, val_emotions, val_triggers = get_data(dataset, "val")

    conversations = train_conversations + val_conversations
    emotions = train_emotions + val_emotions
    triggers = train_triggers + val_triggers

    flattened_emotions = [sent for conv in emotions for sent in conv]
    unique_emotions = sorted(set(flattened_emotions))

    labels_to_ids = {k: v for v, k in enumerate(unique_emotions)}
    ids_to_labels = {v: k for v, k in enumerate(unique_emotions)}
    emotions = [[labels_to_ids[emotion] for emotion in conv] for conv in emotions]

    X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test = (
        train_val_test_split(
            conversations,
            emotions,
            triggers,
            test_size=0.15,
            val_size=0.15,
            random_state=2024,
        )
    )

    # --- Tokenization & Padding ---

    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_loader = create_dataloader(
        X_train,
        y1_train,
        y2_train,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True,
    )
    val_loader = create_dataloader(
        X_val,
        y1_val,
        y2_val,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )
    test_loader = create_dataloader(
        X_test,
        y1_test,
        y2_test,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )

    # --- Model Definition ---

    # Instantiate the reference model
    reference_model = EDiReFReferenceModel(
        num_classes=len(labels_to_ids), model_name=model_name, train_bert=train_bert
    )

    # Initialize optimizer
    optimizer = AdamW(reference_model.parameters(), lr=learning_rate)

    # Compute positive class weight for trigger classification (if applicable)
    pos_weight = None
    if weight_triggers and dataset == "MaSaC":
        flattened_triggers = [x for sequence in y2_train for x in sequence]
        # Down-weighting negative samples slightly to fix class imbalance
        num_negative_samples = len([x for x in flattened_triggers if x == 0]) * 0.8
        num_positive_samples = len(flattened_triggers) - num_negative_samples
        pos_weight_value = num_negative_samples / num_positive_samples
        pos_weight = torch.tensor([pos_weight_value], device=device)

    # Define loss functions
    emotion_loss_fn = nn.CrossEntropyLoss()
    trigger_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Training ---

    reference_model.to(device)

    logs = train_and_validate(
        model=reference_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        emotion_loss_fn=emotion_loss_fn,
        trigger_loss_fn=trigger_loss_fn,
        num_epochs=num_epochs,
        verbose=True,
    )

    # --- Evaluation ---

    metrics, emotion_cm, trigger_cm = get_metrics(
        model=reference_model,
        data_loader=test_loader,
        device=device,
        labels_to_ids=labels_to_ids,
        ids_to_labels=ids_to_labels,
    )

    # --- Save Experiment ---

    # # Optional: Save the trained model
    # # --------------------------------
    # # Uncomment this section to save the trained model to disk.

    # root = get_project_root()
    # model_dir = root / "trained_models" / dataset
    # model_dir.mkdir(parents=True, exist_ok=True)

    # file_name = (
    #     f"reference_model_{train_bert}_train_bert_"
    #     f"{learning_rate}_lr_{num_epochs}_epochs.pth"
    # )
    # model_path = model_dir / file_name

    # torch.save(reference_model.state_dict(), model_path)

    # Save logging, metrics and confusion matrices
    save_reference_experiment(
        weight_triggers=weight_triggers,
        dataset=dataset,
        train_bert=train_bert,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_length=max_length,
        batch_size=batch_size,
        logs=logs,
        metrics=metrics,
        emotion_cm=emotion_cm,
        trigger_cm=trigger_cm,
        labels_to_ids=labels_to_ids,
    )

    print(f"Finished experiment number {experiment_id}.\n")


def run_experiments(
    dataset: str,
    config_path: str,
    device: torch.device,
    weight_triggers: bool,
    train_bert: bool,
) -> None:
    """
    Runs experiments utilizing the reference model.

    Runs a series of experiments using parameters specified in a YAML configuration
    file. Experiments are done on a specified dataset and device.

    Args:
        dataset (str): The name of the dataset to use ("MELD", or "MaSaC").
        config_path (str): Path to the experiment configuration YAML file.
        device (torch.device): torch.device to use. Recommended "cuda" if available.
        weight_triggers (bool): Whether to apply weight to trigger classification.
        train_bert (bool): Whether to fine-tune the pretrained BERT model.
    """
    # Get experiment configurations
    with open(config_path, "r") as file:
        experiments = yaml.safe_load(file)

    # Run experiments
    for experiment_id, params in experiments.items():
        required_keys = {"LEARNING_RATE", "NUM_EPOCHS", "MAX_LENGTH", "BATCH_SIZE"}
        missing_keys = required_keys - set(params)
        assert required_keys.issubset(params), f"Missing keys in config: {missing_keys}"

        _run_single_experiment(
            experiment_id=experiment_id,
            params=params,
            dataset=dataset,
            weight_triggers=weight_triggers,
            train_bert=train_bert,
            device=device,
        )


if __name__ == "__main__":
    args = parse_arguments()
    dataset = args.dataset
    config_path = args.config_path
    weight_triggers = args.weight_triggers
    train_bert = args.train_bert

    set_seed()
    device = torch.device("cuda" if cuda.is_available() else "cpu")

    run_experiments(dataset, config_path, device, weight_triggers, train_bert)
