from collections import defaultdict
from collections.abc import Callable
from typing import Any, Dict, List, Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)


def remove_padding(
    logits: torch.Tensor, labels: torch.Tensor, task: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove masked padding from logits and labels for loss computation.

    This function flattens both the model outputs and target labels, then removes
    entries where labels are set to -1 (which represent padding tokens).

    Args:
        logits (torch.Tensor): Model outputs of shape (batch_size, seq_len, num_classes)
            for "emotion", or (batch_size, seq_len) for "trigger".
        labels (torch.Tensor): Ground-truth labels of shape (batch_size, seq_len).
        task (str): Classification task type, either "emotion" or "trigger".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing a pair of 1d tensors:
            - logits: Flattened logits excluding padding.
            - labels: Flattened ground-truth labels excluding padding.
    """
    assert task in {"emotion", "trigger"}, "task must be 'emotion' or 'trigger'"

    valid_positions = labels != -1

    logits_flat = (
        logits.view(-1, logits.size(-1)) if task == "emotion" else logits.view(-1)
    )
    labels_flat = labels.view(-1)

    logits = logits_flat[valid_positions.view(-1)]
    labels = labels_flat[valid_positions.view(-1)]

    return logits, labels


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    emotion_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    trigger_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    verbose: bool = True,
) -> Tuple[float, float, List[str]]:
    """
    Evaluates the Mixture-of-Experts model on a validation dataset.

    Computes total loss, emotion classification accuracy, and trigger classification
    accuracy. Logs validation loss every 100 steps.

    Args:
        model (nn.Module): The trained model to evaluate.
        val_loader (DataLoader): Dataloader for the validation set.
        device (torch.device): Represents the device on which a torch.Tensor is
            or will be allocated.
        emotion_loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            Loss function used for emotion classification (e.g., nn.CrossEntropyLoss).
        trigger_loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            Loss function used for trigger classification (e.g., nn.BCEWithLogitsLoss).
        verbose (bool): Controls the verbosity. Set False to hide messages.
            Defaults to True.

    Returns:
        Tuple[float, float, List[str]]: A tuple containing:
            - avg_val_loss: Mean validation loss across all batches.
            - avg_val_accuracy: Mean of emotion and trigger accuracy.
            - val_logs: Log messages recorded during validation.
    """
    model.eval()
    val_loss, nb_steps = 0.0, 0
    total_emotion_preds, correct_emotion_preds = 0, 0
    total_trigger_preds, correct_trigger_preds = 0, 0
    val_logs = []

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            # forward pass
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            trigger_labels = batch["trigger_labels"].to(device)

            emotion_logits, trigger_logits = model(input_ids, attention_mask)

            # remove padding
            emotion_logits, emotion_labels = remove_padding(
                emotion_logits, emotion_labels, "emotion"
            )
            trigger_logits, trigger_labels = remove_padding(
                trigger_logits, trigger_labels, "trigger"
            )

            # compute loss
            emotion_loss = emotion_loss_fn(emotion_logits, emotion_labels.long())
            trigger_loss = trigger_loss_fn(trigger_logits, trigger_labels)

            loss = emotion_loss + trigger_loss
            val_loss += loss.item()

            # compute accuracy
            emotion_preds = torch.argmax(emotion_logits, dim=-1)
            trigger_preds = (torch.sigmoid(trigger_logits).view(-1) > 0.5).long()

            correct_emotion_preds += torch.sum(emotion_preds == emotion_labels).item()
            correct_trigger_preds += torch.sum(trigger_preds == trigger_labels).item()

            total_emotion_preds += emotion_labels.numel()
            total_trigger_preds += trigger_labels.numel()

            nb_steps += 1

            # logging
            if verbose and idx % 100 == 0:
                loss_step = val_loss / nb_steps
                print(f"      Validation loss per 100 training steps: {loss_step:.4f}")
                val_logs.append(
                    f"      Validation loss per 100 training steps: {loss_step:.4f}\n"
                )

        avg_val_loss = val_loss / max(len(val_loader), 1)
        emotion_accuracy = correct_emotion_preds / max(total_emotion_preds, 1)
        trigger_accuracy = correct_trigger_preds / max(total_trigger_preds, 1)
        avg_val_accuracy = (emotion_accuracy + trigger_accuracy) / 2

    return avg_val_loss, avg_val_accuracy, val_logs


def train_and_validate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    emotion_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    trigger_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_epochs: int = 3,
    verbose: bool = True,
) -> List[str]:
    """
    Train and evaluate the Mixture-of-Experts model for emotion and trigger classification.

    For each epoch, computes training loss and accuracy for both tasks. Logs
    training loss every 100 steps and evaluates the model on the validation set
    at the end of each epoch.

    Args:
        model (nn.Module): The model to train and evaluate.
        train_loader (DataLoader): Dataloader for the train set.
        val_loader (DataLoader): Dataloader for the validation set.
        optimizer (torch.optim.Optimizer): PyTorch optimizer.
        device (torch.device): Represents the device on which a torch.Tensor is
            or will be allocated.
        emotion_loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            Loss function used for emotion classification (e.g., nn.CrossEntropyLoss).
        trigger_loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            Loss function used for trigger classification (e.g., nn.BCEWithLogitsLoss).
        num_epochs (int, optional): Number of training epochs. Defaults to 3.
        verbose (bool): Controls the verbosity. Set False to hide messages.
            Defaults to True.

    Returns:
        List[str]: A list of formatted strings logging training and validation
        progress for each epoch and key steps within.
    """
    train_logs = []

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        train_logs.append(f"Epoch [{epoch + 1}/{num_epochs}]\n")
        model.train()

        train_loss, nb_steps = 0.0, 0
        total_emotion_preds, correct_emotion_preds = 0, 0
        total_trigger_preds, correct_trigger_preds = 0, 0

        for idx, batch in enumerate(train_loader):
            # forward pass
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            trigger_labels = batch["trigger_labels"].to(device)

            emotion_logits, trigger_logits = model(input_ids, attention_mask)

            # remove padding
            emotion_logits, emotion_labels = remove_padding(
                emotion_logits, emotion_labels, "emotion"
            )
            trigger_logits, trigger_labels = remove_padding(
                trigger_logits, trigger_labels, "trigger"
            )

            # compute loss
            emotion_loss = emotion_loss_fn(emotion_logits, emotion_labels.long())
            trigger_loss = trigger_loss_fn(trigger_logits, trigger_labels)

            loss = emotion_loss + trigger_loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            # compute accuracy
            emotion_preds = torch.argmax(emotion_logits, dim=-1)
            trigger_preds = (torch.sigmoid(trigger_logits).view(-1) > 0.5).long()

            correct_emotion_preds += torch.sum(emotion_preds == emotion_labels).item()
            correct_trigger_preds += torch.sum(trigger_preds == trigger_labels).item()

            total_emotion_preds += emotion_labels.numel()
            total_trigger_preds += trigger_labels.numel()

            nb_steps += 1

            # logging
            if verbose and idx % 100 == 0:
                loss_step = train_loss / nb_steps
                print(f"      Training loss per 100 training steps: {loss_step:.4f}")
                train_logs.append(
                    f"      Training loss per 100 training steps: {loss_step:.4f}\n"
                )

        avg_train_loss = train_loss / max(len(train_loader), 1)
        emotion_accuracy = correct_emotion_preds / max(total_emotion_preds, 1)
        trigger_accuracy = correct_trigger_preds / max(total_trigger_preds, 1)
        avg_train_accuracy = (emotion_accuracy + trigger_accuracy) / 2

        val_loss, val_accuracy, val_logs = evaluate(
            model,
            val_loader,
            device=device,
            emotion_loss_fn=emotion_loss_fn,
            trigger_loss_fn=trigger_loss_fn,
            verbose=verbose,
        )
        train_logs.extend(val_logs)
        train_logs.append(
            f"   Training Loss: {avg_train_loss:.4f}, Training Accuracy: {avg_train_accuracy:.4f}\n"
        )
        train_logs.append(
            f"   Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n\n"
        )

        if verbose:
            print(
                f"   Training Loss: {avg_train_loss:.4f}, Training Accuracy: {avg_train_accuracy:.4f}"
            )
            print(
                f"   Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n"
            )

    return train_logs


def get_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    labels_to_ids: Dict[str, int],
    ids_to_labels: Dict[int, str],
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluate a trained Mixture-of-Experts model on a test set and compute metrics.

    Calculates accuracy, precision, recall, and F1 scores for both emotion and
    trigger classification tasks, as well as confusion matrices.

    Args:
        model (nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the test set.
        device (torch.device): The device to use for computation.
        labels_to_ids (Dict[str, int]): Dictionary mapping strings into integers.
        ids_to_labels (Dict[int, str]): Dictionary mapping integers to labels.

    Returns:
        Tuple[Dict[str, Any], np.ndarray, np.ndarray]: A tuple containing:
            - metrics (dict): Dictionary with aggregated performance metrics.
            - emotion_cm (np.ndarray): Confusion matrix for emotion classification.
            - trigger_cm (np.ndarray): Confusion matrix for trigger classification.
    """
    model.eval()

    # Initialize metrics
    emotion_accuracy, emotion_precision, emotion_recall, emotion_f1 = [], [], [], []
    trigger_accuracy, trigger_precision, trigger_recall, trigger_f1 = [], [], [], []

    emotion_cm = None
    trigger_cm = None

    unique_emotions_f1 = {k: [] for k in labels_to_ids.keys()}

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            trigger_labels = batch["trigger_labels"].to(device)

            emotion_logits, trigger_logits = model(input_ids, attention_mask)

            # Emotion classification
            emotion_logits, emotion_labels = remove_padding(
                emotion_logits, emotion_labels, "emotion"
            )

            emotion_preds = torch.argmax(emotion_logits, dim=-1)

            emotion_preds_np = emotion_preds.cpu().numpy()
            emotion_labels_np = emotion_labels.cpu().numpy()

            emotion_accuracy.append(accuracy_score(emotion_labels_np, emotion_preds_np))
            emotion_precision.append(
                precision_score(
                    emotion_labels_np,
                    emotion_preds_np,
                    average="weighted",
                    zero_division=0,
                )
            )
            emotion_recall.append(
                recall_score(
                    emotion_labels_np,
                    emotion_preds_np,
                    average="weighted",
                    zero_division=0,
                )
            )
            emotion_f1.append(
                f1_score(
                    emotion_labels_np,
                    emotion_preds_np,
                    average="weighted",
                    zero_division=0,
                )
            )

            for idx, score in enumerate(
                f1_score(
                    emotion_labels_np, emotion_preds_np, average=None, zero_division=0
                )
            ):
                unique_emotions_f1[ids_to_labels[idx]].append(score)

            emotion_cm_batch = confusion_matrix(
                emotion_labels_np,
                emotion_preds_np,
                labels=list(range(len(labels_to_ids))),
            )
            emotion_cm = (
                emotion_cm_batch
                if emotion_cm is None
                else emotion_cm + emotion_cm_batch
            )

            # Trigger classification
            trigger_logits, trigger_labels = remove_padding(
                trigger_logits, trigger_labels, "trigger"
            )

            trigger_preds = (torch.sigmoid(trigger_logits).view(-1) > 0.5).long()

            trigger_preds_np = trigger_preds.cpu().numpy()
            trigger_labels_np = trigger_labels.cpu().numpy()

            trigger_accuracy.append(accuracy_score(trigger_labels_np, trigger_preds_np))
            trigger_precision.append(
                precision_score(
                    trigger_labels_np,
                    trigger_preds_np,
                    average="weighted",
                    zero_division=0,
                )
            )
            trigger_recall.append(
                recall_score(
                    trigger_labels_np,
                    trigger_preds_np,
                    average="weighted",
                    zero_division=0,
                )
            )
            trigger_f1.append(
                f1_score(
                    trigger_labels_np,
                    trigger_preds_np,
                    average="weighted",
                    zero_division=0,
                )
            )

            trigger_cm_batch = confusion_matrix(
                trigger_labels_np, trigger_preds_np, labels=[0, 1]
            )
            trigger_cm = (
                trigger_cm_batch
                if trigger_cm is None
                else trigger_cm + trigger_cm_batch
            )

    # Aggregate metrics
    metrics = defaultdict(lambda: {})

    metrics["emotion_classification"] = {
        "accuracy": np.mean(emotion_accuracy),
        "precision": np.mean(emotion_precision),
        "recall": np.mean(emotion_recall),
        "f1": {"avg": np.mean(emotion_f1)},
    }

    for key in unique_emotions_f1:
        metrics["emotion_classification"]["f1"][key] = np.mean(unique_emotions_f1[key])

    metrics["trigger_classification"] = {
        "accuracy": np.mean(trigger_accuracy),
        "precision": np.mean(trigger_precision),
        "recall": np.mean(trigger_recall),
        "f1": np.mean(trigger_f1),
    }

    return metrics, emotion_cm, trigger_cm
