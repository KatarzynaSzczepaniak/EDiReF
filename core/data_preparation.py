from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer

from sklearn.model_selection import train_test_split

from .utilities import get_project_root


def get_data(
    dataset_name: str, stage: str
) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Load and clean EDiReF JSON data for a given dataset and stage.

    The function loads a JSON file, fills missing values in the "triggers" column,
    filters out invalid rows, and extracts "utterances", "emotions", and "triggers".

    Args:
        dataset_name (str): Name of the dataset ("MELD", or "MaSaC").
        stage (str): Subset name corresponding to the file suffix ("train" or "val").

    Returns:
        Tuple[List[List[str]], List[List[str]], List[List[float]]]: A tuple containing:
            - conversations: List of conversations (utterances).
            - emotions: List of corresponding emotion labels.
            - triggers: List of corresponding trigger values.
    """

    def to_float(x):
        try:
            return float(x)
        except ValueError:
            print(f"Could not convert {x} to float. Defaulting to 1.0.")
            return 1.0

    root = get_project_root()
    data_path = (
        root / "data" / f"EDiReF_{stage}_data" / f"{dataset_name}_{stage}_efr.json"
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_json(data_path)
    df["triggers"] = df["triggers"].apply(
        lambda lst: [np.nan if x is None else x for x in lst]
    )
    df = df[df["triggers"].apply(lambda lst: not any(pd.isna(x) for x in lst))]
    df["triggers"] = df["triggers"].apply(lambda lst: [to_float(x) for x in lst])

    conversations = list(df["utterances"])
    emotions = list(df["emotions"])
    triggers = list(df["triggers"])

    return conversations, emotions, triggers


def train_val_test_split(
    X: List[List[str]],
    y1: List[List[int]],
    y2: List[List[float]],
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = None,
) -> Tuple[
    List[List[str]],
    List[List[str]],
    List[List[str]],
    List[List[int]],
    List[List[int]],
    List[List[int]],
    List[List[float]],
    List[List[float]],
    List[List[float]],
]:
    """
    Split data into train, validation, and test sets with consistent label alignment.

    This function performs a two-step split:
    - First, it splits the dataset into training+validation and test sets.
    - Then it splits the training+validation set again to separate out a validation set.
    This ensures y1 and y2 (labels) stay aligned with X throughout the process.

    Args:
        X (List[List[str]]): The main input data (e.g., tokenized conversations).
        y1 (List[List[int]]): The first set of target labels (e.g., emotions).
        y2 (List[List[float]]): The second set of target labels (e.g., triggers).
        val_size (float, optional): Proportion of data to use for validation.
            Defaults to 0.2.
        test_size (float, optional): Proportion of data to use for test set.
            Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility.
            Defaults to None.

    Returns:
        Tuple[
            List[List[str]], List[List[str]], List[List[str]],
            List[List[int]], List[List[int]], List[List[int]],
            List[List[float]], List[List[float]], List[List[float]]
        ]: A tuple containing:
            - X_train, X_val, X_test
            - y1_train, y1_val, y1_test
            - y2_train, y2_val, y2_test
    """

    def validate_lengths(X, y1, y2):
        assert (
            len(X) == len(y1) == len(y2)
        ), f"Data misalignment detected: X={len(X)}, y1={len(y1)}, y2={len(y2)}"

    X_train_val, X_test, y1_train_val, y1_test, y2_train_val, y2_test = (
        train_test_split(X, y1, y2, test_size=test_size, random_state=random_state)
    )

    val_relative_size = val_size / (1 - test_size)

    X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
        X_train_val,
        y1_train_val,
        y2_train_val,
        test_size=val_relative_size,
        random_state=random_state,
    )

    validate_lengths(X_train, y1_train, y2_train)
    validate_lengths(X_val, y1_val, y2_val)
    validate_lengths(X_test, y1_test, y2_test)

    return (
        X_train,
        X_val,
        X_test,
        y1_train,
        y1_val,
        y1_test,
        y2_train,
        y2_val,
        y2_test,
    )


def tokenize_conversation(
    conversations: List[List[str]], tokenizer: BertTokenizer, max_length: int = 128
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Tokenize and pad a list of conversations using pretrained BertTokenizer.

    Each conversation is flattened into a single string using [SEP] as a delimiter.
    Tokenization is performed with padding and truncation to a specified max length.

    Args:
        conversations (List[List[str]]): A list of conversations, where each
            conversation is a list of utterances (strings).
        tokenizer (BertTokenizer): A HuggingFace tokenizer used to tokenize
            the input conversations.
        max_length (int, optional): Maximum length (in tokens) for padding/truncation.
            Defaults to 128.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: A tuple containing:
            - input_ids: Token ID tensors for each conversation.
            - attention_masks: Attention mask tensors for each conversation.
    """
    input_ids = []
    attention_masks = []

    for conversation in conversations:
        dialogue = f" {tokenizer.sep_token} ".join(conversation)
        encoded = tokenizer(
            dialogue,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids.append(encoded["input_ids"].squeeze(0))
        attention_masks.append(encoded["attention_mask"].squeeze(0))

    return input_ids, attention_masks


def pad_labels(
    labels: List[List[Union[int, float]]], max_length: int = 128
) -> List[torch.Tensor]:
    """
    Pads each list of labels to a specified max length.

    Each list of labels is converted to a float tensor and padded with -1.0
    to a specified max length. Useful for masking loss during training on
    token-level tasks.

    Args:
        labels (List[List[Union[int, float]]]): A list of label sequences
            (e.g., emotions or triggers).
        max_length (int, optional): Maximum length to pad/truncate each
            sequence to. Defaults to 128.

    Returns:
        List[torch.Tensor]: A list of padded 1D tensors, one per input sequence.
    """
    padded_labels = []

    for label_set in labels:
        label_tensor = torch.tensor(label_set[:max_length], dtype=torch.float)
        pad_len = max_length - len(label_tensor)
        if pad_len > 0:
            padding_tensor = torch.full((pad_len,), -1.0)
            label_tensor = torch.cat([label_tensor, padding_tensor])
        padded_labels.append(label_tensor)

    return padded_labels


class ConversationDataset(Dataset):
    """
    A PyTorch-compatible dataset for emotion and trigger classification tasks.

    Stores tokenized conversations along with attention masks, and emotion and
    trigger labels for each utterance in the input.
    """

    def __init__(
        self,
        input_ids: List[torch.Tensor],
        attention_masks: List[torch.Tensor],
        emotion_labels: List[torch.Tensor],
        trigger_labels: List[torch.Tensor],
    ) -> None:
        """
        Initialize the dataset with tokenized inputs and their corresponding labels.

        Args:
            input_ids (List[torch.Tensor]): Token IDs for each conversation.
            attention_masks (List[torch.Tensor]): Attention masks for each conversation.
            emotion_labels (List[torch.Tensor]): Label tensors for emotion classification.
            trigger_labels (List[torch.Tensor]): Label tensors for trigger classification.
        """
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.emotion_labels = emotion_labels
        self.trigger_labels = trigger_labels

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: The total number of data points.
        """
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - "input_ids"
                - "attention_mask"
                - "emotion_labels"
                - "trigger_labels"
        """
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "emotion_labels": self.emotion_labels[idx],
            "trigger_labels": self.trigger_labels[idx],
        }


def create_dataloader(
    conversations: List[List[str]],
    emotions: List[List[int]],
    triggers: List[List[float]],
    tokenizer: BertTokenizer,
    batch_size: int,
    max_length: int = 128,
    shuffle: bool = False,
) -> DataLoader:
    """
    Create a DataLoader from conversations and corresponding labels.

    Conversations are first tokenized using the `tokenize_conversation()` function,
    and the emotion and trigger labels are padded using `pad_labels()`. These are then
    wrapped in a `ConversationDataset` and returned as a PyTorch DataLoader.

    Args:
        conversations (List[List[str]]): A list of conversations, where each conversation
            is a list of utterances (strings).
        emotions (List[List[int]]): A list of emotion labels.
        triggers (List[List[float]]): A list of trigger labels.
        tokenizer (BertTokenizer): A HuggingFace tokenizer used to tokenize
            the input conversations.
        batch_size (int): Number of samples per batch.
        max_length (int, optional): Maximum length to pad/truncate each sequence to.
            Defaults to 128.
        shuffle (bool, optional): Whether to shuffle the data at every epoch.
            Defaults to False.

    Returns:
        DataLoader: An iterable over the constructed ConversationDataset.
    """
    input_ids, attention_masks = tokenize_conversation(
        conversations, tokenizer, max_length=max_length
    )
    emotions_labels = pad_labels(emotions, max_length=max_length)
    triggers_labels = pad_labels(triggers, max_length=max_length)

    dataset = ConversationDataset(
        input_ids, attention_masks, emotions_labels, triggers_labels
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
