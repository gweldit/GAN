import argparse
import glob
import json
import os
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def read_sequences_from_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        sequence = [int(num) for num in file.read().strip().split()]
    return sequence


def read_sequences_from_folder(folder_path):
    sequences = []
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        if not os.path.basename(file_path).startswith("._"):
            sequences.append(read_sequences_from_file(file_path))
    return sequences


def read_sequences_from_folder_with_subfolders(folder_path):
    sequences = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                sequences.append(read_sequences_from_file(file_path))

    return sequences


def read_all_sequences(dataset_folder, dataset_name):

    dataset_folder = os.path.join(os.getcwd(), dataset_folder, dataset_name)

    print("dataset folder path = ", dataset_folder)

    path_sub_folders = [
        os.path.join(dataset_folder, sub_folder)
        for sub_folder in os.listdir(dataset_folder)
        if os.path.isdir(os.path.join(dataset_folder, sub_folder))
    ]

    print("sub folders :", path_sub_folders)
    sequences = []
    labels = []
    for folder_path in path_sub_folders:
        if "Attack" in folder_path:
            data = read_sequences_from_folder_with_subfolders(folder_path)
            sequences.extend(data)
            labels.extend(["malware"] * len(data))

        else:
            data = read_sequences_from_folder(folder_path)
            sequences.extend(data)
            labels.extend(["normal"] * len(data))
    return sequences, labels


def save_sequence_data(sequences, labels, output_file_path):
    # Serialize all sequence data and vocabulary to a single file

    # check if folder exists, else create new folder
    data_dir = os.path.join(os.getcwd(), "data")
    print("data dir path = ", data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Create directory: {data_dir}")

    with open(output_file_path, "w") as f:
        data_to_save = {
            "sequence_data": sequences,
            # 'vocab_size': len(encoder.syscall_vocabs),
            # 'vocab': encoder.syscall_vocabs,
            "labels": labels,
            # 'vocab': syscall_vocab,
        }

        # check if file alreay exists in a folder
        if os.path.exists(output_file_path):
            print(f"File {output_file_path} already exists. Overwriting...")

        else:
            print(f"File {output_file_path} does not exist. Creating new file")
        with open(output_file_path, "w") as f:
            json.dump(data_to_save, f)

    return output_file_path


def encode_sequences(sequences, vocabs):
    """
    Encodes a list of sequences using a given vocabulary.

    Args:
    sequences (list of list of integers): The sequences to encode.
    vocabs (dict): A dictionary mapping each element to an index.

    Returns:
    list of list of int: The encoded sequences.
    """
    encoded_data = []
    # Define an index for unknown elements, default to -1 if not provided
    unk_index = vocabs.get("UNK", -1)

    for sequence in sequences:
        encoded_sequence = [vocabs.get(str(element), unk_index) for element in sequence]
        encoded_data.append(encoded_sequence)

    return encoded_data


def load_and_print_dataset(file_path, print_data: bool = False):
    with open(file_path, "r") as f:
        loaded_data = json.load(f)

    sequences = loaded_data["sequence_data"]
    labels = loaded_data["labels"]
    # vocab_size = loaded_data['vocab_size']
    # syscall_vocab = loaded_data['vocab']
    # max_seq_len = loaded_data['max_seq_len']
    if print_data:
        for idx, (sequence, label) in enumerate(zip(sequences, labels)):
            print(f"Sequence: {sequence}\nLabel: {label}\n")
        # print(f"Vocabulary Size: {vocab_size}")
        # print('Vocab: ', syscall_vocab)
        # print(f"Max Length: {max_seq_len}")

    return sequences, labels


def read_malapi_2019(
    api_data_path: str, labels_path: str
) -> Tuple[List[List[str]], pd.DataFrame]:
    with open(api_data_path, "r") as file:
        lines = file.readlines()
        data = [line.strip().split(" ") for line in lines]

    with open(labels_path, "r") as f:
        lines = f.readlines()
        labels = [line.strip() for line in lines if line.strip()]

    return data, labels


def preprocess_data(
    dataset_name: str,
    call_data: List[List[str]],
    labels: List,
    vocab_path="vocabs.json",
    train_path="train_dataset.json",
    test_path="test_dataset.json",
    test_size=0.2,
    random_seed=42,
):

    # create full path to save vocabs, train and test data
    train_path = f"dataset/{dataset_name}/{train_path}"
    test_path = f"dataset/{dataset_name}/{test_path}"

    vocab_path = (
        "syscall_" + vocab_path if dataset_name == "adfa" else "apicall_" + vocab_path
    )

    vocab_path = f"dataset/{dataset_name}/{vocab_path}"
    # Create vocabulary
    all_calls = sorted(set(call for sequence in call_data for call in sequence))
    call_vocab = {call: idx for idx, call in enumerate(all_calls)}

    # Save vocabulary
    with open(vocab_path, "w") as f:
        json.dump(call_vocab, f)

    # Merge API calls and labels
    combined = [
        {"sequence_data": seq, "label": label} for seq, label in zip(call_data, labels)
    ]

    # Split dataset
    train_data, test_data = train_test_split(
        combined,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )

    # Save datasets
    with open(train_path, "w") as f:
        json.dump(train_data, f)

    with open(test_path, "w") as f:
        json.dump(test_data, f)

    print(
        f"Saved {len(train_data)} training samples and {len(test_data)} testing samples."
    )


def fetch_graphs(encoder, sequences, labels):
    graphs = []
    for sequence, label in zip(sequences, labels):
        graphs.append(encoder.sequence_to_graph(sequence, label))
    return graphs


if __name__ == "__main__":
    # Paths to folders : two normal data folder and third attack folder containing subfolders

    # data_dir = os.path.join("dataset", "ADFA")
    parser = argparse.ArgumentParser(
        description="Read ADFA dataset from given directory"
    )

    parser.add_argument(
        "--dataset_folder",
        type=str,
        help="The folder path to the dataset",
        default="dataset",
    )
    parser.add_argument("--dataset_name", type=str, default="ADFA", choices=["ADFA"])
    args = parser.parse_args()

    if args.dataset_name == "ADFA":  # handle ADFA-LD dataset here
        sequences, labels = read_all_sequences(
            args.dataset_folder, dataset_name=args.dataset_name
        )

        # save train and test data
        preprocess_data(
            dataset_name=args.dataset_name, call_data=sequences, labels=labels
        )

    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
