import argparse
import glob
import json
import os
import re
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class FileReader:

    def __init__(
        self,
    ):
        pass
        # self.tokenizer = tokenizer
        # self.label_encoder = label_encoder

    def _read_sequences_from_file(self, file_path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            sequence = [int(num) for num in file.read().strip().split()]
        return sequence

    def _read_sequences_from_folder(self, folder_path):
        sequences = []
        for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
            if not os.path.basename(file_path).startswith("._"):
                sequences.append(self._read_sequences_from_file(file_path))
        return sequences

    def _read_sequences_from_folder_with_subfolders(self, folder_path):
        sequences = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    sequences.append(self._read_sequences_from_file(file_path))

        return sequences

    def read_all_sequences(self, dataset_folder, dataset_name):

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
                data = self._read_sequences_from_folder_with_subfolders(folder_path)
                sequences.extend(data)
                labels.extend(["malware"] * len(data))

            else:
                data = self._read_sequences_from_folder(folder_path)
                sequences.extend(data)
                labels.extend(["normal"] * len(data))
        return sequences, labels

    def encode_sequences(self, sequences, vocabs):
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
            encoded_sequence = [
                vocabs.get(str(element), unk_index) for element in sequence
            ]
            encoded_data.append(encoded_sequence)

        return encoded_data

    def load_and_print_dataset(self, file_path, print_data: bool = False):
        with open(file_path, "r") as f:
            loaded_data = json.load(f)

        sequences = [item["sequence_data"] for item in loaded_data]
        labels = [item["label"] for item in loaded_data]

        if print_data:
            for idx, (sequence, label) in enumerate(zip(sequences, labels)):
                print(f"Sequence: {sequence}\nLabel: {label}\n")

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
            "syscall_" + vocab_path
            if dataset_name == "ADFA"
            else "apicall_" + vocab_path
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
            {"sequence_data": seq, "label": label}
            for seq, label in zip(call_data, labels)
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

        print(f"Saved {len(train_data)} training and {len(test_data)} testing samples.")

    def load_json_data(self, file_path):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        with open(file_path, "r") as f:
            self.data = json.load(f)

            sequences = [item["sequence_data"] for item in self.data]
            labels = []

            # assert len(sequences) == len(
            #     labels
            # ), "There is sanity issue with your data."

            for idx, item in enumerate(self.data):
                label = item["label"]
                # print(idx, "label = ", label)
                assert len(label) > 0, f"Empty label encountered for a sequence = {idx}"
                labels.append(label)

        return sequences, labels

    def encoded_data(self, tokenizer=None, label_encoder=None):
        return [
            tokenizer(element) for sequence in self.sequences for element in sequence
        ]

        self.labels  # labels are unencoded

        # return encode_sequences(self.sequences, data_tokenizer), self.labels

    def get_decoded_data(self):
        return self.decode_data(self.sequences)

    def decode_data(self, sequences, data_tokenizer=None):
        """The dataset are in integers, decode them back to system calls"""
        decoded_data = []
        for sequence in sequences:
            decoded_sequence = [self.data.get(str(element), -1) for element in sequence]
            decoded_data.append(decoded_sequence)
        return decoded_data

    @staticmethod
    def extract_syscalls(filename):
        """
        Extracts system call names and numbers from a file.

        Args:
            filename (str): The name of the file to extract from.

        Returns:
            dict: A dictionary where keys are syscall names and values are syscall numbers.
        """

        syscalls_table = {}
        with open(filename, "r") as f:
            for line in f:
                # Extract syscall number and name using regex
                match = re.search(r"#define\s+__NR_(\w+)\s+(\d+)", line)
                if match:
                    syscall_name = match.group(1)
                    syscall_number = int(match.group(2))
                    syscalls_table[syscall_name] = syscall_number
        return syscalls_table

    # 25, 43 - 46, 62, 71, 79 -80, 222 - 223, 245 - 259, 265 - 1023, 1038 - 1039, ...

    # Example: Accessing a specific syscall number
    # print(f"\nThe syscall number for 'read' is: {syscall_dict['read']}")


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

    reader = FileReader()

    if args.dataset_name == "ADFA":  # handle ADFA-LD dataset here
        sequences, labels = reader.read_all_sequences(
            args.dataset_folder, dataset_name=args.dataset_name
        )

        # save train and test data
        reader.preprocess_data(
            dataset_name=args.dataset_name, call_data=sequences, labels=labels
        )

    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")

    # map system calls

    vocab_path = os.path.join(
        os.getcwd(), args.dataset_folder, args.dataset_name, "ADFA-LD+Syscall+List.txt"
    )
    # reader = LoadJsonFromFile(vocab_path).encoded_data()

    syscalls = reader.extract_syscalls(vocab_path)

    print(sorted(syscalls.items(), key=lambda x: x[1]))

    print(len(syscalls))
