import os
import random

import torch
from sklearn.preprocessing import LabelEncoder

from file_reader import FileReader


class Tokenizer:
    def __init__(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>"]
        self.special_tokens = special_tokens
        self.token_to_id = {tok: i for i, tok in enumerate(special_tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        self.fitted = False

    def fit(self, sequences):
        """
        Fit tokenizer on a list of sequences containing syscall numbers (integers).
        """
        idx = len(self.token_to_id)
        for seq in sequences:
            for syscall_num in seq:
                if syscall_num not in self.token_to_id:
                    self.token_to_id[syscall_num] = idx
                    self.id_to_token[idx] = syscall_num
                    idx += 1
        self.fitted = True

    def encode(self, sequence):
        """
        Encodes a list of syscall numbers (e.g., [1024, 1050]) into token IDs.
        """
        if not self.fitted:
            raise ValueError("Tokenizer is not fitted.")
        tokens = ["<START>"] + sequence + ["<END>"]
        return [self.token_to_id.get(tok, self.token_to_id["<UNK>"]) for tok in tokens]

    def decode(self, token_ids):
        """
        Decode token IDs back to syscall numbers (removes special tokens).
        """
        tokens = [self.id_to_token.get(idx, "<UNK>") for idx in token_ids]
        tokens = [t for t in tokens if t not in ("<PAD>", "<START>", "<END>")]
        return tokens

    def get_vocab(self):
        return list(self.token_to_id.keys())

    def n_tokens(self):
        return len(self.token_to_id)

    def pad_token_id(self):
        return self.token_to_id["<PAD>"]

    def start_token_id(self):
        return self.token_to_id["<START>"]

    def end_token_id(self):
        return self.token_to_id["<END>"]

    def unk_token_id(self):
        return self.token_to_id["<UNK>"]


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        label_encoder: LabelEncoder,
        training: bool,
        tokenize_data: bool = True,
    ):
        self.training = training  # is it training or testing sets
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.reader = FileReader()

        file_prefix = "train" if training else "test"
        filename = f"{file_prefix}_dataset.json"
        file_path = os.path.join("dataset", "ADFA", filename)

        raw_sequences, raw_labels = self.reader.load_json_data(
            file_path
        )  # raw data, labels

        # fit the raw data to tokenizer
        if self.training:
            self.tokenizer.fit(raw_sequences)
            self.label_encoder.fit(raw_labels)

        self.labels = self.label_encoder.transform(raw_labels)

        if tokenize_data:
            self.sequences = [self.tokenizer.encode(seq) for seq in raw_sequences]
        else:
            self.sequences = [[int(t) for t in seq] for seq in raw_sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), self.labels[idx]


def custom_collate_fn(batch, seq_len=150, pad_value=0):
    sequences = []
    labels = []

    for seq, label in batch:
        # print(label)
        seq_len_actual = len(seq)

        # Truncate or pad
        if seq_len_actual > seq_len:
            start_idx = random.randint(0, seq_len_actual - seq_len)
            seq = seq[start_idx : start_idx + seq_len]
        elif seq_len_actual < seq_len:
            padding = torch.full(
                (seq_len - seq_len_actual,),
                pad_value,
                dtype=seq.dtype,
                device=seq.device,
            )
            seq = torch.cat([seq, padding], dim=0)

        sequences.append(seq)
        labels.append(label.item() if isinstance(label, torch.Tensor) else label)

    return torch.stack(sequences), torch.tensor(labels)


if __name__ == "__main__":
    tokenizer = Tokenizer()
    label_encoder = LabelEncoder()
    train_data = CustomDataset(
        tokenizer, label_encoder, training=True, tokenize_data=True
    )  # ADFA data encoded from syscall numbers to token ids

    train_data = CustomDataset(
        train_data.tokenizer,
        train_data.label_encoder,
        training=False,
        tokenize_data=True,
    )  # AD
    print(train_data[0])
