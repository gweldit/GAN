import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import RichProgressBar
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from custom_dataset import CustomDataset, Tokenizer, custom_collate_fn


class MalwareClassifier(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_classes: int = 10,
        num_filters: int = 128,
        kernel_size: int = 5,
        hidden_dim: int = 256,
        lr: float = 0.001,
        max_steps: int = 100,
    ):
        super().__init__()
        # self.save_hyperparameters()

        # Layers
        self.lr = lr
        self.max_steps = max_steps

        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)

        self.conv1 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
        )

        self.fc1 = nn.Linear(num_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        """
        x = self.embedding(x)  # (B, L, E)
        x = x.permute(0, 2, 1)  # (B, E, L)
        x = F.relu(self.conv1(x))  # (B, F, L)
        x = F.relu(self.conv2(x))  # (B, F, L)
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, F)
        x = F.relu(self.fc1(x))  # (B, H)
        x = self.fc2(x)  # (B, C)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print(self.device)
        x = x.long()
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        f1 = f1_score(
            y.cpu().numpy(), preds.cpu().numpy(), pos_label=0, average="binary"
        )
        # self.log("val_loss", loss, prog_bar=True)
        # self.log("val_f1", f1, prog_bar=True)
        self.log_dict(
            {"train_loss": loss, "train_f1": f1},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        # self.log("train_loss", loss, on_step=True, prog_bar=True, on_epoch=False)

        # print(f" Step {self.global_step} - Loss: {loss.item():.4f}")

        return loss

    def validation_step(self, batch, batch_idx):
        # print(batch[0])
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        f1 = f1_score(
            y.cpu().numpy(), preds.cpu().numpy(), pos_label=0, average="binary"
        )
        # self.log("val_loss", loss, prog_bar=True)
        # self.log("val_f1", f1, prog_bar=True)
        self.log_dict(
            {"val_loss": loss, "val_f1": f1},
            on_epoch=False,
            on_step=True,
            prog_bar=True,
        )

        return loss, f1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.max_steps,
                pct_start=0.3,
                anneal_strategy="cos",
                final_div_factor=100,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def test_step(self, batch, batch_idx):
        x, y = batch
        return self.validation_step(batch, batch_idx)


if __name__ == "__main__":
    # load training and testing datasets
    tokenizer = Tokenizer()
    label_encoder = LabelEncoder()

    train_dataset = CustomDataset(
        tokenizer, label_encoder, training=True, tokenize_data=True
    )  # ADFA data encoded from syscall numbers to token ids

    test_dataset = CustomDataset(
        train_dataset.tokenizer,
        train_dataset.label_encoder,
        training=False,
        tokenize_data=True,
    )  # ADFA data encoded from syscall numbers to token ids

    batch_size = 256
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True
    )
    val_loader = DataLoader(
        test_dataset, collate_fn=custom_collate_fn, batch_size=batch_size
    )

    # initialize the model
    # print(train_dataset[10])

    classifier_max_steps = 250

    # print(train_dataset.tokenizer.n_tokens())
    model_1 = MalwareClassifier(
        vocab_size=train_dataset.tokenizer.n_tokens(),
        embed_dim=128,
        num_classes=len(train_dataset.label_encoder.classes_),
        lr=1e-3,
        max_steps=classifier_max_steps,
    )

    trainer = pl.Trainer(
        # max_epochs=-1,
        max_steps=classifier_max_steps,
        val_check_interval=10,  # validate every 10 steps
        log_every_n_steps=10,
        accelerator="auto",
        devices="auto",
        callbacks=[RichProgressBar()],  # show step progress
        enable_progress_bar=True,
    )

    trainer.fit(model_1, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Your Next steps:
    # Step 0: Read ADFA -LD dataset: use the file_reader.py file
    # 1)  We have train our GAN on the ADFA-LD dataset

    # 2) Once we trained our GAN, we have generate fake samples ( 500 samples of sequences)

    # 3 ) Evaluate your Malware Classifier on ADFA-LD before balancing dataset

    # 4) Then, load generator model and generate fake samples of malware. Then, we evaluate Malware Classifier after balancing ( add fake samples of malware with our original training datasets from the ADFA dataset)
    # model2 = MalwareClassifier(...)
