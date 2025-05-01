import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class GAN(pl.LightningModule):
    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int,
        seq_len: int,
        vocab_size: int,
        max_steps: int,
    ):
        super().__init__()

        self.noise_dim = noise_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.max_steps = (
            max_steps  # should be equal number of iterations to train your GAN model
        )

        # define a generator
        self.G = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),  # activation function to introduce non-linearity
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, seq_len * vocab_size),
            # nn.Tanh(),  # optional: normalize outputs between [-1, 1]
        )

        self.D = nn.Sequential(
            nn.Linear(seq_len * vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # two classes: real (label=1) or fake (0)
        )

        self.automatic_optimization = False  # we'll manually control both G and D steps

    def _sample_noise(self, batch_size):
        """Sample random noise for the generator."""
        z = torch.normal(
            mean=0,
            std=math.sqrt(1.0 / self.noise_dim),
            size=(batch_size, self.noise_dim),
            device=self.device,
        )
        return z

    def generator_training_step(self, batch):
        (real_sequences,) = batch  # get samples

        batch_size = real_sequences.size(0)

        # generate fake sequences
        noise = self._sample_noise(batch_size)
        fake_sequences = self.G(noise)

        # pass fake sequences through discriminator
        logits_fake = self.D(fake_sequences)

        # labels: generator tries to trick D into thinking fakes are real (label=1)
        labels_real = torch.ones(batch_size, dtype=torch.long, device=self.device)

        g_loss = nn.functional.cross_entropy(logits_fake, labels_real)
        self.log("g_loss", g_loss, prog_bar=True, on_step=True, on_epoch=True)

        return g_loss

    def discriminator_training_step(self, batch):
        (real_sequences,) = batch
        batch_size = real_sequences.size(0)

        # flatten real sequences (from [batch, seq_len, vocab_size] to [batch, seq_len * vocab_size])
        real_sequences = real_sequences.view(batch_size, -1).float()

        # generate fake sequences
        noise = self._sample_noise(batch_size)
        fake_sequences = self.G(noise).detach()  # detach so G is not updated here

        # discriminator logits
        logits_real = self.D(real_sequences)
        logits_fake = self.D(fake_sequences)

        # real labels = 1, fake labels = 0
        labels_real = torch.ones(batch_size, dtype=torch.long, device=self.device)
        labels_fake = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # compute losses
        loss_real = nn.functional.cross_entropy(logits_real, labels_real)
        loss_fake = nn.functional.cross_entropy(logits_fake, labels_fake)

        print("loss = ", loss_real.item(), torch.mean(logits_real).item())

        d_loss = 0.5 * (loss_real + loss_fake)
        self.log("d_loss", d_loss, prog_bar=True, on_step=True, on_epoch=True)

        return d_loss

    def training_step(self, batch, batch_idx):
        """Custom training loop handling both G and D steps."""
        (real_sequences,) = batch

        # Get optimizers and schedulers
        g_optimizer, d_optimizer = self.optimizers()
        g_scheduler, d_scheduler = self.lr_schedulers()

        # 1. Update Discriminator
        d_loss = self.discriminator_training_step(batch)
        self.manual_backward(d_loss)
        d_optimizer.step()
        d_optimizer.zero_grad()

        if d_scheduler is not None:
            d_scheduler.step()  # update learning rate of D

        # 2. Update Generator
        g_loss = self.generator_training_step(batch)
        self.manual_backward(g_loss)
        g_optimizer.step()
        g_optimizer.zero_grad()

        if g_scheduler is not None:
            g_scheduler.step()  # update learning rate of G

        # 4. Log losses
        self.log("g_loss", g_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("d_loss", d_loss, prog_bar=True, on_step=True, on_epoch=True)
        # log learning rates
        if g_scheduler is not None:
            self.log(
                "g_lr",
                g_scheduler.get_last_lr()[0],
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
        if d_scheduler is not None:
            self.log(
                "d_lr",
                d_scheduler.get_last_lr()[0],
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # Schedulers
        g_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            g_optimizer,
            max_lr=2e-3,  # peak learning rate (higher than base lr)
            # steps_per_epoch=100,  # number of batches per epoch
            # epochs=10,  # number of epochs
            total_steps=self.max_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            final_div_factor=10,
        )
        d_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            d_optimizer,
            max_lr=2e-3,
            steps_per_epoch=100,
            epochs=10,
            pct_start=0.3,
            anneal_strategy="cos",
            final_div_factor=10,
        )

        return (
            [g_optimizer, d_optimizer],
            [
                {"scheduler": g_scheduler, "interval": "step"},
                {"scheduler": d_scheduler, "interval": "step"},
            ],
        )


class GANDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()


# Our Malware Classifier using Convolutional Neural Networks


if __name__ == "__main__":

    # Dummy dataset
    batch_size = 16  # 64 samples
    seq_len = 10  # my sequence will have 10 words / tokens / api calls / system calls
    vocab_size = 20  # unique tokens in your entire dataset
    noise_dim = 32  # noise dimenions

    # fake data (normally you have real sequences encoded somehow)
    total_samples = 64
    data = torch.randint(
        0, vocab_size, (total_samples, seq_len)
    )  # [500 samples, seq_len tokens]

    print("Data = ", data)

    print("shape of my data : ", data.shape)

    # one-hot encode

    data_onehot = nn.functional.one_hot(
        data, num_classes=vocab_size
    ).float()  # feature represetation

    dataset = TensorDataset(data_onehot)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Model
    MAX_STEPS = 5  # number of iterations to train our GAN

    gan = GAN(
        noise_dim,
        hidden_dim=128,
        seq_len=seq_len,
        vocab_size=vocab_size,
        max_steps=MAX_STEPS,
    )

    # z = gan.sample_noise(batch_size)

    # print("noise smaples =", z.shape)

    # Trainer
    trainer = pl.Trainer(
        max_steps=MAX_STEPS,  # number of iterations: usually up to 10k
        log_every_n_steps=2,
        accelerator="auto",  # mps/cuda/cpu
    )
    trainer.fit(gan, dataloader)

    # # # Save generator model
    torch.save(gan.G.state_dict(), "generator.pt")

    # # # Save discriminator model
    torch.save(gan.D.state_dict(), "discriminator.pt")


# train CNN model


# Dummy data
# x = torch.randint(0, 100, (200, 50))  # (batch_size, seq_len)
# y = torch.randint(0, 2, (200,))  # (batch_size,) # 0 or  1
# dataset = TensorDataset(x, y)
