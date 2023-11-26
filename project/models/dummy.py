__all__ = ("FashionMNISTClassifier",)

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from typing_extensions import Self


class FashionMNISTClassifier(LightningModule):
    def __init__(
        self: Self,
        in_channels: int = 1,
        out_channels: int = 10,
        learning_rate: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(6400, out_channels),
            nn.Softmax(dim=1),
        )
        self.num_classes = out_channels

        self.train_step_losses: list[float] = []
        self.train_epoch_losees: list[float] = []

        self.valid_step_losses: list[float] = []
        self.valid_epoch_losees: list[float] = []

        self.lr = learning_rate
        self.betas = (beta1, beta2)

    def forward(self: Self, x: Tensor) -> Tensor:
        return self.network.forward(x)  # type: ignore[no-any-return]

    def configure_optimizers(self: Self) -> Optimizer:
        return Adam(self.network.parameters(), lr=self.lr, betas=self.betas)

    def parse_batch(self: Self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        images, labels = batch
        labels = F.one_hot(labels, num_classes=self.num_classes)
        labels = labels.type(torch.FloatTensor).to(self.device)
        return images, labels

    def calc_loss(self: Self, batch: tuple[Tensor, Tensor]) -> Tensor:
        images, labels = self.parse_batch(batch)
        return F.binary_cross_entropy(self.forward(images), labels)

    def training_step(self: Self, batch: tuple[Tensor, Tensor], _: int) -> Tensor:
        loss = self.calc_loss(batch)
        self.train_step_losses.append(loss.item())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_start(self: Self) -> None:
        self.train_step_losses = []

    def on_train_epoch_end(self: Self) -> None:
        epoch_loss = sum(self.train_step_losses) / len(self.train_step_losses)
        self.train_epoch_losees.append(epoch_loss)

    def validation_step(self: Self, batch: tuple[Tensor, Tensor], _: int) -> Tensor:
        loss = self.calc_loss(batch)
        self.valid_step_losses.append(loss.item())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self: Self) -> None:
        self.valid_step_losses = []

    def on_validation_epoch_end(self: Self) -> None:
        epoch_loss = sum(self.valid_step_losses) / len(self.valid_step_losses)
        self.valid_epoch_losees.append(epoch_loss)

    def test_step(self: Self, batch: tuple[Tensor, Tensor], _: int) -> Tensor:
        loss = self.calc_loss(batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss
