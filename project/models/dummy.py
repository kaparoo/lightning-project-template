# -*- coding: utf-8 -*-

__all__ = ("FashionMNISTClassifier",)

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torch.optim import Adam


class FashionMNISTClassifier(LightningModule):
    def __init__(
        self,
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

    def forward(self, x: Tensor) -> Tensor:
        return self.network.forward(x)  # type: ignore

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        beta1 = self.hparams.beta1
        beta2 = self.hparams.beta2
        return Adam(self.network.parameters(), lr=lr, betas=(beta1, beta2))

    def parse_batch(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        images, labels = batch
        labels = F.one_hot(labels, num_classes=self.num_classes)
        labels = labels.type(torch.FloatTensor).to(self.device)
        return images, labels
    
    def calc_loss(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        images, labels = self.parse_batch(batch)
        return F.binary_cross_entropy(self(images), labels)


    def training_step(self, batch: tuple[Tensor, Tensor], _: int) -> Tensor:
        loss = self.calc_loss(batch)
        self.train_step_losses.append(loss.item())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_start(self) -> None:
        self.train_step_losses = []

    def on_train_epoch_end(self) -> None:
        epoch_loss = sum(self.train_step_losses) / len(self.train_step_losses)
        self.train_epoch_losees.append(epoch_loss)

    def validation_step(self, batch: tuple[Tensor, Tensor], _: int) -> Tensor:
        loss = self.calc_loss(batch)
        self.valid_step_losses.append(loss.item())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.valid_step_losses = []

    def on_validation_epoch_end(self) -> None:
        epoch_loss = sum(self.valid_step_losses) / len(self.valid_step_losses)
        self.valid_epoch_losees.append(epoch_loss)

    def test_step(self,  batch: tuple[Tensor, Tensor], _: int) -> Tensor:
        loss = self.calc_loss(batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss


