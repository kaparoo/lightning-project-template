__all__ = ("FashionMNISTClassifier",)

import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torchmetrics.functional.classification import multiclass_accuracy
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
        )
        self.num_classes = out_channels
        self.lr = learning_rate
        self.betas = (beta1, beta2)

    def forward(self: Self, x: Tensor) -> Tensor:
        return self.network.forward(x)  # type: ignore[no-any-return]

    def configure_optimizers(self: Self) -> Optimizer:
        return Adam(self.network.parameters(), lr=self.lr, betas=self.betas)

    def _common_step(self: Self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        images, labels = batch
        logits = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        accuracy = multiclass_accuracy(logits, labels, num_classes=self.num_classes)
        return loss, accuracy

    def training_step(self: Self, batch: tuple[Tensor, Tensor], _: int) -> Tensor:
        loss, accuracy = self._common_step(batch)
        self.log_dict({"train_loss": loss, "train_acc": accuracy}, prog_bar=True)
        return loss

    def validation_step(self: Self, batch: tuple[Tensor, Tensor], _: int) -> Tensor:
        loss, accuracy = self._common_step(batch)
        self.log_dict({"valid_loss": loss, "valid_acc": accuracy}, prog_bar=True)
        return loss

    def test_step(self: Self, batch: tuple[Tensor, Tensor], _: int) -> Tensor:
        loss, accuracy = self._common_step(batch)
        self.log_dict({"test_loss": loss, "test_acc": accuracy}, prog_bar=True)
        return loss
