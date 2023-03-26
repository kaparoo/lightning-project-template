# -*- coding: utf-8 -*-

__all__ = ("MNISTClassifier",)


import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torch.optim import Adam


class MNISTClassifier(LightningModule):
    def __init__(
        self, 
        in_channels: int = 1, 
        out_channels: int = 10, 
        learning_rate: float = 0.0002, 
        beta1: float=0.5, 
        beta2: float=0.999
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

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def training_step(self, batch: tuple[Tensor, Tensor], _: int):
        images, labels = batch
        preds = self(images)
        return F.binary_cross_entropy(preds, labels)
    
    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        beta1 = self.hparams.beta1
        beta2 = self.hparams.beta2
        return Adam(self.network.parameters(), lr=lr, betas=(beta1, beta2))