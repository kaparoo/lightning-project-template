# -*- coding: utf-8 -*-

from lightning.pytorch.cli import LightningCLI

from project.datasets.dummy import FashionMNISTDataModule
from project.models.dummy import FashionMNISTClassifier

if __name__ == "__main__":
    LightningCLI(FashionMNISTClassifier, FashionMNISTDataModule)
