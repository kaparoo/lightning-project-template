__all__ = ("FashionMNISTDataModule",)

import os

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from typing_extensions import Self

_BATCH_SIZE: int = 256 if torch.cuda.is_available() else 64
_NUM_WORKERS: int = num_cpus // 2 if isinstance(num_cpus := os.cpu_count(), int) else 0


class FashionMNISTDataModule(LightningDataModule):
    def __init__(
        self: Self,
        root: os.PathLike,
        batch_size: int = _BATCH_SIZE,
        num_workers: int = _NUM_WORKERS,
    ) -> None:
        super().__init__()
        self.root = str(root)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        FashionMNIST(self.root, train=True, download=True)
        FashionMNIST(self.root, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            data_total = FashionMNIST(self.root, train=True, transform=self.transform)
            self.data_train, self.data_valid = random_split(data_total, [55000, 5000])
        if stage == "test" or stage is None:
            self.data_test = FashionMNIST(
                self.root, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
