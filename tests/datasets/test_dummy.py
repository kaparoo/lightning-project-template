# -*- coding: utf-8 -*-

import os

import dotenv
import pytest

from project.datasets.dummy import FashionMNISTDataModule


# You have to create .env file and assign DATASETS_ROOT to any path/to/datasets/
@pytest.fixture()
def datasets_root():
    dotenv.load_dotenv()
    return os.environ.get("DATASETS_ROOT", ".")


def test_init(datasets_root):
    FashionMNISTDataModule(datasets_root)


def test_prepare_data(datasets_root):
    datamodule = FashionMNISTDataModule(datasets_root)
    datamodule.prepare_data()


def test_setup(datasets_root):
    datamodule = FashionMNISTDataModule(datasets_root)
    datamodule.prepare_data()
    for stage in ("fit", "validate", "test", "predict", None):
        datamodule.setup(stage)
