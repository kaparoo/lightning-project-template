# -*- coding: utf-8 -*-

import random

import pytest
import torch

from project.models.dummy import MNISTClassifier


@pytest.fixture()
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture()
def random_input(device):
    batch_size = random.randint(1, 64)
    return torch.randn(batch_size, 1, 28, 28, device=device)

def test_init(device):
    MNISTClassifier().to(device)

def test_forward(device, random_input):
    model = MNISTClassifier().to(device)
    y_hat = model(random_input)
    assert y_hat.shape == torch.Size([len(random_input), model.hparams.out_channels])