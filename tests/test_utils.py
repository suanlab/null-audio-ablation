"""Unit tests for common utilities."""

from __future__ import annotations

import torch.nn as nn

from videollm.utils import count_parameters, format_param_count, freeze_module, unfreeze_module


def test_count_parameters_trainable() -> None:
    """count_parameters should count only trainable parameters."""
    model = nn.Linear(10, 20)  # 10*20 + 20 = 220 params
    assert count_parameters(model, trainable_only=True) == 220


def test_count_parameters_all() -> None:
    """count_parameters(trainable_only=False) should count all parameters."""
    model = nn.Linear(10, 20)
    model.weight.requires_grad = False  # freeze weight
    all_count = count_parameters(model, trainable_only=False)
    trainable_count = count_parameters(model, trainable_only=True)
    assert all_count == 220
    assert trainable_count == 20  # only bias is trainable


def test_format_param_count_billions() -> None:
    """Billions should be formatted as 'X.XB'."""
    assert format_param_count(7_200_000_000) == "7.2B"


def test_format_param_count_millions() -> None:
    """Millions should be formatted as 'X.XM'."""
    assert format_param_count(350_000_000) == "350.0M"


def test_format_param_count_thousands() -> None:
    """Thousands should be formatted as 'X.XK'."""
    assert format_param_count(1_500) == "1.5K"


def test_format_param_count_small() -> None:
    """Small counts should be returned as plain string."""
    assert format_param_count(42) == "42"


def test_freeze_module() -> None:
    """freeze_module should set requires_grad=False on all parameters."""
    model = nn.Linear(10, 20)
    freeze_module(model)
    for param in model.parameters():
        assert not param.requires_grad


def test_unfreeze_module() -> None:
    """unfreeze_module should set requires_grad=True on all parameters."""
    model = nn.Linear(10, 20)
    freeze_module(model)
    unfreeze_module(model)
    for param in model.parameters():
        assert param.requires_grad
