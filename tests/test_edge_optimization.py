"""
Unit tests for edge optimization: pruning and quantization modules.
"""

import os
import torch
import torch.nn as nn
import pytest
from models.pruning import prune_model, calculate_model_sparsity


class SimpleTestConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


def test_pruning_sparsity():
    net = SimpleTestConvNet()
    total_p, zero_p, spar_init = calculate_model_sparsity(net)
    assert spar_init == 0.0

    # Apply 30% pruning
    res = prune_model(net, amount=0.30, prune_type="l1_unstructured", output_path=None)
    assert res["sparsity_after_pct"] > 25.0
    assert res["zero_parameters_after"] > 0


def test_quantization_import():
    import onnxruntime
    from onnxruntime.quantization import quantize_dynamic
    assert quantize_dynamic is not None
