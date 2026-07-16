"""SimCLR encoder loader.

Cleaned from the original `model.py` at the repo root. The reference SimCLR
class is preserved verbatim for state_dict compatibility with the trained
checkpoint `aurora-fm-no-finetune.tar`.

Public surface:
    load_simclr(checkpoint_path, device="cuda") -> nn.Module
        Returns the ResNet-18 encoder (fc replaced by Identity), in eval()
        mode, on the requested device. Output shape: (B, 512).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torchvision

PROJECTION_DIM = 64
N_FEATURES = 512


class Identity(nn.Module):
    def forward(self, x):
        return x


class SimCLR(nn.Module):
    """SimCLR wrapper used during training. We instantiate it only to load
    the published state_dict with strict=True (validates key parity), then
    return `model.encoder` for inference. The projector is unused at query
    time."""

    def __init__(self, encoder: nn.Module, projection_dim: int, n_features: int):
        super().__init__()
        self.encoder = encoder
        self.n_features = n_features
        self.encoder.fc = Identity()
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )


def load_simclr(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
) -> nn.Module:
    """Load the SimCLR encoder from a training checkpoint.

    The returned module accepts (B, 3, 224, 224) float32 tensors in [0,1]
    and produces (B, 512) representations.
    """
    encoder = torchvision.models.resnet18(weights=None)
    n_features = encoder.fc.in_features  # 512
    assert n_features == N_FEATURES, f"unexpected encoder feature dim {n_features}"

    model = SimCLR(encoder=encoder, projection_dim=PROJECTION_DIM, n_features=n_features)
    state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model.encoder
