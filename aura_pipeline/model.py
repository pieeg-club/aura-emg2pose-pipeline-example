from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AuraNet(nn.Module):
    """Temporal-spatial EMG classifier.

    Input is (batch, time, channels). Dropped channels are zeroed by a mask stored in the graph.
    """

    def __init__(
        self,
        n_classes: int = 6,
        n_channels: int = 8,
        n_samples: int = 250,
        dropout: float = 0.3,
        channel_mask: list[bool] | None = None,
    ):
        super().__init__()
        f1, depth = 8, 2
        f2 = f1 * depth
        self.n_samples = n_samples
        self.n_channels = n_channels
        mask = channel_mask if channel_mask is not None else [True] * n_channels
        self.register_buffer(
            "channel_mask",
            torch.tensor([1.0 if k else 0.0 for k in mask], dtype=torch.float32),
        )
        self.temporal = nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial = nn.Conv2d(f1, f2, (n_channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.sep_dw = nn.Conv2d(f2, f2, (1, 16), padding=(0, 8), groups=f2, bias=False)
        self.sep_pw = nn.Conv2d(f2, f2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        feat = self._probe(n_samples, n_channels)
        self.head = nn.Linear(feat, n_classes)
        self._reset_bn()

    def _reset_bn(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()

    def _probe(self, n_samples: int, n_channels: int) -> int:
        was = self.training
        self.eval()
        with torch.no_grad():
            y = self.features(torch.zeros(1, n_samples, n_channels))
        self.train(was)
        if y.shape[-1] < 1:
            raise ValueError("Window is too short for this network. Use at least 1 second at 250 Hz.")
        return int(y.shape[-1])

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_mask.view(1, 1, -1)
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.bn1(self.temporal(x))
        x = F.elu(self.bn2(self.spatial(x)))
        x = self.drop1(self.pool1(x))
        x = F.elu(self.bn3(self.sep_pw(self.sep_dw(x))))
        x = self.drop2(self.pool2(x))
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ProbModel(nn.Module):
    """Wraps AuraNet so the exported output is class probabilities, not logits."""

    def __init__(self, net: AuraNet):
        super().__init__()
        self.net = net

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(emg), dim=-1)
