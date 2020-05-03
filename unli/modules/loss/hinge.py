import torch
from typing import *
import torch.nn.functional as F


class HingeLoss(torch.nn.Module):

    def __init__(self, margin: float = 1.0):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, y_pred: torch.Tensor, y_gold: torch.Tensor) -> torch.Tensor:
        t = y_gold.type_as(y_pred)
        t = (t - 0.5) * 2  # {0, 1} -> {-1, 1}
        h = -(y_pred * t) + 1
        return F.relu(h).mean()

    @classmethod
    def from_params(cls, params):
        margin = params["margin"]
        return cls(margin)
