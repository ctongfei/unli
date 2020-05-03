from typing import *
import numpy as np
import torch
import torch.nn.functional as F
from abc import *


class Joiner(ABC, torch.nn.Module):

    def __init__(self):
        super(Joiner, self).__init__()

    @abstractmethod
    def forward(self, l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        :param l: encodings of the left sentences: F[Batch, Encoding]
        :param r: encodings of the right sentences: F[Batch, Encoding]
        :return: F[Batch, Features]
        """
        pass


class CosineJoiner(Joiner):

    def __init__(self):
        super(CosineJoiner, self).__init__()

    def forward(self, l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        sim = F.cosine_similarity(l, r, dim=1)
        return sim.unsqueeze(dim=1)  # F[Batch, Feature=1]


class InferSentJoiner(Joiner):

    def __init__(self):
        super(InferSentJoiner, self).__init__()

    def forward(self, l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        p = l * r
        d = torch.abs(l - r)
        joined = torch.cat([l, r, p, d], dim=-1)
        return joined


class InnerProductJoiner(Joiner):

    def __init__(self):
        super(InnerProductJoiner, self).__init__()

    def forward(self, l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return (l * r).sum(dim=1, keepdim=True)


class WeightedInnerProductJoiner(Joiner):

    def __init__(self, dim: int):
        super(WeightedInnerProductJoiner, self).__init__()
        self.weight = torch.nn.Linear(dim, 1, bias=False)
        self.weight.weight.data.normal_(1.0, 1.0)  # default 1.0 mean (back to IPS)

    def forward(self, l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return self.weight(l * r)
