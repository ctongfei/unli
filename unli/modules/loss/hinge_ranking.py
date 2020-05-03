from typing import *
import torch


class MaxTripletLoss(torch.nn.Module):  # TODO

    def __init__(self, margin: float):
        super(MaxTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, s_pos: torch.Tensor, s_neg: torch.Tensor) -> torch.Tensor:
        """
        Computes the max triplet loss.
        :param s_pos: F[Batch, PosCand, Output=1]
        :param s_neg: F[Batch, NegCand, Output=1]
        :return: F[]
        """
        s1 = s_pos.squeeze(dim=2)  # F[Batch, PosCand]
        s0 = s_neg.squeeze(dim=2)  # F[Batch, NegCand]

        s0_best, _ = torch.max(s0, dim=1)  # F[Batch]
        s0_best_expanded = s0_best.unsqueeze(dim=1).expand(-1, s1.size(1))  # F[Batch, PosCand]

        diff = s0_best_expanded - s1 + self.margin  # F[Batch, PosCand]
        return torch.mean(torch.relu(diff))


class MeanTripletLoss(torch.nn.Module):

    def __init__(self, margin: float):
        super(MeanTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, s_pos: torch.Tensor, s_neg: torch.Tensor) -> torch.Tensor:
        #  y_pos: torch.Tensor, y_neg: torch.Tensor
        """
        Computes the mean triplet loss.
        :param s_pos: F[Batch, PosCand, Output=1]
        :param s_neg: F[Batch, NegCand, Output=1]
        :return: F[]
        """
        s1 = s_pos.squeeze(dim=2)  # F[Batch, PosCand]
        s0 = s_neg.squeeze(dim=2)  # F[Batch, NegCand]

        s1e = s1.unsqueeze(dim=2).expand(-1, -1, s0.size(1))  # F[Batch, PosCand, NegCand]
        s0e = s0.unsqueeze(dim=1).expand(-1, s1.size(1), -1)  # F[Batch, PosCand, NegCand]
        #y1e = y_pos.unsqueeze(dim=2).expand(-1, -1, s0.size(1))
        #y0e = y_neg.unsqueeze(dim=1).expand(-1, s1.size(1), -1)

        diff = s0e - s1e + self.margin  # F[Batch, PosCand, NegCand]
        return torch.mean(torch.relu(diff))


class PairwiseHingeLoss(torch.nn.Module):

    def __init__(self, margin: float):
        super(PairwiseHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, s_pos: torch.Tensor, s_neg: torch.Tensor, y_pos: torch.Tensor, y_neg: torch.Tensor) -> torch.Tensor:
        """
        :param s_pos: F[Batch, Output=1]
        :param s_neg: F[Batch, Output=1]
        :return:
        """
        diff = s_neg - s_pos + self.margin * (y_pos - y_neg)  # F[Batch, Score]
        return torch.mean(torch.relu(diff))
