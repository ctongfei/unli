from typing import *
import torch
import numpy as np
from abc import ABC, abstractmethod
from unli.modules.token_embedders.embedding_with_mask_output import *


class SentencePairFeatureExtractor(ABC, torch.nn.Module):

    @abstractmethod
    def forward(self, l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        :param l: LongTensor[Batch, Word]
        :param r: LongTensor[Batch, Word]
        :return: FloatTensor[Batch, Feature]
        """
        pass

    @abstractmethod
    def forward_2(self, l: torch.Tensor, r1: torch.Tensor, r0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param l: LongTensor[Batch, Word]
        :param r: LongTensor[Batch, Word]
        :return: (FloatTensor[Batch, Feature], FloatTensor[Batch, Feature])
        """

    @abstractmethod
    def forward_multi_candidate(self, l: torch.Tensor, rs: torch.Tensor) -> torch.Tensor:
        """
        :param l: LongTensor[Batch, Word]
        :param rs: LongTensor[Batch, Candidate, Word]
        :return: FloatTensor[Batch, Candidate, Feature]
        """
        pass

    @abstractmethod
    def forward_multi_candidate_2(self,
                                  l: torch.Tensor, r1s: torch.Tensor, r0s: torch.Tensor
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param l: LongTensor[Batch, Word]
        :param r1s: LongTensor[Batch, PosCand, Word]
        :param r0s: LongTensor[Batch, NegCand, Word]
        :return: (FloatTensor[Batch, PosCand, Feature], FloatTensor[Batch, NegCand, Feature]
        """
        pass
