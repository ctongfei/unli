from typing import *
import torch
import numpy as np
import pdb
import pytorch_transformers as pt
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder


class BertGetClsEncoder(Seq2VecEncoder):

    def __init__(self, dim: int = 768):
        super(BertGetClsEncoder, self).__init__()
        self.dense = torch.nn.Linear(dim, dim)
        self.activation = torch.nn.Tanh()

    def forward(self,
                word_embs: torch.Tensor,  # Float[Batch, Word, Embedding]
                mask: torch.Tensor  # Byte[Batch, Word]
                ) -> torch.Tensor:  # Float[Batch, Embedding]

        first_token_tensor = word_embs[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    def get_input_dim(self) -> int:
        pass

    def get_output_dim(self) -> int:
        pass
