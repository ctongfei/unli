from typing import *

import torch
import numpy as np
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_mask_from_sequence_lengths


class BiLSTMMaxPoolingEncoder(Seq2VecEncoder):
    """
    Bi-LSTM then max-pooling sentence encoder.
    This is the preferred architecture in InferSent (Conneau et al., 2017).
    """

    def __init__(self,
                 embedding_dim: int,
                 output_dim: int,
                 num_layers: int = 1,
                 with_linear_transform: bool = False,
                 dropout_rate: float = 0.0
    ):
        super(BiLSTMMaxPoolingEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._output_dim = output_dim
        self._num_layers = num_layers
        self._dropout_rate = dropout_rate

        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=output_dim // 2,
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True
        )

        self.with_linear_transform = with_linear_transform

        if with_linear_transform:
            self.linear = torch.nn.Linear(self._output_dim, self._output_dim)
        self.final_dropout = torch.nn.Dropout(self._dropout_rate)

    @classmethod
    def from_params(cls, params):
        return BiLSTMMaxPoolingEncoder(
            embedding_dim=params["input_size"],
            output_dim=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout_rate=params["dropout"],
            with_linear_transform=params.get("with_linear_transform", False)
        )

    def get_input_dim(self) -> int:
        return self._embedding_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self,
                word_embs: torch.Tensor,  # Float[Batch, Word, Embedding]
                mask: torch.Tensor,  # Byte[Batch, Word]
                left: bool = False,
                ) -> torch.Tensor:  # Float[Batch, Embedding]

        device = word_embs.device

        lengths = mask.long().sum(dim=1).cpu().numpy()  # Long[Batch]
        sorted_lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)  # sort descendingly w.r.t. length of sequence
        idx_unsort = np.argsort(idx_sort)  # get inverse permutation

        x_sorted = word_embs.index_select(
            0,
            torch.from_numpy(idx_sort).to(device=device)
        )  # Float[Batch, Word, Embedding]

        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            x_sorted,
            lengths=sorted_lengths.copy(),
            batch_first=True
        )

        y_output, _ = self.lstm(x_packed)
        y_output, _ = torch.nn.utils.rnn.pad_packed_sequence(y_output, batch_first=True)

        y_unsorted = y_output.index_select(
            0,
            torch.from_numpy(idx_unsort).to(device=device)
        )  # Float[Batch, Word, Encoding]

        y_unsorted_inf = torch.where(
            get_mask_from_sequence_lengths(
                torch.tensor(lengths).to(device=device),
                max_length=y_unsorted.size(1)
            ).unsqueeze(dim=2).expand(-1, -1, y_unsorted.size(2)),
            y_unsorted,
            torch.ones_like(y_unsorted) * float('-inf')
        )
        pooled, _ = torch.max(y_unsorted_inf, dim=1)

        output = self.final_dropout(pooled)
        if self.with_linear_transform and left:
            output = self.linear(output)

        return output
