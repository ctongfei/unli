import torch
from typing import *
from unli.modules.joiner import Joiner


class BERTConcatenator(Joiner):

    def __init__(self, max_wordpiece_len: int = 128):
        super(BERTConcatenator, self).__init__()
        self.max_wordpiece_len = max_wordpiece_len

    @classmethod
    def from_params(cls, params):
        return cls(params.get("max_wordpiece_len", 128))

    def forward(self, l: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param l: LongTensor[Batch, WordPiece]
        :param r: LongTensor[Batch, WordPiece]
        :return: input_ids, token_type_ids, attention_mask
        """
        batch_size = l.size(0)

        r = r[:, 1:]  # remove [CLS] prefixing R
        l_mask = l > 0  # B[Batch, WordPiece]
        r_mask = r > 0  # B[Batch, WordPiece]
        l_lengths = l_mask.sum(dim=1).long()  # L[Batch]
        r_lengths = r_mask.sum(dim=1).long()  # L[Batch]
        l_maxlen = l_lengths.max().item()
        r_maxlen = r_lengths.max().item()

        #c_len = l_maxlen + r_maxlen
        c_len = max(self.max_wordpiece_len, l_maxlen + r_maxlen)

        s = torch.LongTensor(batch_size, c_len).to(l.device).zero_()
        t = torch.LongTensor(batch_size, c_len).to(l.device).zero_()
        m = torch.LongTensor(batch_size, c_len).to(l.device).zero_()

        for i in range(batch_size):
            l_len = l_lengths[i].item()
            r_len = r_lengths[i].item()
            s[i, 0:l_len] = l[i, 0:l_len]
            t[i, 0:l_len] = 0
            s[i, l_len:l_len + r_len] = r[i, 0:r_len]
            t[i, l_len:l_len + r_len] = 1
            m[i, 0:l_len + r_len] = 1

        s = s[:, :self.max_wordpiece_len]
        t = t[:, :self.max_wordpiece_len]
        m = m[:, :self.max_wordpiece_len]

        s.requires_grad = False
        t.requires_grad = False
        m.requires_grad = False

        return s, t, m
