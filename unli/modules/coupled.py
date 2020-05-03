from typing import Tuple

import torch
from unli.modules import Joiner, BERTConcatenator, BertSeq2VecEncoderForPairs
from unli.modules.sentence_pair_feature_extractor import SentencePairFeatureExtractor
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder


class CoupledSentencePairFeatureExtractor(SentencePairFeatureExtractor):

    def __init__(self,
                 joiner: Joiner,
                 encoder: Seq2VecEncoder,
                 l_token_index_field: str = "wordpiece",
                 r_token_index_field: str = "wordpiece"
                 ):
        super(CoupledSentencePairFeatureExtractor, self).__init__()
        self.joiner = joiner
        self.encoder = encoder
        self.l_token_index_field = l_token_index_field
        self.r_token_index_field = r_token_index_field

    @classmethod
    def from_params(cls, vocab, params):
        return cls(
            joiner={
                "bert_concat": lambda: BERTConcatenator()
            }[params["joiner"]](),
            encoder=BertSeq2VecEncoderForPairs.from_params(params)
        )

    def forward(self, l, r) -> torch.Tensor:
        joined = self.joiner(l, r)
        encoded = self.encoder(joined)
        return encoded

    def forward_2(self, l: torch.Tensor, r1: torch.Tensor, r0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(l, r1), self.forward(l, r0)

    def forward_multi_candidate(self, l, rs) -> torch.Tensor:
        batch_size = l.size(0)
        cand_size = rs.size(1)
        max_len = l.size(1)
        l_expanded = l.unsqueeze(dim=1).expand(batch_size, cand_size, max_len).contiguous()
        lx = l_expanded.view(batch_size * cand_size, max_len)
        rx = rs.view(batch_size * cand_size, -1)
        xx = self.forward(lx, rx)
        return xx.view(batch_size, cand_size, -1)

    def forward_multi_candidate_2(self, l: torch.Tensor, r1s: torch.Tensor, r0s: torch.Tensor):
        return self.forward_multi_candidate(l, r1s), self.forward_multi_candidate(l, r0s)


