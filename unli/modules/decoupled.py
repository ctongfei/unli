from typing import *
import torch
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from unli.modules.token_embedders import EmbeddingWithMaskOutput, UncontextualizedEmbedding, WrappedELMo, WrappedBERT
from unli.modules import BiLSTMMaxPoolingEncoder, BertGetClsEncoder, InferSentJoiner, InnerProductJoiner, \
    CosineJoiner, WeightedInnerProductJoiner, SentencePairFeatureExtractor


class DecoupledSentencePairFeatureExtractor(SentencePairFeatureExtractor):

    def __init__(self,
                 l_embedding: EmbeddingWithMaskOutput,
                 l_token_index_field: str,
                 r_embedding: EmbeddingWithMaskOutput,
                 r_token_index_field: str,
                 l_encoder: Seq2VecEncoder,
                 r_encoder: Seq2VecEncoder,
                 joiner: torch.nn.Module
                 ):
        super(DecoupledSentencePairFeatureExtractor, self).__init__()
        self.l_embedding = l_embedding
        self.l_token_index_field = l_token_index_field
        self.r_embedding = r_embedding
        self.r_token_index_field = r_token_index_field
        self.l_encoder = l_encoder
        self.r_encoder = r_encoder
        self.joiner = joiner

    @classmethod
    def from_params(cls, vocab, params):

        l_embedding = {
            "uncontextualized": lambda: UncontextualizedEmbedding.from_params(vocab, params=params["l_embedding"]),
            "elmo": lambda: WrappedELMo.from_params(vocab, params["l_embedding"]),
            "bert": lambda: WrappedBERT.from_params(vocab, params["l_embedding"])
        }[params["l_embedding_type"]]()

        l_token_index_field = {
            "uncontextualized": "tokens",
            "elmo": "elmo_characters",
            "bert": "wordpiece"
        }[params["l_embedding_type"]]

        r_embedding = {
            "uncontextualized": lambda: UncontextualizedEmbedding.from_params(vocab, params=params["r_embedding"]),
            "elmo": lambda: WrappedELMo.from_params(vocab, params["r_embedding"]),
            "bert": lambda: WrappedBERT.from_params(vocab, params["r_embedding"]),
            "tied": lambda: l_embedding
        }[params["r_embedding_type"]]()

        r_token_index_field = {
            "uncontextualized": "tokens",
            "elmo": "elmo_characters",
            "bert": "wordpiece",
            "tied": l_token_index_field
        }[params["r_embedding_type"]]

        l_encoder = {
            "bilstmmax": lambda: BiLSTMMaxPoolingEncoder.from_params(params=params["l_encoder"]),
            "bert-cls": lambda: BertGetClsEncoder()
        }[params["l_encoder_type"]]()

        r_encoder = {
            "bilstmmax": lambda: BiLSTMMaxPoolingEncoder.from_params(params=params["r_encoder"]),
            "bert-cls": lambda: BertGetClsEncoder(),
            "tied": lambda: l_encoder
        }[params["r_encoder_type"]]()

        joiner = {
            "infersent": lambda: InferSentJoiner(),
            "inner-product": lambda: InnerProductJoiner(),
            "cosine": lambda: CosineJoiner(),
            "weighted-inner-product": lambda: WeightedInnerProductJoiner(dim=params["wips"]["dim"])
        }[params["joiner"]]()

        return DecoupledSentencePairFeatureExtractor(
            l_embedding=l_embedding,
            l_token_index_field=l_token_index_field,
            r_embedding=r_embedding,
            r_token_index_field=r_token_index_field,
            l_encoder=l_encoder,
            r_encoder=r_encoder,
            joiner=joiner
        )

    def embed(self, s: torch.LongTensor, right: bool = True) -> torch.Tensor:
        if right:
            embedded, mask = self.r_embedding(s)
            encoded = self.r_encoder(embedded, mask)
        else:
            embedded, mask = self.l_embedding(s)
            encoded = self.l_encoder(embedded, mask)
        return encoded

    def forward(self, l: torch.LongTensor, r: torch.LongTensor) -> torch.Tensor:
        l_embedded, l_mask = self.l_embedding(l)  # F[Batch, Word, Embedding], B[Batch, Word]
        r_embedded, r_mask = self.r_embedding(r)  # F[Batch, Word, Embedding], B[Batch, Word]

        l_encoded = self.l_encoder(l_embedded, l_mask, True)  # F[Batch, Feature]
        r_encoded = self.r_encoder(r_embedded, r_mask)  # F[Batch, Feature]
        joined = self.joiner(l_encoded, r_encoded)  # F[Batch, Feature]
        return joined

    def forward_2(self, l: torch.Tensor, r1: torch.Tensor, r0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l_embedded, l_mask = self.l_embedding(l)  # F[Batch, Word, Embedding], B[Batch, Word]
        r1_embedded, r1_mask = self.r_embedding(r1)  # F[Batch, Word, Embedding], B[Batch, Word]
        r0_embedded, r0_mask = self.r_embedding(r0)  # F[Batch, Word, Embedding], B[Batch, Word]

        l_encoded = self.l_encoder(l_embedded, l_mask, True)  # F[Batch, Feature]
        r1_encoded = self.r_encoder(r1_embedded, r1_mask)  # F[Batch, Feature]
        r0_encoded = self.r_encoder(r0_embedded, r0_mask)  # F[Batch, Feature]

        joined1 = self.joiner(l_encoded, r1_encoded)  # F[Batch, Feature]
        joined0 = self.joiner(l_encoded, r0_encoded)  # F[Batch, Feature]
        return joined1, joined0

    def forward_multi_candidate(self, l: torch.LongTensor, rs: torch.LongTensor) -> torch.Tensor:
        batch_size = rs.size(0)
        cand_size = rs.size(1)

        if rs.ndimension() == 3:
            rx = rs.view(batch_size * cand_size, -1)  # L[Batch*Cand, Word]
        elif rs.ndimension() == 4:
            rx = rs.view(batch_size * cand_size, rs.size(2), rs.size(3))  # L[Batch*Cand, Word, Char]

        l_embedded, l_mask = self.l_embedding(l)  # F[Batch, Word, Embedding], B[Batch, Word]
        rx_embedded, rx_mask = self.r_embedding(rx)  # F[Batch*Cand, Word, Embedding], B[Batch*Cand, Word]

        l_encoded = self.l_encoder(l_embedded, l_mask, True)  # F[Batch, Feature]
        rx_encoded = self.r_encoder(rx_embedded, rx_mask)  # F[Batch*Cand, Feature]
        lx_encoded = l_encoded.unsqueeze(dim=1) \
            .expand(batch_size, cand_size, -1) \
            .view(batch_size * cand_size, -1)  # F[Batch*Cand, Feature]

        x_joined = self.joiner(lx_encoded, rx_encoded)  # F[Batch*Cand, Feature]
        joined = x_joined.view(batch_size, cand_size, -1)  # F[Batch, Cand, Feature]
        return joined

    def forward_multi_candidate_2(self,
                                  l: torch.Tensor, r1s: torch.Tensor, r0s: torch.Tensor
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = l.size(0)
        pos_cand_size = r1s.size(1)
        neg_cand_size = r0s.size(1)

        if r0s.ndimension() == 3:
            r1x = r1s.view(batch_size * pos_cand_size, -1)  # L[Batch*PosCand, Word]
            r0x = r0s.view(batch_size * neg_cand_size, -1)  # L[Batch*NegCand, Word]
        elif r0s.ndimension() == 4:
            char_size = r1s.size(3)
            r1x = r1s.view(batch_size * pos_cand_size, -1, char_size)  # L[Batch*PosCand, Word, Char]
            r0x = r0s.view(batch_size * neg_cand_size, -1, char_size)  # L[Batch*NegCand, Word, Char]
        else:
            raise Exception("Bad dimensionality in sampled tensor.")

        l_embedded, l_mask = self.l_embedding(l)  # F[Batch, Word, Embedding], # B[Batch, Word]
        r1x_embedded, r1x_mask = self.r_embedding(r1x)  # F[Batch*PosCand, Word, Embedding], B[Batch*PosCand, Word]
        r0x_embedded, r0x_mask = self.r_embedding(r0x)  # F[Batch*NegCand, Word, Embedding], B[Batch*NegCand, Word]

        l_encoded = self.l_encoder(l_embedded, l_mask, True)  # F[Batch, Feature]
        r1x_encoded = self.r_encoder(r1x_embedded, r1x_mask)  # F[Batch*PosCand, Feature]
        r0x_encoded = self.r_encoder(r0x_embedded, r0x_mask)  # F[Batch*NegCand, Feature]

        l1x_encoded = l_encoded.unsqueeze(dim=1) \
            .expand(batch_size, pos_cand_size, -1) \
            .contiguous().view(batch_size * pos_cand_size, -1)  # F[Batch*PosCand, Feature]
        l0x_encoded = l_encoded.unsqueeze(dim=1) \
            .expand(batch_size, neg_cand_size, -1) \
            .contiguous().view(batch_size * neg_cand_size, -1)  # F[Batch*NegCand, Feature]

        joined_x1 = self.joiner(l1x_encoded, r1x_encoded)  # F[Batch*PosCand, Feature]
        joined_x0 = self.joiner(l0x_encoded, r0x_encoded)  # F[Batch*NegCand, Feature]

        joined_1 = joined_x1.view(batch_size, pos_cand_size, -1)  # F[Batch, PosCand, Feature]
        joined_0 = joined_x0.view(batch_size, neg_cand_size, -1)  # F[Batch, NegCand, Feature]

        return joined_1, joined_0
