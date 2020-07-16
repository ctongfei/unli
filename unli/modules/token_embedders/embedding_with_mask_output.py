from typing import *
import torch
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.elmo import Elmo
from abc import ABC, abstractmethod
import transformers


class EmbeddingWithMaskOutput(ABC, torch.nn.Module):

    def __init__(self):
        super(EmbeddingWithMaskOutput, self).__init__()

    @classmethod
    @abstractmethod
    def from_params(cls, vocab, params):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: LongTensor[Batch, Word]  OR  LongTensor[Batch, Word, Char]
        :return: [0: embs]: FloatTensor[Batch, Word, Emb];  [1: masks]: LongTensor[Batch, Word]
        """
        pass


class UncontextualizedEmbedding(EmbeddingWithMaskOutput):

    def __init__(self, embedding: Embedding):
        super(UncontextualizedEmbedding, self).__init__()
        self.embedding = embedding

    @classmethod
    def from_params(cls, vocab, params):
        return UncontextualizedEmbedding(
            embedding=Embedding.from_params(vocab, params)
        )

    def forward(self,
                x: torch.Tensor  # L[Batch, Word]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = x > 0  # B[Batch, Word]
        embs = self.embedding(x)
        return embs, mask


class WrappedELMo(EmbeddingWithMaskOutput):

    def __init__(self, elmo: Elmo):
        super(WrappedELMo, self).__init__()
        self.elmo = elmo

    @classmethod
    def from_params(cls, vocab, params):
        elmo = Elmo(
            options_file=params["options_file"],
            weight_file=params["weight_file"],
            num_output_representations=params.get("num_output_representations", 1)
        )  # somehow Elmo.from_params did not work
        return WrappedELMo(elmo)

    def forward(self,
                x: torch.Tensor,  # L[Batch, Word, Char=50]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        elmo_out = self.elmo(x)
        elmo_repr = elmo_out["elmo_representations"][0]  # num_elmo_representations = 1
        elmo_mask = elmo_out["mask"].byte()
        return elmo_repr, elmo_mask


class WrappedBERT(transformers.BertPreTrainedModel, EmbeddingWithMaskOutput):

    def __init__(self, config: transformers.BertConfig):
        super(WrappedBERT, self).__init__(config)
        self.bert = transformers.BertModel(config)

    @classmethod
    def from_params(cls, vocab, params):
        return cls.from_pretrained(params["pretrained_bert_model_name"])

    def forward(self,
                x: torch.Tensor  # F[Batch, WordPieceToken]
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        mask = x > 0
        emb, pooled = self.bert(
            input_ids=x,
            attention_mask=mask,
            output_hidden_states=False
        )
        return emb, mask
