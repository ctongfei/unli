from allennlp.data.tokenizers import Token, Tokenizer
from typing import *
import transformers


class BertTokenizer(Tokenizer):

    def __init__(self,
                 pretrained_model_name: str = "bert-base-uncased",
                 prefix: str = "[CLS]",
                 suffix: str = "[SEP]"):
        self.underlying = transformers.BertTokenizer.from_pretrained(pretrained_model_name)
        self.prefix = prefix
        self.suffix = suffix

    def tokenize(self, text: str) -> List[Token]:
        tokens = self.underlying.tokenize(text)
        if self.prefix is not None:
            tokens = [self.prefix] + tokens
        if self.suffix is not None:
            tokens = tokens + [self.suffix]
        return [Token(t) for t in tokens]

    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(s) for s in texts]
