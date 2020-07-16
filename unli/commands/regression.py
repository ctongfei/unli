import argparse
from unli.models import SentencePairModel
from unli.modules import CoupledSentencePairFeatureExtractor, BERTConcatenator, BertSeq2VecEncoderForPairs, MLP
from unli.data.qrels import QRelsPointwiseReader
from unli.data.tokenizers import BertTokenizer
from allennlp.common import Params
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.iterators import BasicIterator
from allennlp.training import Trainer
from allennlp.data import Vocabulary
import torch
import logging
from torch.optim import Adam


parser = argparse.ArgumentParser(description="")
parser.add_argument("--data", type=str, default="", help="Path to QRels data")
parser.add_argument("--pretrained", type=str, default="", help="Pretrained model")
parser.add_argument("--out", type=str, default="", help="Output path")
parser.add_argument("--margin", type=float, default=0.3, help="")
parser.add_argument("--num_samples", type=int, default=1, help="")
parser.add_argument("--seed", type=int, default=0xCAFEBABE, help="")
parser.add_argument("--batch_size", type=int, default=16, help="")
parser.add_argument("--gpuid", type=int, default=0)
ARGS = parser.parse_args()

logging.basicConfig(level=logging.INFO)
torch.manual_seed(ARGS.seed)
vocab = Vocabulary()

model: torch.nn.Module = SentencePairModel(
    extractor=CoupledSentencePairFeatureExtractor(
        joiner=BERTConcatenator(),
        encoder=BertSeq2VecEncoderForPairs.from_pretrained("bert-base-uncased")
    ),
    mlp=torch.nn.Sequential(
        torch.nn.Linear(768, 1),
        torch.nn.Sigmoid()
    ),
    loss_func=torch.nn.BCELoss(),
    mode="regression"
)
model.cuda()

if ARGS.pretrained != "":
    model.load_state_dict(torch.load(ARGS.pretrained))

reader = QRelsPointwiseReader(
    lazy=True,
    token_indexers={"wordpiece": PretrainedTransformerIndexer("bert-base-uncased", do_lowercase=True)},
    left_tokenizer=BertTokenizer(),
    right_tokenizer=BertTokenizer()
)
iterator = BasicIterator(batch_size=ARGS.batch_size)
iterator.index_with(vocab)


trainer = Trainer(
    model=model,
    optimizer=Adam(params=model.parameters(), lr=0.00001),
    grad_norm=1.0,
    train_dataset=reader.read(f"{ARGS.data}/train"),
    validation_dataset=reader.read(f"{ARGS.data}/dev"),
    iterator=iterator,
    validation_metric="+pearson",
    num_epochs=3,
    patience=3,
    serialization_dir=ARGS.out,
    cuda_device=ARGS.gpuid
)

trainer.train()
