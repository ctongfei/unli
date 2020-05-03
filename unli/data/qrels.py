from typing import *
from abc import *
import numpy as np
from itertools import groupby
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers import WordTokenizer, Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import TextField, LabelField, MetadataField, ListField

from unli.data.fields import RealField
from unli.data.storage import StringStringStorage


class QRelsReader(ABC, DatasetReader):
    """
    Reads data in the TREC format.
    A dataset of this form comprises of 3 files, with different
    extensions. Here the default is `.l`, `.r` and `.qrels`.

    A `.l` (left) file could be either questions in QA or premises in NLI.
    A `.r` (right) file could be either answer candidates in QA or hypotheses in NLI.
    These files take the form of a TSV with rows [id, text].

    The third file, the `.qrels` file, is a TSV file with rows of format
    [leftId, ITER, rightId, relevanceLabel]. This file should be directly usable
    with the standard evaluation tool `trec_eval`.
    """
    def __init__(self,
                 lazy: bool = False,
                 left_ext: str = "l",
                 right_ext: str = "r",
                 rel_ext: str = "qrels",
                 label_type: str = "float",
                 left_tokenizer: Tokenizer = None,
                 right_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ):

        super().__init__(lazy)
        self._left_tokenizer = left_tokenizer or WordTokenizer()
        self._right_tokenizer = right_tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._left_ext = left_ext
        self._right_ext = right_ext
        self._rel_ext = rel_ext
        self._label_type = {
            "int": int,
            "float": float
        }[label_type]

    def label_field(self, value: str):
        return {
            int: lambda x: LabelField(int(x), skip_indexing=True),
            float: lambda x: RealField(float(x))
        }[self._label_type](value)

    def _read_file(self, path: str, right: bool) -> Dict[str, List[Token]]:
        """
        Reads a TSV file with rows of form [id, text].
        """
        m = {}
        tokenizer = self._right_tokenizer if right else self._left_tokenizer
        with open(path, 'r') as f:
            for l in f:
                sid, text = l.strip('\n').split("\t")
                if text == "":
                    text = "."
                m[sid] = tokenizer.tokenize(text)
        return m

    def _read_pairs(self, file_path: str) -> Tuple[
        Dict[str, List[Token]],
        Dict[str, List[Token]],
        Callable[[], Iterator[Tuple[str, str, Union[str, int]]]]
    ]:
        l_path = f"{file_path}.{self._left_ext}"
        r_path = f"{file_path}.{self._right_ext}"
        qrels_path = f"{file_path}.{self._rel_ext}"

        ls = self._read_file(l_path, right=False)
        rs = self._read_file(r_path, right=True)

        def pairs():
            with open(qrels_path, 'r') as f:
                for l in f:
                    lid, _, rid, y = l.strip().split("\t")
                    if y == "?":
                        continue
                    yield lid, rid, self._label_type(y)

        return ls, rs, pairs


class QRelsPointwiseReader(QRelsReader):

    def __init__(self,
                 lazy: bool = False,
                 left_ext: str = "l",
                 right_ext: str = "r",
                 rel_ext: str = "qrels",
                 label_type: str = "float",
                 left_tokenizer: Tokenizer = None,
                 right_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None
                 ):
        super(QRelsPointwiseReader, self).__init__(
            lazy, left_ext, right_ext, rel_ext,  label_type, left_tokenizer, right_tokenizer, token_indexers
        )
        self.mode = {
            "int": "pointwise-classification",
            "float": "pointwise-regression"
        }[label_type]

    def _read(self, file_path: str) -> Iterable[Instance]:
        ls, rs, pairs = self._read_pairs(file_path)

        for lid, rid, y in pairs():
            if y == "?":
                continue
            yield Instance({
                "mode": MetadataField(self.mode),
                "lid": MetadataField(lid),
                "rid": MetadataField(rid),
                "l": TextField(ls[lid], self._token_indexers),
                "r": TextField(rs[rid], self._token_indexers),
                "y": self.label_field(y)
            })

    def text_to_instance(self, *inputs) -> Instance:
        pass


class QRelsPairwiseReader(QRelsReader):

    def __init__(self,
                 lazy: bool = False,
                 left_ext: str = "l",
                 right_ext: str = "r",
                 rel_ext: str = "qrels",
                 label_type: str = "int",
                 left_tokenizer: Tokenizer = None,
                 right_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ):
        super(QRelsPairwiseReader, self).__init__(
            lazy, left_ext, right_ext, rel_ext, label_type, left_tokenizer, right_tokenizer, token_indexers
        )

    def _read(self, file_path: str) -> Iterable[Instance]:
        ls, rs, pairs = self._read_pairs(file_path)

        for lid, l_rs_gen in groupby(pairs(), key=lambda t: t[0]):  # t is a triple (lid, rid, label)
            l_rs: List[Tuple[str, str, float]] = list(l_rs_gen)  # R's relevant to this L
            l_rs.sort(key=lambda t: t[2], reverse=True)  # t[2] is the label
            for i in range(0, len(l_rs) - 1):
                y1 = l_rs[i][2]
                y0 = l_rs[i + 1][2]
                if y1 > y0:
                    r1id = l_rs[i][1]
                    r0id = l_rs[i + 1][1]

                    yield Instance({
                        "mode": MetadataField("pairwise"),
                        "lid": MetadataField(lid),
                        "r0id": MetadataField(r0id),
                        "r1id": MetadataField(r1id),
                        "l": TextField(ls[lid], self._token_indexers),
                        "r0": TextField(rs[r0id], self._token_indexers),
                        "r1": TextField(rs[r1id], self._token_indexers),
                        "y1": RealField(y1),
                        "y0": RealField(y0)
                    })


class QRelsListwiseReader(QRelsReader):

    def __init__(self,
                 lazy: bool = False,
                 left_ext: str = "l",
                 right_ext: str = "r",
                 rel_ext: str = "qrels",
                 label_type: str = "float",
                 skip_queries_with_no_relevant_candidate: bool = True,
                 left_tokenizer: Tokenizer = None,
                 right_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None
                 ):
        super(QRelsListwiseReader, self).__init__(
            lazy, left_ext, right_ext, rel_ext, label_type, left_tokenizer, right_tokenizer, token_indexers
        )
        self.skip_queries_with_no_relevant_candidate = skip_queries_with_no_relevant_candidate

    def _read(self, file_path: str) -> Iterable[Instance]:
        ls, rs, pairs = self._read_pairs(file_path)

        for lid, l_rs_gen in groupby(pairs(), key=lambda t: t[0]):
            l_rs: List[Tuple[str, str, float]] = list(l_rs_gen)
            l_rs.sort(key=lambda t: t[2], reverse=True)
            if l_rs[0][2] < 1 and self.skip_queries_with_no_relevant_candidate:
                continue
            else:
                yield Instance({
                    "mode": MetadataField("listwise"),
                    "lid": MetadataField(lid),
                    "rids": ListField([MetadataField(rid) for _, rid, _ in l_rs]),
                    "l": TextField(ls[lid], self._token_indexers),
                    "rs": ListField([
                        TextField(rs[rid], self._token_indexers)
                        for _, rid, _ in l_rs
                    ]),
                    "ys": ListField([RealField(y) for _, _, y in l_rs])
                })


class QRelsPairListwiseReader(QRelsReader):

    def __init__(self,
                 num_groups: int = 2,
                 lazy: bool = True,
                 left_ext: str = "l",
                 right_ext: str = "r",
                 rel_ext: str = "qrels",
                 left_tokenizer: Tokenizer = None,
                 right_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None
                 ):
        super(QRelsPairListwiseReader, self).__init__(
            lazy, left_ext, right_ext, rel_ext, "int", left_tokenizer, right_tokenizer, token_indexers
        )
        self.num_groups = num_groups

    def _read(self, file_path: str) -> Iterable[Instance]:
        ls, rs, pairs = self._read_pairs(file_path)

        for lid, l_rs_gen in groupby(pairs(), key=lambda t: t[0]):  # t is a triple (lid, rid, label)
            l_rs: List[Tuple[str, str, str]] = list(l_rs_gen)  # R's relevant to this L
            l_rs.sort(key=lambda t: t[2], reverse=True)  # t[2] is the label
            rid_groups_dict: Dict[int, List[str]] = {
                rel: [t[1] for t in rel_group]  # t[1] is the rid
                for rel, rel_group in groupby(l_rs, key=lambda t: t[2])
            }
            if len(rid_groups_dict) == 2:  # has positive candidates and negative candidates!
                assert rid_groups_dict[1] is not None
                assert rid_groups_dict[0] is not None
                rid_groups: List[List[str]] = [
                    rid_groups_dict[i]
                    for i in range(self.num_groups)
                ]
                instance_fields = {
                    "mode": MetadataField("pair-listwise"),
                    "lid": MetadataField(lid),
                    "l": TextField(ls[lid], self._token_indexers)
                }
                for i in range(self.num_groups):
                    if len(rid_groups[i]) != 0:
                        instance_fields[f"r{i}ids"] = ListField([
                            MetadataField(rid) for rid in rid_groups[i]
                        ])
                        instance_fields[f"r{i}s"] = ListField([
                            TextField(rs[rid], self._token_indexers) for rid in rid_groups[i]
                        ])
                    else:  # fit AllenNLP's stupid empty_field
                        instance_fields[f"r{i}ids"] = ListField([MetadataField(None)]).empty_field()
                        instance_fields[f"r{i}s"] = ListField([TextField([], self._token_indexers)]).empty_field()

                yield Instance(instance_fields)

            else:  # either lacks positive candidates or negative candidates
                pass  # ignored in one-vs-many style training


class QRelsRandomSampleListwiseReader(QRelsReader):

    def __init__(self,
                 k1: float,
                 b: float,
                 avg_doc_len: float,
                 idf: str,
                 corpus: str,
                 num_neg_samples: int = 500,
                 num_top: int = 5,
                 num_queries_to_resample: int = 20,
                 left_ext: str = "l",
                 right_ext: str = "r",
                 rel_ext: str = "qrels",
                 left_tokenizer: Tokenizer = None,
                 right_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None
                 ):
        super(QRelsRandomSampleListwiseReader, self).__init__(
            lazy=True,
            left_ext=left_ext,
            right_ext=right_ext,
            rel_ext=rel_ext,
            label_type="int",
            left_tokenizer=left_tokenizer,
            right_tokenizer=right_tokenizer,
            token_indexers=token_indexers
        )
        self.k1 = k1
        self.b = b
        self.avg_doc_len = avg_doc_len
        self.num_queries_to_resample = num_queries_to_resample

        self.idf = {}
        for l in open(idf):
            t, w = l.strip('\n').split("\t")
            self.idf[t] = float(w)

        self.corpus = StringStringStorage.open(corpus)

        self.num_neg_samples = num_neg_samples
        self.num_top = num_top

        self.n = len(self.corpus)
        self.all_keys = [k for k, _ in self.corpus.items()]

        self.negative_samples = {}
        self.iteration = 0

    @staticmethod
    def to_dict(x: List[str]) -> Dict[str, int]:
        m = {}
        for t in x:
            if t not in m:
                m[t] = 0
            m[t] += 1
        return m

    def bm25(self, q: Dict[str, int], raw_d: str):
        tokenized_d = [t.lower() for t in raw_d.split(" ")]
        d = self.to_dict(tokenized_d)
        len_d = len(tokenized_d)
        s = 0.0
        for t, wq in q.items():
            t = t.lower()
            wd = self.idf.get(t, 0.0) * d.get(t, 0.0) * (self.k1 + 1) / \
                 (d.get(t, 0.0) + self.k1 * (1.0 - self.b + self.b * len_d / self.avg_doc_len))
            s += wq * wd
        return s

    def sample(self):
        indices = np.random.randint(self.n, size=self.num_neg_samples)
        keys = [self.all_keys[i] for i in indices]
        self.negative_samples: Dict[str, str] = {k: self.corpus[k] for k in keys}

    def _read(self, file_path: str) -> Iterable[Instance]:
        ls, rs, pairs = self._read_pairs(file_path)

        for lid, l_rs_gen in groupby(pairs(), key=lambda t: t[0]):  # t is a triple (lid, rid, label)
            pos_rs: List[Tuple[str, str, float]] = list(lry for lry in l_rs_gen if lry[2] == 1)

            if self.iteration % self.num_queries_to_resample:
                self.sample()
            self.iteration += 1

            q = ls[lid]
            q_dict = self.to_dict([t.text for t in q])

            pos_rids = set(rid for _, rid, _ in pos_rs)
            neg_rs = [(rid, r) for rid, r in self.negative_samples.items() if rid not in pos_rids]
            neg_rs.sort(key=lambda p: -self.bm25(q_dict, p[1]))  # p[1] is the raw text

            for _, r1id, _ in pos_rs:
                for r0id, r0_text in neg_rs[:self.num_top]:
                    r0_tokenized = self._right_tokenizer.tokenize(r0_text)
                    yield Instance({
                        "mode": MetadataField("pairwise"),
                        "lid": MetadataField(lid),
                        "r0id": MetadataField(r0id),
                        "r1id": MetadataField(r1id),
                        "l": TextField(ls[lid], self._token_indexers),
                        "r0": TextField(r0_tokenized, self._token_indexers),
                        "r1": TextField(rs[r1id], self._token_indexers),
                        "y1": RealField(1.0),
                        "y0": RealField(0.0)
                    })

