"""
  Contains classes for working with the standard trec_eval utility.
"""
__author__ = "Tongfei Chen"

from typing import *


class TrecEvalRefItem:
    """
    Adapted from my previous code at
    https://github.com/ctongfei/omnivore/blob/master/src/main/scala/me/tongfei/omnivore/treceval/TrecEvalReferenceFile.scala.
    """

    def __init__(self,
                 query_id: str,
                 doc_id: str,
                 relevance: Union[int, float]):
        self.query_id = query_id
        self.doc_id = doc_id
        self.relevance = relevance

    def __repr__(self) -> str:
        return f"{self.query_id}\tITER\t{self.doc_id}\t{self.relevance}"

    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def parse(s: str) -> "TrecEvalRefItem":
        query_id, _, doc_id, relevance = s.split()
        return TrecEvalRefItem(query_id, doc_id, float(relevance))


class TrecEvalResItem:
    """
    Adapted from my previous code at
    https://github.com/ctongfei/omnivore/blob/master/src/main/scala/me/tongfei/omnivore/treceval/TrecEvalResultFile.scala
    """

    def __init__(self,
                 query_id: str,
                 doc_id: str,
                 rank: int,
                 sim: Union[float, int]):
        self.query_id = query_id
        self.doc_id = doc_id
        self.rank = rank
        self.sim = sim

    def __repr__(self) -> str:
        return f"{self.query_id}\tITER\t{self.doc_id}\t{self.rank}\t{self.sim}\tRUN_ID"

    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def parse(s: str) -> "TrecEvalResItem":
        query_id, _, doc_id, rank, sim, _ = s.split()
        return TrecEvalResItem(query_id, doc_id, rank, sim)
